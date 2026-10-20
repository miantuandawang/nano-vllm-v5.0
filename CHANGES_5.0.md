# V5.0：A100 80GB 双模型同卡与固定 KV 显存预算

修改日期：2026-10-20。源码位置：`/home/zhj/decode/nano-vllm-5.0`。
基线为修改前的 V5.0（由本地 V4.2 完整复制）。Python 包版本更新为 `0.5.0`。

## 1. 最终配置

| 项目 | Target | Draft |
|---|---:|---:|
| 模型 | Qwen3-14B | Qwen3-0.6B |
| GPU | 物理 GPU 0 | 同一物理 GPU 0 |
| Tensor Parallel | 1 | 1 |
| BF16 权重估算 | 27.5081 GiB | 1.1102 GiB |
| 固定 KV 预算 | 21.25 GiB | 14.875 GiB |
| KV block 大小 | 256 token / 40 MiB | 256 token / 28 MiB |
| 总 block 数 | 544 | 544 |
| 总 token 槽位 | 139,264 | 139,264 |

KV 合计 36.125 GiB，比例为 10:7，权重和 KV 合计约 64.7433 GiB。
这些容量取决于当前模型配置与 BF16 dtype。代码读取实际模型配置计算块大小，
而不是把 544 写死到执行器里。

Target 开启 verify Graph 且 K=1～8 时，池内扣除一个 scratch block，
调度器可用 543 个 block / 139,008 token；关闭 verify Graph 时全部 544 块可用。
更多 scratch blocks 的情形按实际 K 计算。Draft 的 544 块全部可用于请求。
两模型分别拥有 KV 张量和 BlockManager，不共享 KV 内容。

默认 batch 仍为 8，K 仍为 5。Python Config 和 CLI 的
`max_num_batched_tokens` 默认值由 16384 改成 4096，以降低同卡 prefill 峰值；
这是参数默认值调整，选批算法没有改动，可以通过 CLI 覆盖。

## 2. 调度和通信约束

本次遵循“原有调度和通信方案不变”的要求，没有实现先前讨论的运行时串行仲裁、
联合准入、MPS、单进程合并或多 stream 重构。

以下文件与修改前逐字节相同：

- `nanovllm/engine/scheduler.py`
- `nanovllm/engine/draft_scheduler.py`
- `nanovllm/engine/target_scheduler.py`
- `nanovllm/engine/block_manager.py`
- `nanovllm/engine/sequence.py`

`LLMEngine.step()` 保持先收集 Draft/Target 结果，再执行
`target.schedule()`、`draft.schedule()`。两模型可以同时处于 pending 状态；
实际 GPU 执行由 CUDA/驱动安排，不假设同卡能复现三卡的执行重叠。
`add_request()`、`generate()`、`is_finished()` 同样未改动。

每模型一个 worker（TP=1），spawn、每 rank Pipe、独立 NCCL group、
ready/result/error/closed 消息及其 payload 格式均保留。
`ModelRunnerGroup` 的创建、等待、提交、轮询、关闭方法未改动。
worker 只在原有任务执行前后加入显存观察，整卡超限输出限频告警并继续执行；
实际分配器 OOM 等异常仍通过已有 error 消息报告。
原本返回结果前的 `torch.cuda.synchronize()` 保留。

唯一的启动时序变化：同卡时沿用 Draft→Target 的创建顺序，但每创建一个 worker
就等待其 ready，避免两模型同时加载、预热和捕获 Graph。该等待只发生在初始化。
独立 GPU 模式保留原本启动两组后再等待的方式。

## 3. 整卡 0.9 预算如何执行

新增 `nanovllm/utils/memory_budget.py`，所有容量内部使用整数 bytes。

1. 父进程使用当前 `sys.executable` 启动一个短命探测子进程，
   通过与 worker 相同的 `CUDA_VISIBLE_DEVICES` 映射选卡，
   调用 `torch.cuda.mem_get_info()` 获取 CUDA 可见总量和空闲量，返回 JSON。
   探测完成并退出后才创建模型 worker。父进程不创建 CUDA context，
   不改变原有模型 worker IPC，不再使用 `nvidia-smi` 总量做规划。
   探测失败明确报错，不回退到 NVML 总量。
2. 整卡预算为 `floor(CUDA 可见总显存 × device_memory_utilization)`，
   默认 0.9，配置层拒绝大于 0.9 的值。
3. 从预算中扣除探测时已有占用，以及默认 2 GiB 的非 PyTorch 余量。
   探测占用包含短命子进程自身的 context，退出后仍将这部分计入基线，
   这是保守记账，不尝试从估算值里扣除它。
   该余量用于 CUDA context、NCCL、库级分配等；它在 90% 内，而不是使用外面的 10%。
4. 按实际 Qwen3 配置估算权重存储，包括 embedding 是否绑定、QKV、MLP 和 norm。
   固定 KV 向下取整为完整 blocks。
5. 每模型先预留至少 1 GiB 的 PyTorch 运行空间；如果该模型配置的
   graph reserve 更大，则用较大的值。剩余空间按 Target:Draft=4:1 分给分配器。
   **4:1 只分配额外运行空间，KV 本身仍是 10:7。**
6. 为两个 worker 分别传入绝对 bytes 的 allocator cap，确保：

   ```text
   Target allocator cap + Draft allocator cap
   + 探测时已有占用（含探测 context）+ 非 PyTorch 余量
   ≤ CUDA 可见总显存 × 0.9
   ```

7. worker 在模型加载前调用 `torch.cuda.set_per_process_memory_fraction()`，
   用 cap / CUDA 实际总量设置 PyTorch native caching allocator 限制。
   不接受 `cudaMallocAsync` allocator backend。
8. 预热后检查“max(allocated 峰值, reserved 峰值) + 固定 KV + Graph reserve”
   是否在进程 cap 内，计入预热时观察到的分配器缓存/碎片。
   这是保守的容量检查，不保证未来所有分配形状都无碎片。
   Graph 私有池仍受该进程的分配器额度约束。
9. 在模型 ready 和推理任务前后观察整卡实际使用量，越过预算时打印 warning，
   不因此终止 worker 或当前生成。同一模型持续超限时每 30 秒最多告警一次；
   恢复后再次超限可立即告警。两进程的分配器额度不会因此放宽。
   启动预算不足、预热容量不足、真实 OOM 仍报错；不会偷偷缩小 KV、
   改模型精度或关闭 Graph。

仅作预算示例，假设 CUDA 可见容量 80 GiB、探测占用为零，权重与 KV 占用 64.7433 GiB，
90% 内剩余 7.2567 GiB。其中 2 GiB 用于非 PyTorch 余量，其余约
5.2567 GiB 属于两个分配器的运行空间。分配器 cap 约为
Target 52.363 GiB、Draft 17.637 GiB，合计 70 GiB。
CUDA 实际容量不同或探测占用不为零时会重新计算并打印日志。
上述零占用示例不是 A100 实测值。

这些是应用级预算和 PyTorch allocator 限制，不是硬件级的整卡隔离。
PyTorch API 不控制其他进程或所有 CUDA 库的直接分配；任务边界检查也不能
捕获每一个短暂峰值。因此，外部进程在启动后增加占用或非 PyTorch 分配超出预留，
仍可能使整卡超过 90%；告警模式不会因该指标超限自动停止生成。
需要在目标 A100 上实测最大工作负载，确认预留充分。
该限制依据 [PyTorch 2.8 API 文档](https://docs.pytorch.org/docs/2.8/generated/torch.cuda.memory.set_per_process_memory_fraction.html)。

## 4. 逐文件修改

| 文件 | 修改 |
|---|---|
| `nanovllm/config.py` | 默认两模型 GPU 0、TP=1；允许仅 TP=1 的完整共卡，拒绝 TP>1 的部分 GPU 重叠；添加固定 KV 和进程预算字段；同卡自动补齐 21.25/14.875 GiB；K=0 允许 TP=1；整卡比例最大 0.9 |
| `model_paths.py` | 添加固定 KV、整卡利用率和非 PyTorch 余量参数；默认 GPU IDs 改为 0；默认 token budget 改为 4096 |
| `nanovllm/utils/memory_budget.py` | 整数容量换算、Qwen3 权重估算、独立 CUDA 探测子进程、整卡预算、进程分配器限额、allocated/reserved 峰值检查及限频告警 |
| `nanovllm/engine/llm_engine.py` | 同卡启动前统一规划预算；初始化逐个等待 ready；K=0 使用通用 NativeTargetEngine；运行时 step 等方法保持不变 |
| `nanovllm/engine/model_runner.py` | 在加载模型前设置分配器 cap；固定预算按完整 block 分配；保留 scratch 扣除；加入预热容量检查和 KV 日志；模型 forward、Graph、验证逻辑不变 |
| `nanovllm/engine/runner_group.py` | 仅在 worker 任务前后观察整卡使用量，超限告警；不修改 Pipe 消息、提交与结果收集机制 |
| `nanovllm/engine/native_engine.py` | K=0 的 TP 大小由配置透传；单卡使用统一预算；透传固定 KV 和限额；旧名称 NativeTP2Engine 保留为兼容别名 |
| `nanovllm/native/config.py` | 接收 K=0 所需的固定 KV、Graph 余量和限额字段 |
| `nanovllm/native/engine/model_runner.py` | K=0 应用分配器限额和固定 KV；任务前后检查使用量；将 set_device 放在 NCCL 初始化前；保留原生调度、计算与共享内存 TP 通信 |
| `pyproject.toml` | 包版本更新为 0.5.0 |
| `README.md` | 更新默认设备、运行命令、预算说明和 K=0 描述 |
| `tests/test_single_gpu.py` | 无需 GPU/模型依赖的 25 项测试，含口径差异与碎片回归 |

未修改 attention kernel、权重加载器、greedy 规则、接受/拒绝算法、
回滚、队列优先级和请求 KV 释放机制。原有 batch 是单次派发上限，
不是新增的全局活跃请求上限；原有容量不足时的行为保留。

## 5. 参数语义与启动

新参数：

| 参数 | 默认/作用 |
|---|---|
| `--target-kv-cache-gib` | 同卡默认 21.25；单卡 K=0 也默认 21.25 |
| `--draft-kv-cache-gib` | 同卡默认 14.875 |
| `--gpu-memory-utilization` | 同卡和 K=0 TP=1 的整卡上限，默认 0.9，可降低，不能超过 0.9 |
| `--device-memory-reserve-gib` | 预算内的非 PyTorch 余量，默认 2.0，至少 1.0 |

`--target-gpu-memory-utilization` / `--draft-gpu-memory-utilization` 仍为旧的
按比例自动分配 KV 的参数。固定 KV 模式使用固定 bytes，不再用它们计算 KV。
整卡预算由新的 `--gpu-memory-utilization` 控制。

`--target-gpus` 和 `--draft-gpus` 延续原代码的物理 GPU ID 语义。
共卡模式从两组相同的 GPU IDs 自动识别，不需要额外的 `--shared-gpu-mode`。
独立 GPU 模式仍可显式指定 `--target-gpus 0 1 --draft-gpus 2`。
未指定固定 KV 参数时，该模式保留原自动 KV 分配，新的联合预算只针对同卡/
K=0 TP=1 生效。

在目标 A100 机器的已安装环境中：

```bash
cd /home/zhj/decode/nano-vllm-5.0

python example.py \
  --target-model-path /models/Qwen3-14B \
  --draft-model-path /models/Qwen3-0.6B \
  --target-gpus 0 --draft-gpus 0 \
  --target-kv-cache-gib 21.25 \
  --draft-kv-cache-gib 14.875 \
  --gpu-memory-utilization 0.9 \
  --device-memory-reserve-gib 2 \
  --batch 8 --k 4 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096
```

替换 `/models/...` 为实际目录。源码迁移不包含权重或服务器环境安装。
默认沿用 Draft decode Graph 和 Target verify Graph。
排查时添加 `--draft-enforce-eager --target-enforce-eager` 可使用原有 eager 开关。
不启用 MPS、不设置 GPU compute mode、不修改服务器级配置。

K=0 对照实验：

```bash
python example.py \
  --target-model-path /models/Qwen3-14B \
  --target-gpus 0 --target-kv-cache-gib 21.25 \
  --gpu-memory-utilization 0.9 \
  --batch 8 --k 0 \
  --max-model-len 4096 --max-num-batched-tokens 4096
```

K=0 不启动 Draft，使用单模型的整卡预算，KV 仍为 21.25 GiB。
其余可用空间属于运行余量，不自动扩大 KV。

碎片优化可在启动 Python 前选择启用：

```bash
export PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:True
```

这仍使用 native 后端，不强制写入代码或覆盖用户已有环境设置。
`expandable_segments` 是实验选项，主要帮助变化批大小导致的碎片；
仍需验证实际 PyTorch/CUDA 版本与 Graph 工作负载，不能保证消除 OOM。
参考 [PyTorch 2.8 内存管理说明](https://docs.pytorch.org/docs/2.8/notes/cuda.html#memory-management)。

## 6. 验证结果与尚未验证部分

执行：

```bash
python3 -m unittest discover -s tests -v
```

25 项 CPU-only 测试通过；47 个 Python 文件的编译检查通过。
另外对比修改前快照，确认上述 5 个调度/状态文件逐字节不变、
引擎运行与 RunnerGroup 的 11 个方法 AST 不变。V4.2 Python 源码未被修改。
测试覆盖：

- 模型权重公式、544 块、10:7 配比、scratch 扣除和非法预算；
- 实际总容量变化、其他 GPU 占用、预算不足和大于 0.9 的拒绝；
- 两进程 cap 之和、native allocator 要求、整卡超限和预热超限；
- NVML 81920 MiB / CUDA 81100、81200、81250 MiB 的模拟差异场景；
- 独立 Python 子进程探测、与 worker 一致的 GPU 映射、父进程无 CUDA 调用；
- 探测失败不回退 NVML、CUDA 设备实际变小时要求重新规划；
- 整卡超限只告警、告警限频/恢复，以及 allocated 能放下而 reserved 超限的碎片场景；
- 默认共卡、旧三卡布局、TP 部分重叠拒绝、CLI 覆盖和 K=0；
- 实际 KV 分配函数在模拟 CUDA 张量工厂上的固定形状；
- 初始化等待顺序、原有 step 同时派发两模型、原始 Pipe payload；
- 使用真实调度器代码执行接受、拒绝、回滚和最终 KV 块归还。

这些测试模拟 torch/transformers/GPU 执行，不代表真实模型已经运行。
本地 WSL 仅识别到 RTX 5060 Laptop 8GB，默认 Python 未安装 torch/transformers，
不能在本地运行 14B + 0.6B + 36.125 GiB KV 的目标配置。
尚未在 A100 验证模型加载、NCCL、FlashAttention、Graph 捕获、实际峰值或吞吐。

目标机器建议验证 batch=1/8/16、K=0/1/4/8，逐步覆盖 eager 和 Graph，
以及短 prompt、接近 4096 总长度、混合长度和多轮拒绝。
记录实际总显存、两个 cap、KV blocks 和运行峰值，确认整卡使用不超过 90%。
跨 GPU 架构、TP 大小和批形状的 token 差异需定位首个分歧，不能仅按逐 token
不一致就认定调度错误。新硬件性能以实测结果为准。

## 7. 复核修正记录

原实现将 NVML framebuffer 总量用于规划，再与 CUDA 总量做 1 MiB 容差比较，
混用了两种容量口径。现已移除该跨口径判断，统一使用 CUDA 探测值。
不把评审中给出的 A100 数值作为硬件常数；它们仅作为回归测试输入。
如果 CUDA 探测与 worker 的 CUDA 可见容量本身发生下降，仍要求重新启动并
统一规划两进程额度，避免仅调整后启动的一个 worker。

NVIDIA 的文档区分 framebuffer 的 total/reserved/used，并说明 ECC、
驱动预留等会影响报告；不能把任一驱动版本的数值视为固定不变。
参考 [nvidia-smi 文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html)。

本次复核继续保留双进程原调度。没有开启 MPS，也没有宣称同卡异步派发意味着
两模型计算实际重叠。后续在 A100 上应与相同 batch、长度和数据集的 K=0
比较端到端吞吐与延迟，确认 context 切换和 Draft 成本后的净收益。
