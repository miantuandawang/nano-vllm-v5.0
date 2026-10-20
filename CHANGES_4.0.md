# Nano-vLLM 4.0 修改说明

日期：2026-10-13

## 1. 版本与范围

- 来源：WSL `Ubuntu-22.04-D` 中的 `/home/zhj/decode/nano-vllm-2.0`。
- 新副本：`/home/zhj/decode/nano-vllm-4.0`。
- 本次只在 4.0 中修改 draft CUDA Graph 路径、相关配置及文档。
- 2.0 保留；4.0 中的 `nanovllm/native/` 与复制时的 2.0 相同，不改动 K=0 原生路径。
- 没有运行测试、编译检查、模型推理或 benchmark；没有下载模型或安装依赖。

## 2. 执行路径

| 模式 | 执行方式 |
| --- | --- |
| K=0 | 复制自 2.0 的原生 nano-vllm TP=2 路径 |
| K>0 target | 原有 eager prefill / verification |
| K>0 draft 首次 prefill | eager 输入 prompt + target 首个 token，产生候选 1 |
| K>0 draft 多 token 补算 | eager 补齐有效上下文，产生本轮首个候选 |
| K>0 draft 单 token 步骤 | bucket 向上匹配，然后 CUDA Graph replay |

一次调度仍最多生成 K 个候选，遇到 EOS（未设置 ignore_eos 时）或剩余输出额度耗尽就停止。
不是将整个 K 步循环捕获成一张图，而是在循环内部重复使用单步图。

当一个混合批次包含多 token 输入，当前这一步整体使用 eager，不拆成多个子批次。
后续所有有效请求都只缺一个 token 时再使用图。有效 batch 大于 512 时当前步骤使用 eager；
如果提前结束的请求退出后 batch 降到捕获范围内，后续步骤可以使用图。

## 3. 对照原生实现

原生参考文件：`nanovllm/native/engine/model_runner.py`。

沿用的机制：

1. 只捕获 `model(input_ids, positions)`，结果写入静态 hidden-state buffer。
2. LM head 和 greedy argmax 在图外执行，只处理真实请求行。
3. 从大 bucket 向小 bucket 捕获；各图共享一个 graph memory pool。
4. 输入、位置、slot、上下文长度和 block table 使用固定地址的 CUDA buffers。
5. 实际 batch 选择第一个大于或等于它的 bucket。
6. padding 行使用 `slot_mapping=-1`、`context_lens=0`，忽略其输出。

为投机执行增加的适配：

- 单 token 判断使用实际 `len(work_tokens) - num_cached_tokens`，不能只检查 is_prefill。
- KV slot 按本次绝对位置和 block table 计算，不取预留列表的最后一个 block。
- 每次 replay 清理整个 bucket 的 padding 行和 block table 区域，避免活动 batch 收缩后残留映射。
- 输出 buffer 明确使用模型 dtype 和 CUDA device。
- 图外保持 TP rank 0 的 argmax 结果广播，保证各 rank 下一步输入和活动 batch 一致。

## 4. Bucket 与 batch

```python
max_bs = min(config.max_num_seqs, 512)
buckets = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
graph_bs = sorted({bs for bs in buckets if bs <= max_bs} | {max_bs})
```

`EngineConfig.batch` 已约束模型的 max_num_seqs，因此不会为配置上限之外的 batch 捕获图。

| 配置上限 | 捕获 bucket |
| --- | --- |
| 6 | 1、2、4、6 |
| 24 | 1、2、4、8、16、24 |
| 64 | 1、2、4、8、16、32、48、64 |

例如上限 64 时，实际 batch=17 使用 bucket=32；若下一步剩 15 个请求，则改用 bucket=16。
padding 不创建 Sequence，不改变队列，也不增加调度请求数。

## 5. 字段与函数

修改文件：`nanovllm/engine/model_runner.py`。

新增字段：

- `use_decode_graph`：仅 draft 且 enforce_eager=False 时开启。
- `graph_bs`、`graphs`、`graph_pool`：分档、图实例和共享 pool。
- `graph_vars`：input_ids、positions、slot_mapping、context_lens、block_tables、outputs。

新增函数：

- `capture_cudagraph()`：创建 buffers、侧流预热、按 bucket 捕获。
- `prepare_decode()`：选择 bucket，并将当前请求输入复制到静态 buffers。
- `replay_decode_graph()`：replay 主干，图外执行 logits / argmax / TP 广播。

调整函数：

- 初始化：KV 分配完成后、worker ready 之前捕获图。
- `allocate_kv_cache()`：从 KV 容量预算扣除图内存余量。
- `run_draft_batch()`：每一步选择 graph 或 eager，沿用 EOS、输出额度和缓存长度更新。
- `exit()`：同步后释放 graphs、pool 引用和静态 buffers。

没有修改 Scheduler 队列、两侧 request_id 交接、Sequence 字段、验证接受规则或按轮分配 blocks 的方式。

## 6. KV 与显存预算

KV 张量的底层地址在图捕获后保持固定。请求的 block 分配、裁剪仍只调整映射。

固定大小的 `block_tables[max_bs, ceil(max_model_len / block_size)]` 只是 block ID 元数据，
不代表给每个请求分配最大长度的实际 KV blocks。

新增 `Config.graph_memory_reserve_mb`，默认 256 MiB，仅在 draft graph 启用时从 KV 容量预算扣除。
用于为静态 buffers 和 graph pool 留出空间；这是可调整的预算估计，不保证所有模型、batch 和设备都足够。
捕获失败直接报告错误，不静默关闭 graph，也不自动重试分配。

候选生成后的最后一个 token 仍未写入 KV；只有它作为下一次输入时才写入。
拒绝后的缓存裁剪、全部接受时的缺失 token 补算逻辑保持原样。

## 7. 配置与使用

默认 EngineConfig 自动创建的 draft 配置开启 graph；自定义 Config 沿用其显式 enforce_eager 设置。
需要手动创建 draft Config 时，通过 `enforce_eager=False` 开启。

```bash
cd /home/zhj/decode/nano-vllm-4.0

# draft decode graph，target verification eager
python example.py --k 5 --batch 24

# draft 全部 eager
python example.py --k 5 --batch 24 --draft-enforce-eager

# 增加 draft graph 预留显存
python example.py --k 5 --batch 24 --draft-graph-memory-reserve-mb 512

# 原生 TP=2 路径
python example.py --k 0 --batch 8
```

`--enforce-eager` 保持其原有含义：控制 K=0 原生路径；
K>0 的 target 始终 eager，draft 使用独立的 `--draft-enforce-eager`。

## 8. 限制与后续验证

本次仅完成代码修改和源码阅读，未验证运行结果或性能。
K 步循环仍在 CPU 上管理候选和 EOS，每一步仍将 greedy token 取回 CPU。
这一版主要减少模型主干逐层 kernel launch 的 CPU 开销，不声称消除了 CPU/GPU 同步。

后续运行验证应覆盖：非标准 batch 上限、bucket 边界、EOS 后缩 batch、K=1、
缓存跨 block、拒绝回滚、多 token 补算、graph/eager 输出对比，以及 graph 显存占用。
默认 draft TP=1；代码保留 TP 广播和统一 bucket 选择，多卡 draft 的捕获/collective 兼容性也尚未运行验证。
