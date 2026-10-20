# Nano-vLLM 4.1：target verify CUDA Graph 修改说明

日期：2026-10-13

## 1. 修改范围

本次在 WSL `Ubuntu-22.04-D` 的 `/home/zhj/decode/nano-vllm-4.1` 中，基于已经复制的 4.0 实现 target verify CUDA Graph。

主要修改文件：

| 文件 | 修改内容 |
| --- | --- |
| `nanovllm/engine/model_runner.py` | verify 分桶、静态 buffers、capture/replay、padding scratch KV、Graph/eager 分发、资源释放 |
| `nanovllm/config.py` | 新增 target verify 开关及 worker 所需 K；将 EngineConfig 的 K 传给 target |
| `model_paths.py` | 新增 target eager 开关和 Graph 显存预留参数 |
| `README.md` | 更新 4.1 执行路径、参数与使用方式 |
| `CHANGES_4.1.md` | 本说明 |

没有修改 4.0。4.1 中现有 draft Graph、Scheduler、Sequence、attention 算法和 native K=0 模块保留。注意：复制得到的 4.0 native 路径已经包含 greedy 适配，本次不重新覆盖它。

本次仅修改代码、阅读源码和比较文件差异。没有运行测试、语法/编译检查、模型推理或 benchmark，没有下载模型或安装依赖。不能据此声称运行正确性或加速幅度已得到验证。

## 2. 执行路径

| 输入类型 | 执行方式 |
| --- | --- |
| K=0 | 现有 native TP=2 路径 |
| target 首次 prefill | eager |
| target 完整 K 个候选、仅欠一个正式 token 的 KV | batch 向上匹配后 Graph replay |
| target 候选不足 K，例如 EOS/剩余额度导致提前停止 | eager |
| target 需要补算多个正式 token | eager |
| target 符合条件的 verify 请求数超过捕获上限 | 当前批次 eager |
| draft 单 token 步骤 | 4.0 已有的 draft Graph |
| draft prefill/多 token 补算 | 原有 eager |

target Graph 默认开启。匹配条件：

```python
not seq.is_prefill
len(seq.draft_tokens) == config.speculative_k
len(seq) - seq.num_cached_tokens == 1
```

请求长度仍由现有模型上下文和调度预算约束。若 `K+1` 超过上下文上限，或 token 预算无法容纳一个完整验证请求，不捕获无用途的 verify Graph，沿用 eager/现有预算处理。

## 3. 原生风格的 batch 分桶

每请求 query 长度为 `Q = K+1`，包括一个尚未缓存的正式 token 和 K 个候选。

```python
max_bs = min(config.max_num_seqs, 512,
             config.max_num_batched_tokens // (K + 1))
sizes = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
graph_bs = sorted({bs for bs in sizes if bs <= max_bs} | {max_bs})
```

`EngineConfig.batch` 已约束 max_num_seqs。每个 worker 的 K 在启动后固定，Graph 字典用 bucket B 索引；改变 K 需要重建引擎。

| 有效上限 | 捕获桶 |
| --- | --- |
| 6 | 1、2、4、6 |
| 16 | 1、2、4、8、16 |
| 24 | 1、2、4、8、16、24 |
| 64 | 1、2、4、8、16、32、48、64 |

例如 B=10、K=6、上限16：使用 B=16 的 Graph，前向处理112个位置，其中70个真实输入、42个padding位置。padding 不创建请求，不增加 scheduler 的 batch 计数。

## 4. 捕获边界

遵循原生 nano-vllm 的捕获边界：

```text
图外：整理 Sequence 数据，原地更新静态 CUDA buffers
图内：Embedding -> Transformer（含 attention、KV 写入、TP all-reduce）-> RMSNorm
图外：真实 hidden states -> LM head -> greedy argmax -> TP 广播 -> CPU token IDs
图外：原有接受判断、EOS/额度处理、KV 回滚、Scheduler 直接交接
```

LM head 每次最多处理128个位置，继续限制词表 logits 的瞬时占用。padding hidden states 不进入 LM head。

提取 `_greedy_predictions()` 供 eager 与 verify Graph 共用，原来在 `_predict()` 内的 logits/argmax/广播逻辑保持其计算方式。没有把 `.tolist()` 或 Python 接受循环放进 Graph。

## 5. 静态输入与 attention

继续使用现有 `store_kvcache` 和 `flash_attn_varlen_func(..., causal=True)`，没有新增 attention 分支，也没有改用 `flash_attn_with_kvcache()`。

静态 buffers：

| 名称 | 最大形状 |
| --- | --- |
| input_ids、positions、slot_mapping | `[max_bs * (K+1)]` |
| cu_seqlens_q、cu_seqlens_k | `[max_bs+1]` |
| block_tables | `[max_bs, ceil(max_model_len/block_size)]` |
| outputs | `[max_bs*(K+1), hidden_size]` |

cu_seqlens_q 固定为等间隔的 K+1；cu_seqlens_k 每轮按真实上下文长度更新。max_seqlen_q 固定为 K+1，max_seqlen_k 固定为配置的上下文上界。capture 与 replay 始终使用固定地址的张量。

每次 replay 清理整个所选 bucket 的 token、position、slot 和 block table 区域，防止 batch 收缩或请求切换后残留旧映射。CPU 数据先写入 pinned tensors，再复制到静态 GPU buffers。

## 6. Padding 与 scratch KV

在全局 KV 张量末尾保留 `ceil((K+1)/block_size)` 个专用 scratch blocks，并清零所有层的 K/V。

- 这些 block IDs 不属于 Scheduler 的可分配范围。
- dummy 请求 query 长度和 key 长度均设为 K+1，提供合法 attention 形状。
- dummy block table 指向 scratch blocks，多个dummy请求只读共享。
- dummy slot_mapping=-1，store_kvcache 不向任何 block 写入其 K/V。
- 真实请求只写入调度器实际分配的 slots。
- warmup/capture 的全部 dummy 行也使用这套 scratch 映射。

以 K=6、block_size=256 为例，每个 target rank 只保留一个 scratch block，不是每请求一个，也不是按最大上下文长度分配。

## 7. 调度和结果对应关系

Scheduler 继续先选 verify，再用剩余名额补 prefill；KV blocks 在下发前按本轮实际需要分配。

`run_target_batch()` 将已经选定的请求分为：

1. 满足固定 K+1 形状的 Graph verify；
2. 短草稿、多 token 补算、prefill 等 eager 请求。

先完成 Graph 子批次，再执行 eager 子批次，不为任何子批次重新补请求。预测结果按 request_id 合并，最终按原输入顺序执行同一套接受判断并返回。

例如调度了10个完整 verify、2个短草稿、4个prefill：真实请求仍为16；10个完整verify使用16桶，另外6个请求使用eager。两个子批次顺序执行，可能增加一次前向调用，实际端到端收益待测。

没有增加 Sequence 字段，没有修改 pausing/verify/drafting 状态转换和 draft/target 直接通信接口。

## 8. TP=2 与资源生命周期

两个 target rank 使用相同配置、输入顺序、子批次划分和 bucket 序列。

- 捕获按大桶到小桶进行；每个桶在侧流预热三次。
- 预热结束后同步，TP ranks 在 capture 前进行 barrier。
- 图内保留 Transformer 层本来的 collective。
- 各桶共享一个 verify graph pool，运行时顺序 replay。
- LM head 的词表 gather 和 greedy token broadcast 在图外执行。
- worker 的 ready 通知在 Graph 捕获完成之后发送。
- worker 原有同步/收齐所有rank结果机制继续保证回调前 GPU 工作已完成。
- exit 同步后清理 verify graphs、pool 引用和 buffers。

KV 张量分配后一直保持原地址，回滚只调整有效长度和映射，不重建 KV 张量。

## 9. 显存预算

复用 `Config.graph_memory_reserve_mb`，draft、target 各自独立配置。target verify 开启时，也会在计算 KV 池容量前扣除这一预算。

容量计算顺序：

```text
模型加载和 eager warmup
-> 按现有显存预算扣除 graph_memory_reserve_mb
-> TP ranks 取可用 block 数的最小值
-> 从该数量再扣除 scratch blocks，得到调度器可用数量
-> 一次分配完整 KV 张量并清零 scratch 区域
-> 分配静态 buffers，预热并捕获 Graph
```

默认预留256 MiB是配置预算，不是实测 Graph 占用，也不是保证足够。共享 pool 可复用桶之间的工作区，但真实占用受模型、B、K、attention、PyTorch/CUDA版本影响。

捕获失败会通过现有 worker 错误通道报告，不静默退回 eager。运行时不满足形状的请求正常走 eager，属于设计内的分发。

请求仍按本轮需要增配 blocks，没有预分配请求最大输出长度，没有引入抢占、换出或重算。资源全部阻塞时继续使用现有容量错误行为。

## 10. 配置与使用

新增/复用字段：

| 字段 | 含义 |
| --- | --- |
| Config.target_verify_graph=True | K>0 target verify Graph 开关 |
| Config.speculative_k | 由 EngineConfig.speculative_k 写入 target worker 配置 |
| Config.graph_memory_reserve_mb=256 | 本模型 Graph 的显存预算余量 |

K>0 target 的 enforce_eager 仍用于隔离旧配置语义；新的 target_verify_graph 是 verify 路径的开关。`--enforce-eager` 仍只控制 K=0 native；draft 仍使用已有独立开关。

```bash
cd /home/zhj/decode/nano-vllm-4.1

# draft decode graph + target verify graph
python example.py --k 6 --batch 16

# target verify eager，draft graph 保留
python example.py --k 6 --batch 16 --target-enforce-eager

# draft、target 均 eager
python example.py --k 6 --batch 16 --draft-enforce-eager --target-enforce-eager

# 增加 target Graph 的预算余量
python example.py --k 6 --batch 16 --target-graph-memory-reserve-mb 512

# 已有 native TP=2 路径
python example.py --k 0 --batch 16
```

Python：先创建 `config = EngineConfig(speculative_k=6, batch=16)`；需要关闭 target Graph 时，在创建 LLM 前设置 `config.target.target_verify_graph = False`。

新增函数：`_greedy_predictions()`、`_set_verify_graph_context()`、`capture_verify_graphs()`、`prepare_verify_graph_inputs()`、`replay_verify_graph()`。

## 11. 尚未运行验证的项目

本次没有执行任何测试、推理或 capture。因此以下事项仍待实际GPU环境验证：

- 当前安装 FlashAttention 的 paged varlen capture/replay 兼容性；固定 max_seqlen_k 上界配合动态 cu_seqlens_k 的行为。
- TP=2 collective 捕获及 replay。
- Graph/eager 的 logits/greedy、接受数量和最终输出一致性；不同执行形状可能带来浮点差异。
- bucket 边界、batch收缩、短草稿回退、跨block、拒绝回滚和上下文边界。
- Graph 实际显存增量，以及混合批次拆分后的端到端速度。

后续应比较同一套 K>0 逻辑下开启/关闭 target Graph 的结果；本说明不宣称已经测得加速。
