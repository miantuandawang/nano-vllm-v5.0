# Nano-vLLM 5.0: shared A100 GPU with fixed KV budgets

This copy is `/home/zhj/decode/nano-vllm-5.0`, based on V4.2.
Both K=0 and K>0 use greedy argmax sampling. Default placement: Qwen3-14B and
Qwen3-0.6B both on physical GPU 0, each TP=1. The intended device is a full A100
80GB. Local paths are in `nanovllm/config.py`; override them for your server.
See [CHANGES_5.0.md](CHANGES_5.0.md) for the detailed Chinese implementation report.

The default KV pools are target 21.25 GiB and draft 14.875 GiB (10:7). The combined
device budget is at most 0.9 of the CUDA-visible capacity. A short-lived subprocess
using the same Python interpreter and GPU visibility as the workers reads
`torch.cuda.mem_get_info()`; it exits before workers start. The parent does not
initialize CUDA and no NVML-total fallback is used. Startup subtracts the
observed usage (conservatively including the probe context) and a 2 GiB
non-PyTorch reserve, then assigns independent PyTorch native allocator caps.
These caps control PyTorch allocations, not arbitrary driver/library allocations
or other processes. Task-boundary device overruns log rate-limited warnings and
do not abort generation; allocator-cap OOM still raises an error.
Insufficient budget raises an error instead of silently shrinking the KV pools.
Warmup checks include both allocated and reserved memory peaks. For fragmentation
experiments, optionally set `PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:True`
before starting Python; this experimental option does not guarantee a fit.

Workers initialize one at a time on a shared GPU. Runtime scheduler order,
request queues, IPC messages, NCCL group structure, and acceptance/rollback
remain as in V4.2. Both model workers may have pending work at the same time;
there is no serial GPU-task arbiter or new admission policy.

```bash
python example.py --target-model-path /models/Qwen3-14B \
  --draft-model-path /models/Qwen3-0.6B \
  --target-gpus 0 --draft-gpus 0 --k 4 --batch 8 \
  --target-kv-cache-gib 21.25 --draft-kv-cache-gib 14.875 \
  --gpu-memory-utilization 0.9 --max-num-batched-tokens 4096
python example.py --target-model-path /models/Qwen3-14B --k 0 --batch 8
```

`EngineConfig(speculative_k=5, batch=8)` configures the same settings in Python.
The batch limit applies per scheduler dispatch, including mixed prefill/work
batches. Smaller max_num_seqs and the token budget may further limit selection.
K=0 defaults to target TP=1 and does not require draft files or a draft worker.
Both the Python and CLI default token budgets are 4096; this limits each dispatch,
not the total tokens resident in the KV pools. It does not add chunked prefill
to the speculative scheduler. Separate-GPU placements remain supported by
explicitly passing `--target-gpus 0 1 --draft-gpus 2`; without fixed-KV arguments
they retain the original per-model utilization-based allocation.

## Queues and lifecycle

States belong to each Sequence. Queues contain request IDs, and two model copies
of a request share an ID while retaining independent KV blocks.

Target uses prefill, running, verify and pausing. Draft uses prefill, running,
drafting and pausing. The unused role-specific queue stays empty in the shared
scheduler implementation.

1. An incoming request is created only in target.prefill. Draft has no copy yet.
2. Target selects verify first, then fills spare dispatch slots from prefill.
   A new request moves prefill -> running. Prompt prefill yields its first greedy
   token directly from the last prompt logits; no redundant decode is required.
3. If this token ends the request, target completes locally. Otherwise target
   moves running -> pausing and calls draft.receive_prefill with request_id,
   prompt tokens, first token and output parameters.
4. Draft now creates its Sequence in prefill. Its initial queue was empty.
   Draft selects drafting first and then fills spare slots with prefill.
5. A draft prefill task processes prompt + target's first token, writes their KV,
   and uses those logits to produce candidate 1. The same task continues until K
   candidates, EOS (unless ignore_eos), or the remaining output allowance.
   There is no intermediate prefill -> drafting round trip.
6. Draft moves running -> pausing and directly calls target.receive_draft.
   Target moves pausing -> verify. Target accepts the matching greedy prefix and
   adds a replacement at the first mismatch or a bonus when all candidates match.
7. Target commits results and trims its cache, then sends accepted_count and
   next_token directly to draft.receive_target_result. Draft updates its confirmed
   context, trims its own cache, and moves pausing -> drafting for the next round.
8. Target-confirmed EOS/output limit directly releases both copies and KV blocks.
   A draft EOS is only a candidate until target accepts it.

```text
target: prefill -> running -> pausing -> verify -> running -> pausing ...
draft:  (created after first target token)
        prefill -> running [prefill + K candidates] -> pausing
        pausing -> drafting -> running [K candidates] -> pausing ...
```

High-priority tasks consume a shared batch/token budget first. A deferred
high-priority task prevents admission of additional prefill into that dispatch.
Mixed draft tasks retain the batch cap throughout their internal loop; sequences
that hit EOS/limit leave the loop. Graph-eligible target verification runs first; remaining verification/prefill uses eager in the same capped dispatch.
Engine polls completions and drives scheduling; it never relays peer payloads.

## Allocation and rollback

There is no max-output-length reservation or full-lifetime admission accounting.
Only missing blocks for the current task are allocated:

- Target prefill: prompt length.
- Target verification: confirmed length + received candidate count.
- Draft round, including initial prefill: confirmed length + round limit - 1.
- Ordinary target decode: current confirmed length.

The last generated token has not yet been fed through the model. After accepting
A candidates from confirmed length L, target retains cache through L+A, and
draft retains min(actual cached length, L+A). The replacement/bonus is confirmed
but remains uncached. Next round computes that missing suffix. Unused complete
tail blocks are returned after early draft EOS and verification rollback.

Blocks are exclusive to requests; prefix sharing remains disabled. Draft single-token decode uses CUDA Graphs by default; full-width target verification also uses CUDA Graphs by default, with eager fallback for other shapes.
Requests waiting for a peer retain cache. There is no eviction/recomputation;
if all work is blocked by KV capacity and no worker can make progress, the engine
reports capacity exhaustion instead of spinning. No full-length reservation is
used to avoid this condition.

## Native K=0 target path

K=0 constructs `NativeTargetEngine` (`NativeTP2Engine` is a compatibility alias), a thin configuration/context-manager adapter
over `nanovllm.native.engine.llm_engine.LLMEngine`. It does not construct either
custom scheduler, the custom ModelRunnerGroup, or the speculative model runner.

`nanovllm/native/` is copied from GeeeekExplorer/nano-vllm commit
`bb823b3e06983d71485a8e1f23715ebd87d98ef8`. Package imports are relocated and
sampling is adapted to greedy. V5.0 also adds fixed KV allocation and device
budget checks. The native scheduler, prefix cache, preemption, kernels and
shared-memory TP dispatch are preserved. Native rank 0 executes in the host
process; additional ranks are spawned only when TP is greater than one.
GPU visibility is set to the configured target GPU IDs before initialization;
start this mode in a fresh process before creating any CUDA context.

`batch` maps to upstream `max_num_seqs`. Native CUDA Graphs are enabled by default
for the CLI/default EngineConfig; `--enforce-eager` explicitly disables them.
The upstream implementation's constraints (including graph capture batch sizes)
are retained, not silently patched. There are no five-queue policies in K=0.

```bash
python example.py --k 0 --batch 8
python example.py --k 0 --batch 8 --enforce-eager
```

The pinned upstream sampler originally rejected temperature=0. Per the updated
requirement, native SamplingParams now defaults to zero and the native compiled
sampler returns argmax directly. Both modes support only greedy sampling.
V5.0 keeps the native compute and CUDA Graph paths while making TP configurable
and changing KV sizing when an explicit budget is supplied.

```python
from nanovllm import LLM, EngineConfig, NativeSamplingParams

if __name__ == "__main__":
    with LLM(EngineConfig(speculative_k=0, batch=8)) as llm:
        outputs = llm.generate(["Hello"], NativeSamplingParams(max_tokens=128))
```

For mode-independent callers, `llm.sampling_params(...)` selects the appropriate
parameter class. Example and benchmark use this entry point. K=0 retains native
low-level API behavior: step() returns `(outputs, num_tokens)`, and add_request()
does not return an ID. K>0 keeps its existing speculative API.

Model paths remain local. V5.0 CPU-only tests use mocked CUDA/model loading;
A100 model execution and performance measurements remain to be performed.


## 4.0 draft CUDA Graph

See [CHANGES_4.0.md](CHANGES_4.0.md) for the implementation notes in Chinese.
K>0 uses eager draft prefill/catch-up and graph replay for single-token steps.
Only the model backbone is captured; LM head and greedy argmax remain outside.
Buckets follow upstream 1,2,4,8,16,32,48,...,512, clipped to the configured batch
limit with a final boundary bucket when needed. Replay rounds up the active batch.

```bash
python example.py --k 5 --batch 24
python example.py --k 5 --batch 24 --draft-enforce-eager
python example.py --k 5 --batch 24 --draft-graph-memory-reserve-mb 512
```

Graph memory headroom defaults to 256 MiB per draft rank. For fixed KV pools it
is checked against the separate allocator headroom; it is not deducted from
the configured KV bytes. The original automatic sizing path still deducts it
before allocating KV. This is an estimate, not a guarantee of capture capacity;
there are no automatic retries or silent fallback on capture failure.


## 4.1 target verify CUDA Graph

See [CHANGES_4.1.md](CHANGES_4.1.md) for the detailed Chinese change report.
K>0 target captures only the Transformer backbone using the existing paged varlen
attention. LM head, greedy and acceptance remain outside capture. Native-style
request buckets (1,2,4,8,16,32,48,...) round up the actual verify batch; each row
contains K+1 queries. Buckets are capped by batch, 512 and the configured token
budget, with a final boundary bucket as needed.

Only complete K-candidate requests with one uncached confirmed token use replay.
Short drafts, catch-up and prefill remain eager. Padding reads dedicated zero KV
scratch blocks, skips cache writes via slot=-1, and never enters sampling. Scratch
blocks are excluded from the scheduler's free pool. No per-request maximum-length
cache reservation is introduced.

```bash
python example.py --k 6 --batch 16
python example.py --k 6 --batch 16 --target-enforce-eager
python example.py --k 6 --batch 16 --target-graph-memory-reserve-mb 512
```

Target headroom defaults to 256 MiB per rank and is a configurable estimate, not
a measured memory requirement. Capture errors are reported through the existing
worker channel. `--enforce-eager` retains its K=0 meaning; draft retains its own
switch. `Config.target_verify_graph` controls K>0 target graph dispatch.
No tests, compilation checks, model execution or benchmarks were run for 4.1.


## Installation

This experimental fork retains the upstream MIT license and attribution.
Use Linux with a PyTorch/CUDA/FlashAttention environment that supports A100.
Install PyTorch and the matching FlashAttention build for your environment, then:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

Model weights are not included. Pass your own local model paths as shown above.
The public V5.0 package preserves the local Python sources byte-for-byte; packaging
only completes dependency declarations and adds this installation note.
CPU tests pass; A100 model execution, memory peaks, and performance are not yet
validated. See `SOURCE_MANIFEST.json` and `CHANGES_5.0.md`.
