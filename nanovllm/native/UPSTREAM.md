# Upstream source

GeeeekExplorer/nano-vllm commit `bb823b3e06983d71485a8e1f23715ebd87d98ef8`.
Python modules are copied from the local Git objects; absolute package imports
are relocated from nanovllm to nanovllm.native.

User-requested deviations: SamplingParams defaults to temperature=0 and accepts
only zero; the compiled sampler returns logits.argmax(dim=-1). Native scheduling,
KV allocation, prefix cache, chunked prefill, preemption, model execution, TP
communication and CUDA Graph algorithms are otherwise unchanged.
