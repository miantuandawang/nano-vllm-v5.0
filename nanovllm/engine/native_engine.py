import atexit
import os

from nanovllm.native.engine.llm_engine import LLMEngine as UpstreamEngine
from nanovllm.native.sampling_params import SamplingParams
from nanovllm.utils.memory_budget import prepare_device_budget


class NativeTargetEngine(UpstreamEngine):
    """Configuration/lifecycle adapter; computation and scheduling are upstream."""
    sampling_params = SamplingParams

    def __init__(self, config):
        target = config.target
        if target.tensor_parallel_size == 1:
            prepare_device_budget(
                [target], config.device_memory_utilization,
                config.device_memory_reserve_bytes,
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, target.gpu_ids))
        super().__init__(
            target.model, tensor_parallel_size=target.tensor_parallel_size,
            enforce_eager=target.enforce_eager,
            max_num_seqs=min(config.batch, target.max_num_seqs),
            max_num_batched_tokens=target.max_num_batched_tokens,
            max_model_len=target.max_model_len,
            gpu_memory_utilization=target.gpu_memory_utilization,
            kvcache_block_size=target.kvcache_block_size,
            kv_cache_memory_bytes=target.kv_cache_memory_bytes,
            allocator_limit_bytes=target.allocator_limit_bytes,
            device_memory_limit_bytes=target.device_memory_limit_bytes,
            device_memory_utilization=target.device_memory_utilization,
            graph_memory_reserve_mb=target.graph_memory_reserve_mb,
        )

    def add_request(self, prompt, sampling_params):
        if not isinstance(sampling_params, SamplingParams):
            raise TypeError("K=0 requires NativeSamplingParams (native greedy/sampling parameters)")
        return super().add_request(prompt, sampling_params)

    def exit(self):
        if hasattr(self, "model_runner"):
            super().exit()
        atexit.unregister(self.exit)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit()


# Keep old imports working; TP now follows the supplied configuration.
NativeTP2Engine = NativeTargetEngine
