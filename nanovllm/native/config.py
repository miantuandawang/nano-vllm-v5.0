import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    name: str = "target"
    kv_cache_memory_bytes: int | None = None
    allocator_limit_bytes: int | None = None
    device_memory_limit_bytes: int | None = None
    device_memory_utilization: float = 0.9
    graph_memory_reserve_mb: int = 256

    def __post_init__(self):
        if self.kv_cache_memory_bytes is not None and (
                type(self.kv_cache_memory_bytes) is not int or self.kv_cache_memory_bytes <= 0):
            raise ValueError("kv_cache_memory_bytes must be a positive integer")
        if self.graph_memory_reserve_mb < 0:
            raise ValueError("graph_memory_reserve_mb must be nonnegative")
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
