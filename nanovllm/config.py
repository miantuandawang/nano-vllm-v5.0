import os
import math
from dataclasses import dataclass, field
from transformers import AutoConfig
from nanovllm.utils.memory_budget import TARGET_KV_BYTES, DRAFT_KV_BYTES, GIB


MODEL_PATHS = {
    "qwen3-14B": "/home/zhj/huggingface/Qwen3-14B",
    "qwen3-0.6B": "/home/zhj/huggingface/Qwen3-0.6B",
}


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 4096
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = True
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    name: str = "target"
    gpu_ids: tuple[int, ...] = (0,)
    worker_timeout: float = 600.0
    graph_memory_reserve_mb: int = 256
    target_verify_graph: bool = True
    # Filled from EngineConfig before workers are spawned.
    speculative_k: int = 0
    kv_cache_memory_bytes: int | None = None
    # Filled by the parent device budget planner before spawning workers.
    allocator_limit_bytes: int | None = None
    device_memory_limit_bytes: int | None = None
    device_memory_utilization: float = 0.9

    def __post_init__(self):
        if self.kv_cache_memory_bytes is not None and (
                type(self.kv_cache_memory_bytes) is not int or self.kv_cache_memory_bytes <= 0):
            raise ValueError("kv_cache_memory_bytes must be a positive integer")
        if self.graph_memory_reserve_mb < 0:
            raise ValueError("graph_memory_reserve_mb must be nonnegative")
        self.model = os.path.abspath(os.path.expanduser(self.model))
        self.gpu_ids = tuple(self.gpu_ids)
        if not os.path.isdir(self.model):
            raise ValueError(f"{self.name}: local model directory does not exist: {self.model}")
        if self.name not in ("draft", "target"):
            raise ValueError("Scheduler name must be 'draft' or 'target'")
        if not 1 <= self.tensor_parallel_size <= 8:
            raise ValueError("tensor_parallel_size must be between 1 and 8")
        if len(self.gpu_ids) != self.tensor_parallel_size:
            raise ValueError(f"{self.name}: gpu_ids must contain one GPU per TP rank")
        if len(set(self.gpu_ids)) != len(self.gpu_ids) or any(i < 0 for i in self.gpu_ids):
            raise ValueError(f"{self.name}: GPU IDs must be unique and nonnegative")
        if self.kvcache_block_size <= 0 or self.kvcache_block_size % 256:
            raise ValueError("kvcache_block_size must be a positive multiple of 256")
        if min(self.max_num_batched_tokens, self.max_num_seqs, self.max_model_len) <= 0:
            raise ValueError("Batch and sequence limits must be positive")
        if not 0 < self.gpu_memory_utilization <= 1 or self.worker_timeout <= 0:
            raise ValueError("Invalid GPU memory utilization or worker timeout")
        self.hf_config = AutoConfig.from_pretrained(self.model, local_files_only=True)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)


@dataclass(slots=True)
class EngineConfig:
    speculative_k: int = 5
    batch: int = 8
    draft: Config | None = None
    target: Config = field(default_factory=lambda: Config(
        model=MODEL_PATHS["qwen3-14B"], name="target", gpu_ids=(0,), tensor_parallel_size=1,
        enforce_eager=False,
    ))
    device_memory_utilization: float = 0.9
    # CUDA contexts/library allocations outside the PyTorch allocator.
    device_memory_reserve_bytes: int = 2 * GIB

    @property
    def shared_gpu(self):
        return bool(self.speculative_k and self.draft is not None
                    and set(self.draft.gpu_ids) & set(self.target.gpu_ids))

    def __post_init__(self):
        if (not math.isfinite(self.device_memory_utilization)
                or not 0 < self.device_memory_utilization <= 0.9):
            raise ValueError("Device memory utilization must be in (0, 0.9]")
        if (type(self.device_memory_reserve_bytes) is not int
                or self.device_memory_reserve_bytes < GIB):
            raise ValueError("Reserve at least 1 GiB for non-PyTorch device allocations")
        if self.speculative_k < 0 or self.batch < 1:
            raise ValueError("K must be nonnegative and batch must be positive")
        if self.target.name != "target":
            raise ValueError("Target must use name='target'")
        self.target.speculative_k = self.speculative_k
        if self.speculative_k:
            # enforce_eager controls native K=0; target_verify_graph controls verify.
            self.target.enforce_eager = True
            if self.draft is None:
                self.draft = Config(model=MODEL_PATHS["qwen3-0.6B"], name="draft",
                                    gpu_ids=(0,), tensor_parallel_size=1, enforce_eager=False)
            if self.draft.name != "draft":
                raise ValueError("Draft must use name='draft'")
            if self.shared_gpu:
                if (self.target.tensor_parallel_size != 1 or self.draft.tensor_parallel_size != 1
                        or self.target.gpu_ids != self.draft.gpu_ids):
                    raise ValueError("Shared GPU requires target and draft TP=1 on the same GPU")
                if self.target.kv_cache_memory_bytes is None:
                    self.target.kv_cache_memory_bytes = TARGET_KV_BYTES
                if self.draft.kv_cache_memory_bytes is None:
                    self.draft.kv_cache_memory_bytes = DRAFT_KV_BYTES
        elif self.target.tensor_parallel_size == 1 and self.target.kv_cache_memory_bytes is None:
            self.target.kv_cache_memory_bytes = TARGET_KV_BYTES
        for config in ([self.target, self.draft] if self.speculative_k else [self.target]):
            config.max_num_seqs = min(config.max_num_seqs, self.batch)
