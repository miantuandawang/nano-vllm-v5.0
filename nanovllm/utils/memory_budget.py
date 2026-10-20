"""Fixed KV pools and a shared-device budget; no scheduling or IPC changes."""

import json
import logging
import math
import os
import subprocess
import sys
from time import monotonic

GIB = 1024**3
MIB = 1024**2
TARGET_KV_BYTES = 21760 * MIB   # 21.25 GiB, 544 Qwen3-14B blocks.
DRAFT_KV_BYTES = 15232 * MIB    # 14.875 GiB, 544 Qwen3-0.6B blocks.
_LOGGER = logging.getLogger(__name__)
_LAST_BUDGET_WARNING = {}
_PROBE_PREFIX = "V50_CUDA_MEMORY="
_CUDA_MEMORY_PROBE = """
import json
import torch
torch.cuda.set_device(0)
free, total = torch.cuda.mem_get_info()
print("V50_CUDA_MEMORY=" + json.dumps({
    "name": torch.cuda.get_device_name(0),
    "total": total,
    "free": free,
}), flush=True)
"""


def gib_to_bytes(value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError("KV cache GiB must be finite and positive")
    return int(value * GIB)


def kv_block_bytes(config):
    hf = config.hf_config
    if hf.num_key_value_heads % config.tensor_parallel_size:
        raise ValueError("KV heads must be divisible by tensor_parallel_size")
    heads = hf.num_key_value_heads // config.tensor_parallel_size
    dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
    return (2 * hf.num_hidden_layers * config.kvcache_block_size
            * heads * dim * hf.dtype.itemsize)


def fixed_kv_blocks(config, scratch_blocks=0):
    count = config.kv_cache_memory_bytes // kv_block_bytes(config)
    if count <= scratch_blocks:
        raise ValueError(f"{config.name}: KV budget cannot hold usable blocks and scratch")
    return count


def estimate_weight_bytes(hf):
    """Resident Qwen3 parameter storage, including tied embedding storage once."""
    if hf.model_type != "qwen3":
        raise ValueError("The V5.0 shared-GPU budget supports Qwen3 models")
    h, intermediate = hf.hidden_size, hf.intermediate_size
    dim = getattr(hf, "head_dim", h // hf.num_attention_heads)
    q = hf.num_attention_heads * dim
    kv = hf.num_key_value_heads * dim
    # Matches Qwen3ForCausalLM, including its attention-bias/QK-norm branch.
    attention_bias = getattr(hf, "attention_bias", True)
    per_layer = h * (q + 2 * kv) + q * h + 3 * h * intermediate + 2 * h
    per_layer += q + 2 * kv if attention_bias else 2 * dim
    params = hf.num_hidden_layers * per_layer + h
    params += hf.vocab_size * h * (1 if hf.tie_word_embeddings else 2)
    return params * hf.dtype.itemsize


def query_gpu(gpu_id):
    """Probe CUDA capacity in a fresh interpreter, using the workers' mapping.

    NVML framebuffer totals are not interchangeable with cudaMemGetInfo totals.
    The probe exits before workers start; its context usage remains included in
    the baseline conservatively, on top of the separate non-PyTorch reserve.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        result = subprocess.run(
            [sys.executable, "-c", _CUDA_MEMORY_PROBE],
            env=env, check=True, capture_output=True, text=True, timeout=60,
        )
        rows = [line[len(_PROBE_PREFIX):] for line in result.stdout.splitlines()
                if line.startswith(_PROBE_PREFIX)]
        if len(rows) != 1:
            raise ValueError("expected one CUDA memory probe result")
        data = json.loads(rows[0])
        name, total, free = data["name"], data["total"], data["free"]
        if (not isinstance(name, str) or not name or type(total) is not int
                or type(free) is not int or not 0 <= free <= total or total <= 0):
            raise ValueError("invalid CUDA memory probe result")
        return name, total, total - free
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise RuntimeError(
            f"Cannot inspect GPU {gpu_id} with a CUDA subprocess: {detail}. "
            "Check PyTorch/CUDA and GPU visibility; no NVML-total fallback is used."
        ) from exc


def plan_memory(configs, total, already_used, utilization=0.9, reserve_bytes=2 * GIB):
    """Return independent allocator ceilings whose SUM fits the device budget.

    External usage and the non-PyTorch reserve do not belong to either allocator.
    Remaining allocator headroom is split 4:1 between target and draft.
    """
    if not math.isfinite(utilization) or not 0 < utilization <= 0.9:
        raise ValueError("Device GPU memory utilization must be in (0, 0.9]")
    if total <= 0 or already_used < 0 or reserve_bytes < 0:
        raise ValueError("Invalid device memory accounting")
    names = [c.name for c in configs]
    if not names or len(names) != len(set(names)) or set(names) - {"target", "draft"}:
        raise ValueError("Expected distinct target/draft model configurations")
    bases = {}
    for config in configs:
        if config.tensor_parallel_size != 1 or config.kv_cache_memory_bytes is None:
            raise ValueError("Shared-device budgeting requires TP=1 and explicit KV bytes")
        bases[config.name] = (estimate_weight_bytes(config.hf_config)
                              + fixed_kv_blocks(config) * kv_block_bytes(config))
    limit = int(total * utilization)
    available = limit - already_used - reserve_bytes
    extra = available - sum(bases.values())
    # These minima are allocator headroom, separate from the device reserve.
    minima = {c.name: max(GIB, c.graph_memory_reserve_mb * MIB) for c in configs}
    if extra < sum(minima.values()):
        raise RuntimeError(
            f"GPU budget insufficient: total={total / GIB:.3f} GiB, "
            f"limit={limit / GIB:.3f} GiB, existing={already_used / GIB:.3f} GiB, "
            f"weights+KV={sum(bases.values()) / GIB:.3f} GiB, "
            f"non-PyTorch reserve={reserve_bytes / GIB:.3f} GiB, "
            f"minimum runtime headroom={sum(minima.values()) / GIB:.3f} GiB. "
            "Reduce explicit KV budgets or release other GPU allocations."
        )
    remaining = extra - sum(minima.values())
    shares = {name: 4 if name == "target" else 1 for name in names}
    denominator = sum(shares.values())
    caps = {name: bases[name] + minima[name] + remaining * shares[name] // denominator
            for name in names}
    return limit, caps, bases


def prepare_device_budget(configs, utilization=0.9, reserve_bytes=2 * GIB):
    if any(c.gpu_ids != configs[0].gpu_ids or len(c.gpu_ids) != 1 for c in configs):
        raise ValueError("Device budget requires all models on one physical GPU")
    name, total, used = query_gpu(configs[0].gpu_ids[0])
    limit, caps, bases = plan_memory(configs, total, used, utilization, reserve_bytes)
    print(f"[V5.0 memory] GPU={name}, CUDA total={total / GIB:.3f} GiB, "
          f"device limit={limit / GIB:.3f} GiB ({utilization:.1%}), "
          f"probe usage={used / GIB:.3f} GiB (includes probe context), "
          f"non-PyTorch reserve={reserve_bytes / GIB:.3f} GiB",
          flush=True)
    for config in configs:
        config.allocator_limit_bytes = caps[config.name]
        config.device_memory_limit_bytes = limit
        config.device_memory_utilization = utilization
        print(f"[V5.0 memory] {config.name}: KV={config.kv_cache_memory_bytes / GIB:.3f} GiB, "
              f"allocator cap={caps[config.name] / GIB:.3f} GiB, "
              f"runtime headroom={(caps[config.name] - bases[config.name]) / GIB:.3f} GiB",
              flush=True)


def apply_allocator_limit(config):
    if config.allocator_limit_bytes is None:
        return
    import torch
    if torch.cuda.memory.get_allocator_backend() != "native":
        raise RuntimeError("V5.0 memory caps require the native PyTorch CUDA allocator")
    _, total = torch.cuda.mem_get_info()
    actual_limit = int(total * config.device_memory_utilization)
    if config.device_memory_limit_bytes > actual_limit:
        # Both values now come from the same CUDA API. A smaller device here
        # means the device/configuration changed; shrinking only this worker's
        # limit would not safely re-plan the already-running peer's cap.
        raise RuntimeError(
            "CUDA-visible capacity decreased since the CUDA probe; "
            "restart the engine to re-plan both workers"
        )
    torch.cuda.set_per_process_memory_fraction(config.allocator_limit_bytes / total)
    check_device_budget(config, "before model load")


def check_device_budget(config, stage):
    """Observe whole-device usage without aborting in-flight generation.

    Allocator caps remain hard. External/non-PyTorch usage is advisory and
    rate-limited to one warning per model per 30 seconds while above budget.
    """
    if config.device_memory_limit_bytes is None:
        return
    import torch
    free, total = torch.cuda.mem_get_info()
    used = total - free
    key = (os.getpid(), config.name, config.device_memory_limit_bytes)
    if used > config.device_memory_limit_bytes:
        now = monotonic()
        last = _LAST_BUDGET_WARNING.get(key)
        if last is None or now - last >= 30:
            _LOGGER.warning(
                "%s: GPU usage %.3f GiB exceeds %.3f GiB at %s; continuing "
                "with allocator caps unchanged. Check external/non-PyTorch "
                "allocations and measure the device peak.",
                config.name, used / GIB, config.device_memory_limit_bytes / GIB, stage,
            )
            _LAST_BUDGET_WARNING[key] = now
    else:
        _LAST_BUDGET_WARNING.pop(key, None)


def check_fixed_kv_capacity(config, kv_bytes, graph_reserve):
    """Validate the warmed-up peak plus KV against the per-process ceiling."""
    if config.allocator_limit_bytes is None:
        return
    import torch
    stats = torch.cuda.memory_stats()
    allocated_peak = stats["allocated_bytes.all.peak"]
    reserved_peak = stats["reserved_bytes.all.peak"]
    peak = max(allocated_peak, reserved_peak)
    if peak + kv_bytes + graph_reserve > config.allocator_limit_bytes:
        raise RuntimeError(
            f"{config.name}: warmup allocator peak + fixed KV + graph reserve exceeds "
            f"allocator cap ({peak / GIB:.3f} + {kv_bytes / GIB:.3f} + "
            f"{graph_reserve / GIB:.3f} > {config.allocator_limit_bytes / GIB:.3f} GiB). "
            f"allocated peak={allocated_peak / GIB:.3f} GiB, "
            f"reserved peak={reserved_peak / GIB:.3f} GiB. "
            "Reduce max_num_batched_tokens, batch, or the explicit KV budget. "
            "Expandable segments may reduce fragmentation but cannot guarantee a fit."
        )
