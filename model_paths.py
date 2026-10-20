import argparse

from nanovllm.config import Config, EngineConfig, MODEL_PATHS
from nanovllm.utils.memory_budget import gib_to_bytes


def parse_engine_args():
    parser = argparse.ArgumentParser(description="Greedy speculative decoding")
    parser.add_argument("--draft-model-path", default=MODEL_PATHS["qwen3-0.6B"])
    parser.add_argument("--target-model-path", default=MODEL_PATHS["qwen3-14B"])
    parser.add_argument("--draft-gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--target-gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--target-kv-cache-gib", type=float, default=None,
                        help="Fixed KV per GPU; shared-GPU default: 21.25 GiB")
    parser.add_argument("--draft-kv-cache-gib", type=float, default=None,
                        help="Fixed KV per GPU; shared-GPU default: 14.875 GiB")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="Combined device budget for same-GPU models (maximum 0.9)")
    parser.add_argument("--device-memory-reserve-gib", type=float, default=2.0,
                        help="Reserve inside the device budget for non-PyTorch allocations")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--enforce-eager", action="store_true", help="Disable native CUDA Graphs for K=0")
    parser.add_argument("--draft-enforce-eager", action="store_true",
                        help="Disable draft single-token CUDA Graphs for K>0")
    parser.add_argument("--target-enforce-eager", action="store_true",
                        help="Disable target verify CUDA Graphs for K>0")
    parser.add_argument("--target-graph-memory-reserve-mb", type=int, default=256)
    parser.add_argument("--draft-graph-memory-reserve-mb", type=int, default=256)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--draft-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--target-gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()
    common = dict(max_model_len=args.max_model_len,
                  max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens)
    return EngineConfig(
        speculative_k=args.k, batch=args.batch,
        device_memory_utilization=args.gpu_memory_utilization,
        device_memory_reserve_bytes=gib_to_bytes(args.device_memory_reserve_gib),
        draft=Config(enforce_eager=args.draft_enforce_eager,
                     kv_cache_memory_bytes=(gib_to_bytes(args.draft_kv_cache_gib)
                                            if args.draft_kv_cache_gib is not None else None),
                     graph_memory_reserve_mb=args.draft_graph_memory_reserve_mb,
                     model=args.draft_model_path, name="draft", gpu_ids=tuple(args.draft_gpus),
                     tensor_parallel_size=len(args.draft_gpus),
                     gpu_memory_utilization=args.draft_gpu_memory_utilization, **common) if args.k else None,
        target=Config(target_verify_graph=not args.target_enforce_eager,
                      kv_cache_memory_bytes=(gib_to_bytes(args.target_kv_cache_gib)
                                             if args.target_kv_cache_gib is not None else None),
                      graph_memory_reserve_mb=args.target_graph_memory_reserve_mb,
                      enforce_eager=bool(args.k) or args.enforce_eager, model=args.target_model_path, name="target", gpu_ids=tuple(args.target_gpus),
                      tensor_parallel_size=len(args.target_gpus),
                      gpu_memory_utilization=args.target_gpu_memory_utilization, **common),
    )
