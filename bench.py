from random import Random
from time import perf_counter

from model_paths import parse_engine_args
from nanovllm import LLM


def main():
    rng = Random(0)
    prompts = [[rng.randrange(10000) for _ in range(rng.randint(100, 1024))] for _ in range(256)]
    lengths = [rng.randint(100, 1024) for _ in prompts]
    with LLM(parse_engine_args()) as llm:
        params = [llm.sampling_params(max_tokens=n, ignore_eos=True) for n in lengths]
        llm.generate(["Warmup"], llm.sampling_params(max_tokens=16), use_tqdm=False)
        start = perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        elapsed = perf_counter() - start
        count = sum(len(output["token_ids"]) for output in outputs)
        print(f"Confirmed output: {count} tokens; time: {elapsed:.2f}s; throughput: {count / elapsed:.2f} tok/s")


if __name__ == "__main__":
    main()
