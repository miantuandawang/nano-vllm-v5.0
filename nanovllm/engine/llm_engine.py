import atexit
from multiprocessing.connection import wait
from pathlib import Path
from tempfile import TemporaryDirectory

import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import EngineConfig
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.draft_scheduler import DraftScheduler
from nanovllm.engine.target_scheduler import TargetScheduler
from nanovllm.engine.runner_group import ModelRunnerGroup
from nanovllm.utils.memory_budget import prepare_device_budget


class LLMEngine:
    sampling_params = SamplingParams

    def __new__(cls, config=None):
        if config is not None and config.speculative_k == 0:
            from nanovllm.engine.native_engine import NativeTargetEngine
            return NativeTargetEngine(config)
        return super().__new__(cls)

    def __init__(self, config=None):
        self.config = config or EngineConfig()
        self.model_runners = {}
        self._closed = False
        self._rendezvous = TemporaryDirectory(prefix="nanovllm-")
        try:
            target_tokenizer = AutoTokenizer.from_pretrained(self.config.target.model, local_files_only=True)
            if self.config.speculative_k:
                draft_tokenizer = AutoTokenizer.from_pretrained(self.config.draft.model, local_files_only=True)
                if (target_tokenizer.get_vocab() != draft_tokenizer.get_vocab()
                        or target_tokenizer.all_special_tokens != draft_tokenizer.all_special_tokens
                        or target_tokenizer.eos_token_id != draft_tokenizer.eos_token_id):
                    raise ValueError("Draft and target must share token IDs and special tokens")
            self.tokenizer = target_tokenizer
            if self.config.shared_gpu:
                prepare_device_budget(
                    [self.config.target, self.config.draft],
                    self.config.device_memory_utilization,
                    self.config.device_memory_reserve_bytes,
                )
            context = mp.get_context("spawn")
            for name in (("draft", "target") if self.config.speculative_k else ("target",)):
                model_config = getattr(self.config, name)
                model_config.eos = self.tokenizer.eos_token_id
                init_method = (Path(self._rendezvous.name) / name).as_uri()
                self.model_runners[name] = ModelRunnerGroup(model_config, context, init_method)
                # Initialization only: do not overlap warmup/graph capture on one
                # device. Runtime scheduling and the Pipe protocol stay intact.
                if self.config.shared_gpu:
                    self.model_runners[name].wait_until_ready()
            for runner in self.model_runners.values():
                if not runner.ready:
                    runner.wait_until_ready()
            self.target = TargetScheduler(self.config.target, self.model_runners["target"], self.config.batch, self.config.speculative_k)
            self.draft = None
            self.schedulers = {"target": self.target}
            if self.config.speculative_k:
                self.draft = DraftScheduler(self.config.draft, self.model_runners["draft"], self.config.speculative_k, self.config.batch)
                self.draft.peer = self.target
                self.target.peer = self.draft
                self.schedulers["draft"] = self.draft
        except BaseException:
            self.exit(force=True)
            raise
        atexit.register(self.exit)

    def add_request(self, prompt, sampling_params):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if not prompt or sampling_params.max_tokens <= 0:
            raise ValueError("A nonempty prompt and positive max_tokens are required")
        limit = self.config.target.max_model_len
        if self.config.speculative_k:
            limit = min(limit, self.config.draft.max_model_len)
        if len(prompt) + sampling_params.max_tokens > limit:
            raise ValueError("Prompt plus max_tokens exceeds the models' configured context limit")
        return self.target.add_request(prompt, sampling_params)

    def step(self):
        if self._closed:
            raise RuntimeError("LLMEngine is closed")
        try:
            if self.draft is not None:
                self.draft.poll_results()
            self.target.poll_results()
            # Process both groups' completions before selecting either next batch.
            # Target verification frees/commits work; draft schedules from the
            # queues updated by the direct callbacks above.
            self.target.schedule()
            if self.draft is not None:
                self.draft.schedule()
            if not self.is_finished() and not any(r.pending for r in self.model_runners.values()):
                raise RuntimeError("No request can advance: insufficient KV cache capacity for queued tasks")
            return self.target.pop_finished_outputs()
        except BaseException:
            self.exit(force=True)
            raise

    def is_finished(self):
        return all(scheduler.is_finished() for scheduler in self.schedulers.values())

    def generate(self, prompts, sampling_params, use_tqdm=True):
        if not self.is_finished():
            raise RuntimeError("Finish queued requests before calling generate")
        params = sampling_params if isinstance(sampling_params, list) else [sampling_params] * len(prompts)
        if len(params) != len(prompts):
            raise ValueError("Expected one sampling parameter set per prompt")
        ids = [self.add_request(prompt, sp) for prompt, sp in zip(prompts, params)]
        outputs = {}
        with tqdm(total=len(ids), desc="Speculative greedy" if self.config.speculative_k else "Target greedy TP=2", disable=not use_tqdm) as progress:
            while not self.is_finished():
                for request_id, tokens in self.step():
                    outputs[request_id] = tokens
                    progress.update(1)
                # Wait briefly for a worker response instead of spinning on paused requests.
                handles = [handle for runner in self.model_runners.values() for handle in runner.wait_handles()]
                if handles:
                    wait(handles, timeout=0.01)
        return [{"text": self.tokenizer.decode(outputs[i]), "token_ids": outputs[i]} for i in ids]

    def exit(self, force=False):
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.exit)
        try:
            for runner in self.model_runners.values():
                runner.close(force=force)
        finally:
            self._rendezvous.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit(force=exc_type is not None)
