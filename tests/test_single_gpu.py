"""CPU-only budget/config/integration tests; GPU execution is deliberately mocked.

Run: python3 -m unittest discover -s tests -v
No torch/transformers installation or model weights are required.
"""

import ast
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]
NS = types.SimpleNamespace
DTYPE = NS(itemsize=2)


def hf_config(target=True):
    return NS(model_type="qwen3", hidden_size=5120 if target else 1024,
              intermediate_size=17408 if target else 3072,
              num_attention_heads=40 if target else 16, num_key_value_heads=8,
              head_dim=128, num_hidden_layers=40 if target else 28,
              vocab_size=151936, attention_bias=False,
              tie_word_embeddings=not target, dtype=DTYPE, max_position_embeddings=40960)


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Avoid importing the public package entry point, which requires real GPU deps.
for package in ["nanovllm", "nanovllm.utils", "nanovllm.engine",
                "nanovllm.native", "nanovllm.native.engine"]:
    mod = types.ModuleType(package)
    mod.__path__ = [str(ROOT / package.replace(".", "/"))]
    sys.modules[package] = mod

fake_transformers = types.ModuleType("transformers")
class FakeAutoConfig:
    @staticmethod
    def from_pretrained(path, **kwargs):
        return hf_config("draft" not in str(path))
fake_transformers.AutoConfig = FakeAutoConfig
fake_transformers.AutoTokenizer = NS(from_pretrained=Mock())
sys.modules["transformers"] = fake_transformers

budget = load("nanovllm.utils.memory_budget", "nanovllm/utils/memory_budget.py")
configuration = load("nanovllm.config", "nanovllm/config.py")
cli = load("v50_model_paths", "model_paths.py")
params = load("nanovllm.sampling_params", "nanovllm/sampling_params.py")
load("nanovllm.engine.sequence", "nanovllm/engine/sequence.py")
load("nanovllm.engine.block_manager", "nanovllm/engine/block_manager.py")
load("nanovllm.engine.scheduler", "nanovllm/engine/scheduler.py")
draft_module = load("nanovllm.engine.draft_scheduler", "nanovllm/engine/draft_scheduler.py")
target_module = load("nanovllm.engine.target_scheduler", "nanovllm/engine/target_scheduler.py")
group_module = load("nanovllm.engine.runner_group", "nanovllm/engine/runner_group.py")

fake_torch = types.ModuleType("torch")
fake_mp = types.ModuleType("torch.multiprocessing")
fake_mp.get_context = Mock(return_value=object())
fake_torch.multiprocessing = fake_mp
sys.modules["torch"] = fake_torch
sys.modules["torch.multiprocessing"] = fake_mp
fake_tqdm = types.ModuleType("tqdm.auto")
fake_tqdm.tqdm = Mock()
sys.modules["tqdm"] = types.ModuleType("tqdm")
sys.modules["tqdm.auto"] = fake_tqdm
engine_module = load("nanovllm.engine.llm_engine", "nanovllm/engine/llm_engine.py")


class ConfigTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.target_path = Path(self.temp.name) / "target"
        self.draft_path = Path(self.temp.name) / "draft"
        self.target_path.mkdir()
        self.draft_path.mkdir()
        budget._LAST_BUDGET_WARNING.clear()

    def configs(self, **kwargs):
        return configuration.EngineConfig(
            target=configuration.Config(str(self.target_path)),
            draft=configuration.Config(str(self.draft_path), name="draft", enforce_eager=False),
            **kwargs,
        )


class BudgetTests(ConfigTestCase):
    def test_resident_weights_and_exact_block_capacity(self):
        c = self.configs()
        self.assertEqual(budget.estimate_weight_bytes(c.target.hf_config), 14768307200 * 2)
        self.assertEqual(budget.estimate_weight_bytes(c.draft.hf_config), 596049920 * 2)
        self.assertEqual(budget.kv_block_bytes(c.target), 40 * budget.MIB)
        self.assertEqual(budget.kv_block_bytes(c.draft), 28 * budget.MIB)
        self.assertEqual(budget.fixed_kv_blocks(c.target, 1), 544)
        self.assertEqual(budget.fixed_kv_blocks(c.draft), 544)
        self.assertEqual(c.target.kv_cache_memory_bytes * 7, c.draft.kv_cache_memory_bytes * 10)
        self.assertEqual((budget.fixed_kv_blocks(c.target, 1) - 1) * 256, 139008)

    def test_caps_fit_real_capacity_including_external_usage_and_reserve(self):
        c = self.configs()
        for total in [79 * budget.GIB, 80 * budget.GIB, 81000 * budget.MIB]:
            for used in [0, budget.GIB]:
                limit, caps, bases = budget.plan_memory(
                    [c.target, c.draft], total, used, reserve_bytes=2 * budget.GIB)
                self.assertLessEqual(sum(caps.values()) + used + 2 * budget.GIB, limit)
                self.assertLessEqual(limit, total * 0.9)
                for name in caps:
                    self.assertGreaterEqual(caps[name] - bases[name], budget.GIB)

    def test_reject_small_gpu_or_busy_gpu(self):
        c = self.configs()
        for total, used in [(24 * budget.GIB, 0), (80 * budget.GIB, 8 * budget.GIB)]:
            with self.assertRaisesRegex(RuntimeError, "GPU budget insufficient"):
                budget.plan_memory([c.target, c.draft], total, used)

    def test_reject_invalid_budget_and_impossible_scratch(self):
        for value in [0, -1, math.nan, math.inf]:
            with self.assertRaises(ValueError):
                budget.gib_to_bytes(value)
        c = self.configs()
        c.target.kv_cache_memory_bytes = 40 * budget.MIB
        with self.assertRaisesRegex(ValueError, "scratch"):
            budget.fixed_kv_blocks(c.target, 1)
        with self.assertRaises(ValueError):
            budget.plan_memory([c.target], 80 * budget.GIB, 0, utilization=0.91)

    def test_parent_plan_sets_both_process_limits(self):
        c = self.configs()
        with patch.object(budget, "query_gpu", return_value=("A100 80GB", 80 * budget.GIB, 0)), \
                contextlib.redirect_stdout(io.StringIO()):
            budget.prepare_device_budget([c.target, c.draft])
        self.assertEqual(c.target.device_memory_limit_bytes, 72 * budget.GIB)
        self.assertEqual(c.draft.device_memory_limit_bytes, c.target.device_memory_limit_bytes)
        self.assertLessEqual(c.target.allocator_limit_bytes + c.draft.allocator_limit_bytes,
                             70 * budget.GIB)

    def test_allocator_cap_and_global_overrun(self):
        c = self.configs().target
        c.allocator_limit_bytes = 52 * budget.GIB
        c.device_memory_limit_bytes = 72 * budget.GIB
        cuda = NS(memory=NS(get_allocator_backend=lambda: "native"),
                  mem_get_info=Mock(return_value=(79 * budget.GIB, 80 * budget.GIB)),
                  set_per_process_memory_fraction=Mock())
        with patch.object(fake_torch, "cuda", cuda, create=True):
            budget.apply_allocator_limit(c)
            cuda.set_per_process_memory_fraction.assert_called_once_with(52 / 80)
            cuda.mem_get_info.return_value = (7 * budget.GIB, 80 * budget.GIB)
            with self.assertLogs(budget._LOGGER, level="WARNING") as logs:
                budget.check_device_budget(c, "test")
            self.assertIn("continuing", logs.output[0])
            self.assertEqual(c.allocator_limit_bytes, 52 * budget.GIB)

    def test_reject_async_allocator_and_warmup_overrun(self):
        c = self.configs().target
        c.allocator_limit_bytes = 50 * budget.GIB
        c.device_memory_limit_bytes = 72 * budget.GIB
        cuda = NS(memory=NS(get_allocator_backend=lambda: "cudaMallocAsync"),
                  memory_stats=lambda: {"allocated_bytes.all.peak": 30 * budget.GIB,
                                        "reserved_bytes.all.peak": 31 * budget.GIB})
        with patch.object(fake_torch, "cuda", cuda, create=True):
            with self.assertRaisesRegex(RuntimeError, "native"):
                budget.apply_allocator_limit(c)
            with self.assertRaisesRegex(RuntimeError, "warmup allocator peak"):
                budget.check_fixed_kv_capacity(c, 21 * budget.GIB, budget.GIB)

    def test_cuda_probe_parsing_and_invalid_memory(self):
        with patch.object(budget.subprocess, "run", return_value=NS(
                stdout=budget._PROBE_PREFIX + json.dumps(
                    dict(name="A100", total=81200 * budget.MIB, free=81000 * budget.MIB)))):
            self.assertEqual(budget.query_gpu(0), ("A100",
                                                 81200 * budget.MIB, 200 * budget.MIB))
        with patch.object(budget.subprocess, "run", return_value=NS(stdout="no CUDA result")):
            with self.assertRaisesRegex(RuntimeError, "Cannot inspect"):
                budget.query_gpu(0)

    def test_probe_runs_in_separate_interpreter_with_worker_visibility(self):
        fake_module = Path(self.temp.name) / "torch.py"
        fake_module.write_text(
            "import os\nfrom types import SimpleNamespace as NS\n"
            "assert os.environ['CUDA_VISIBLE_DEVICES'] == '2'\n"
            "cuda = NS(set_device=lambda d: None,\n"
            "          mem_get_info=lambda: (81000 * 1024**2, 81200 * 1024**2),\n"
            "          get_device_name=lambda d: 'fake A100')\n")
        parent_cuda = Mock()
        with patch.dict(budget.os.environ, {"PYTHONPATH": self.temp.name,
                                           "CUDA_VISIBLE_DEVICES": "7,8"}), \
                patch.object(fake_torch, "cuda", parent_cuda, create=True):
            self.assertEqual(budget.query_gpu(2),
                             ("fake A100", 81200 * budget.MIB, 200 * budget.MIB))
            self.assertEqual(budget.os.environ["CUDA_VISIBLE_DEVICES"], "7,8")
        self.assertEqual(parent_cuda.mock_calls, [])

    def test_probe_failure_does_not_fall_back_to_nvml(self):
        error = budget.subprocess.CalledProcessError(
            1, ["python", "-c", "probe"], stderr="CUDA initialization failed")
        with patch.object(budget.subprocess, "run", side_effect=error) as run:
            with self.assertRaisesRegex(RuntimeError, "no NVML-total fallback"):
                budget.query_gpu(0)
        self.assertEqual(run.call_count, 1)

    def test_nvml_cuda_total_difference_does_not_abort_workers(self):
        # Review regression scenario: the NVML figure is NOT the CUDA budget.
        nvml_total = 81920 * budget.MIB
        for cuda_mib in [81100, 81200, 81250]:
            with self.subTest(cuda_mib=cuda_mib):
                c = self.configs()
                total = cuda_mib * budget.MIB
                free = total - 256 * budget.MIB
                output = budget._PROBE_PREFIX + json.dumps(
                    dict(name="A100", total=total, free=free))
                cuda = NS(memory=NS(get_allocator_backend=lambda: "native"),
                          mem_get_info=lambda: (free, total),
                          set_per_process_memory_fraction=Mock())
                with patch.object(budget.subprocess, "run", return_value=NS(stdout=output)), \
                        patch.object(fake_torch, "cuda", cuda, create=True), \
                        contextlib.redirect_stdout(io.StringIO()):
                    budget.prepare_device_budget([c.target, c.draft])
                    budget.apply_allocator_limit(c.draft)
                    budget.apply_allocator_limit(c.target)
                self.assertGreater(int(nvml_total * 0.9) - c.target.device_memory_limit_bytes,
                                   budget.MIB)
                self.assertEqual(c.target.device_memory_limit_bytes, int(total * 0.9))
                self.assertLessEqual(c.target.allocator_limit_bytes + c.draft.allocator_limit_bytes
                                     + 256 * budget.MIB + 2 * budget.GIB, int(total * 0.9))
                self.assertEqual(cuda.set_per_process_memory_fraction.call_count, 2)

    def test_real_cuda_capacity_change_requires_joint_replan(self):
        c = self.configs().target
        c.allocator_limit_bytes = 52 * budget.GIB
        c.device_memory_limit_bytes = 72 * budget.GIB
        cuda = NS(memory=NS(get_allocator_backend=lambda: "native"),
                  mem_get_info=lambda: (78 * budget.GIB, 79 * budget.GIB),
                  set_per_process_memory_fraction=Mock())
        with patch.object(fake_torch, "cuda", cuda, create=True):
            with self.assertRaisesRegex(RuntimeError, "decreased since the CUDA probe"):
                budget.apply_allocator_limit(c)
        cuda.set_per_process_memory_fraction.assert_not_called()

    def test_global_overrun_warnings_are_throttled_and_reset_on_recovery(self):
        c = self.configs().target
        c.device_memory_limit_bytes = 72 * budget.GIB
        cuda = NS(mem_get_info=Mock(return_value=(7 * budget.GIB, 80 * budget.GIB)))
        with patch.object(fake_torch, "cuda", cuda, create=True), \
                patch.object(budget, "monotonic", side_effect=[10, 20, 41, 42]), \
                patch.object(budget._LOGGER, "warning") as warn:
            for _ in range(3):
                budget.check_device_budget(c, "run")
            self.assertEqual(warn.call_count, 2)
            cuda.mem_get_info.return_value = (9 * budget.GIB, 80 * budget.GIB)
            budget.check_device_budget(c, "recovered")
            cuda.mem_get_info.return_value = (7 * budget.GIB, 80 * budget.GIB)
            budget.check_device_budget(c, "new overrun")
            self.assertEqual(warn.call_count, 3)

    def test_reserved_peak_detects_fragmentation_missed_by_allocated_peak(self):
        c = self.configs().target
        c.allocator_limit_bytes = 50 * budget.GIB
        cuda = NS(memory_stats=lambda: {"allocated_bytes.all.peak": 28 * budget.GIB,
                                       "reserved_bytes.all.peak": 31 * budget.GIB})
        with patch.object(fake_torch, "cuda", cuda, create=True):
            # Allocated peak + KV = 49 GiB fits; reserved peak + KV = 52 does not.
            with self.assertRaisesRegex(RuntimeError, "reserved peak=31.000"):
                budget.check_fixed_kv_capacity(c, 21 * budget.GIB, 0)


class ConfigTests(ConfigTestCase):
    def test_shared_defaults(self):
        c = self.configs()
        self.assertTrue(c.shared_gpu)
        self.assertEqual(c.target.gpu_ids, c.draft.gpu_ids)
        self.assertEqual(c.target.tensor_parallel_size, 1)
        self.assertEqual(c.target.max_num_batched_tokens, 4096)
        self.assertEqual(c.target.kv_cache_memory_bytes, budget.TARGET_KV_BYTES)

    def test_legacy_separate_gpus_still_allowed(self):
        c = configuration.EngineConfig(
            target=configuration.Config(str(self.target_path), gpu_ids=(0, 1), tensor_parallel_size=2),
            draft=configuration.Config(str(self.draft_path), name="draft", gpu_ids=(2,)))
        self.assertFalse(c.shared_gpu)
        self.assertIsNone(c.target.kv_cache_memory_bytes)
        self.assertIsNone(c.draft.kv_cache_memory_bytes)

    def test_reject_partial_overlap(self):
        with self.assertRaisesRegex(ValueError, "TP=1"):
            configuration.EngineConfig(
                target=configuration.Config(str(self.target_path), gpu_ids=(0, 1), tensor_parallel_size=2),
                draft=configuration.Config(str(self.draft_path), name="draft", gpu_ids=(1,)))

    def test_k0_single_gpu_and_invalid_utilization(self):
        c = configuration.EngineConfig(speculative_k=0,
                                      target=configuration.Config(str(self.target_path)))
        self.assertIsNone(c.draft)
        self.assertEqual(c.target.kv_cache_memory_bytes, budget.TARGET_KV_BYTES)
        for value in [0, 0.91, math.nan, math.inf]:
            with self.assertRaises(ValueError):
                self.configs(device_memory_utilization=value)

    def test_cli_overrides(self):
        with patch.object(sys, "argv", [
            "example.py", "--target-model-path", str(self.target_path),
            "--draft-model-path", str(self.draft_path),
            "--target-kv-cache-gib", "22.5", "--draft-kv-cache-gib", "15.75",
            "--gpu-memory-utilization", "0.88", "--k", "4",
        ]):
            c = cli.parse_engine_args()
        self.assertEqual(c.target.kv_cache_memory_bytes, budget.gib_to_bytes(22.5))
        self.assertEqual(c.draft.kv_cache_memory_bytes, budget.gib_to_bytes(15.75))
        self.assertEqual(c.device_memory_utilization, 0.88)
        self.assertEqual(c.target.speculative_k, 4)


class IntegrationTests(ConfigTestCase):
    def test_worker_keeps_pipe_message_protocol(self):
        c = self.configs().target
        c.num_kvcache_blocks = 543
        fake_model_module = types.ModuleType("nanovllm.engine.model_runner")
        fake_model_module.ModelRunner = Mock(return_value=NS(
            run_decode=lambda value: value + 1, exit=lambda: None))
        connection = NS(send=Mock(), recv=Mock(side_effect=[
            ("run_decode", (41,)), ("exit", ())]), close=Mock())
        with patch.dict(sys.modules, {"nanovllm.engine.model_runner": fake_model_module}), \
                patch.object(fake_torch, "cuda", NS(synchronize=Mock()), create=True), \
                patch.dict(group_module.os.environ, {}, clear=False):
            group_module._model_worker(c, 0, "file:///unused", connection)
        self.assertEqual([call.args[0] for call in connection.send.call_args_list],
                         [("ready", 543), ("result", 42), ("closed", None)])
        connection.close.assert_called_once()

    def test_both_real_allocation_methods_use_fixed_pool_sizes(self):
        # Execute the actual allocation methods with a fake tensor factory so
        # these tests cannot accidentally allocate tens of GiB on the test host.
        c = self.configs()
        for native in [False, True]:
            for config in ([c.target] if native else [c.target, c.draft]):
                name = ("nanovllm/native/engine/model_runner.py" if native
                        else "nanovllm/engine/model_runner.py")
                tree = ast.parse((ROOT / name).read_text())
                method = next(n for n in ast.walk(tree)
                              if isinstance(n, ast.FunctionDef) and n.name == "allocate_kv_cache")
                ns = dict(fixed_kv_blocks=budget.fixed_kv_blocks,
                          check_fixed_kv_capacity=budget.check_fixed_kv_capacity, GIB=budget.GIB)
                allocator = Mock(return_value=MagicMock())
                ns["torch"] = NS(
                    cuda=NS(mem_get_info=lambda: (70 * budget.GIB, 80 * budget.GIB),
                            memory_stats=lambda: {"allocated_bytes.all.peak": 1,
                                                  "allocated_bytes.all.current": 1}),
                    tensor=lambda *a, **kw: NS(item=lambda: a[0]), int64="int64", empty=allocator)
                exec(compile(ast.Module(body=[method], type_ignores=[]), name, "exec"), ns)
                scratch = 1 if config.name == "target" and not native else 0
                runner = NS(config=config, world_size=1, block_size=256,
                            use_decode_graph=config.name == "draft", use_verify_graph=bool(scratch),
                            verify_scratch_blocks=scratch, enforce_eager=False,
                            model=NS(modules=lambda: []))
                with contextlib.redirect_stdout(io.StringIO()):
                    ns["allocate_kv_cache"](runner)
                self.assertEqual(allocator.call_args.args[2], 544)
                self.assertEqual(config.num_kvcache_blocks, 544 - scratch)

    def test_shared_initialization_uses_same_runner_and_waits_between_starts(self):
        events = []
        class Runner:
            def __init__(self, config, context, init_method):
                self.config = config
                self.ready = False
                events.append(("start", config.name))
            def wait_until_ready(self):
                events.append(("ready", self.config.name))
                self.config.num_kvcache_blocks = 544
                self.ready = True
            def close(self, force=False):
                events.append(("close", self.config.name))
        tokenizer = NS(get_vocab=lambda: {"a": 1}, all_special_tokens=["eos"], eos_token_id=2)
        c = self.configs()
        with patch.object(engine_module, "ModelRunnerGroup", Runner), \
                patch.object(engine_module, "prepare_device_budget") as prepare, \
                patch.object(fake_transformers.AutoTokenizer, "from_pretrained", return_value=tokenizer):
            engine = engine_module.LLMEngine(c)
            engine.exit()
        prepare.assert_called_once_with([c.target, c.draft], 0.9, 2 * budget.GIB)
        self.assertEqual(events[:4], [("start", "draft"), ("ready", "draft"),
                                     ("start", "target"), ("ready", "target")])

    def test_step_still_submits_both_models_without_serial_gate(self):
        events = []
        runners = {name: NS(pending=False) for name in ["draft", "target"]}
        def scheduler(name):
            def schedule():
                events.append(name + ".schedule")
                runners[name].pending = True
            return NS(poll_results=lambda: events.append(name + ".poll"), schedule=schedule,
                      is_finished=lambda: False, pop_finished_outputs=lambda: [])
        engine = object.__new__(engine_module.LLMEngine)
        engine._closed = False
        engine.draft = scheduler("draft")
        engine.target = scheduler("target")
        engine.schedulers = {"draft": engine.draft, "target": engine.target}
        engine.model_runners = runners
        self.assertEqual(engine.step(), [])
        self.assertEqual(events, ["draft.poll", "target.poll", "target.schedule", "draft.schedule"])
        self.assertTrue(all(r.pending for r in runners.values()))

    def test_real_scheduler_handoff_rollback_and_release(self):
        class Runner:
            pending = False
            result = None
            def submit(self, method, *args):
                self.pending = True
                self.task = (method, args)
            def poll_results(self):
                result, self.result = self.result, None
                if result is not None:
                    self.pending = False
                return result
        c = self.configs(speculative_k=4)
        c.target.num_kvcache_blocks = 543
        c.draft.num_kvcache_blocks = 544
        tr, dr = Runner(), Runner()
        target = target_module.TargetScheduler(c.target, tr, 8, 4)
        draft = draft_module.DraftScheduler(c.draft, dr, 4, 8)
        target.peer, draft.peer = draft, target
        rid = target.add_request(list(range(255)), params.SamplingParams(max_tokens=4))
        target.schedule()
        tr.result = [(rid, 0, 1000)]
        target.poll_results()
        draft.schedule()
        self.assertEqual(dr.task[0], "run_draft_batch")
        dr.result = [(rid, [1001, 1002, 1003], 258)]
        draft.poll_results()
        target.schedule()
        # Reject after one accepted token; the correction leaves one token to emit.
        tr.result = [(rid, 1, 2000)]
        target.poll_results()
        self.assertEqual(draft.requests[rid].token_ids[-3:], [1000, 1001, 2000])
        self.assertEqual(draft.requests[rid].num_cached_tokens, 257)
        draft.schedule()
        dr.result = [(rid, [2001], 258)]
        draft.poll_results()
        target.schedule()
        tr.result = [(rid, 1, None)]
        target.poll_results()
        self.assertEqual(target.pop_finished_outputs(), [(rid, [1000, 1001, 2000, 2001])])
        self.assertTrue(target.is_finished() and draft.is_finished())
        self.assertEqual(len(target.block_manager.free_block_ids), 543)
        self.assertEqual(len(draft.block_manager.free_block_ids), 544)

    def test_k0_adapter_forwards_tp1_and_memory_caps(self):
        forwarded = {}
        upstream = types.ModuleType("nanovllm.native.engine.llm_engine")
        class Upstream:
            def __init__(self, model, **kwargs):
                forwarded.update(kwargs)
        upstream.LLMEngine = Upstream
        with patch.dict(sys.modules, {"nanovllm.native.engine.llm_engine": upstream}):
            module = load("v50_native_adapter", "nanovllm/engine/native_engine.py")
        c = configuration.EngineConfig(speculative_k=0,
                                      target=configuration.Config(str(self.target_path)))
        with patch.object(module, "prepare_device_budget") as prepare, \
                patch.dict(module.os.environ, {}, clear=False):
            module.NativeTargetEngine(c)
        prepare.assert_called_once()
        self.assertEqual(forwarded["tensor_parallel_size"], 1)
        self.assertEqual(forwarded["kv_cache_memory_bytes"], budget.TARGET_KV_BYTES)
        self.assertIs(module.NativeTP2Engine, module.NativeTargetEngine)


if __name__ == "__main__":
    unittest.main()
