import os
import traceback
from multiprocessing.connection import wait
from time import monotonic

from nanovllm.config import Config
from nanovllm.utils.memory_budget import check_device_budget


def _model_worker(config: Config, rank: int, init_method: str, connection):
    # Each spawned process owns exactly one NCCL group and CUDA context.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.gpu_ids))
    try:
        from nanovllm.engine.model_runner import ModelRunner

        runner = ModelRunner(config, rank, init_method)
        connection.send(("ready", config.num_kvcache_blocks))
        while True:
            method, args = connection.recv()
            if method == "exit":
                runner.exit()
                connection.send(("closed", None))
                break
            if method not in ("run_prefill", "run_draft_batch", "run_verify", "run_target_batch", "run_decode"):
                raise ValueError(f"Unknown worker command: {method}")
            check_device_budget(config, f"before {method}")
            result = getattr(runner, method)(*args)
            # Do not return blocks to the scheduler until all local GPU work is done.
            import torch
            torch.cuda.synchronize()
            check_device_budget(config, f"after {method}")
            connection.send(("result", result))
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class ModelRunnerGroup:
    """One process per TP rank; all IPC is private to this model group."""

    def __init__(self, config: Config, context, init_method: str):
        self.config = config
        self.processes = []
        self.connections = []
        self.ready = False
        self.pending = False
        self.closed = False
        try:
            for rank in range(config.tensor_parallel_size):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_model_worker,
                    name=f"nanovllm-{config.name}-rank{rank}",
                    args=(config, rank, init_method, child),
                )
                # spawn imports the main module before entering _model_worker.
                # Set visibility before spawning so import-time CUDA discovery
                # cannot bind a draft worker to target GPU 0.
                previous_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.gpu_ids))
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    raise
                finally:
                    if previous_visibility is None:
                        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                    else:
                        os.environ["CUDA_VISIBLE_DEVICES"] = previous_visibility
                    child.close()
                self.connections.append(parent)
                self.processes.append(process)
        except BaseException:
            self.close(force=True)
            raise

    def _collect(self, expected: str, timeout: float | None = None):
        deadline = monotonic() + (self.config.worker_timeout if timeout is None else timeout)
        pending = set(range(len(self.connections)))
        results = [None] * len(self.connections)
        while pending:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{self.config.name}: timed out waiting for {expected}")
            handles = [self.connections[i] for i in pending]
            handles += [self.processes[i].sentinel for i in pending]
            available = wait(handles, timeout=remaining)
            for rank in list(pending):
                connection = self.connections[rank]
                process = self.processes[rank]
                if connection in available or connection.poll():
                    try:
                        status, payload = connection.recv()
                    except (EOFError, OSError) as exc:
                        raise RuntimeError(f"{self.config.name} rank {rank}: worker disconnected") from exc
                    if status == "error":
                        raise RuntimeError(f"{self.config.name} rank {rank}:\n{payload}")
                    if status != expected:
                        raise RuntimeError(f"{self.config.name} rank {rank}: expected {expected}, got {status}")
                    results[rank] = payload
                    pending.remove(rank)
                elif process.sentinel in available:
                    raise RuntimeError(f"{self.config.name} rank {rank}: worker exited unexpectedly")
        return results

    def wait_until_ready(self):
        blocks = self._collect("ready")
        if len(set(blocks)) != 1:
            raise RuntimeError(f"{self.config.name}: TP ranks have inconsistent KV cache capacities")
        self.config.num_kvcache_blocks = blocks[0]
        self.ready = True

    def submit(self, method, *args):
        if not self.ready or self.closed or self.pending:
            raise RuntimeError(f"{self.config.name}: model group is not ready")
        self.pending = True
        self._remaining = set(range(len(self.connections)))
        self._results = [None] * len(self.connections)
        self._deadline = monotonic() + self.config.worker_timeout
        for connection in self.connections:
            connection.send((method, args))

    def poll_results(self):
        if not self.pending:
            return None
        for rank in list(self._remaining):
            connection = self.connections[rank]
            if connection.poll():
                status, payload = connection.recv()
                if status == "error":
                    raise RuntimeError(f"{self.config.name} rank {rank}:\n{payload}")
                if status != "result":
                    raise RuntimeError(f"Unexpected worker response: {status}")
                self._results[rank] = payload
                self._remaining.remove(rank)
            elif not self.processes[rank].is_alive():
                raise RuntimeError(f"{self.config.name} rank {rank} exited")
        if not self._remaining:
            self.pending = False
            return self._results[0]
        if monotonic() >= self._deadline:
            raise TimeoutError(f"{self.config.name}: model task timed out")
        return None

    def wait_handles(self):
        if not self.pending:
            return []
        return ([self.connections[i] for i in self._remaining]
                + [self.processes[i].sentinel for i in self._remaining])

    def close(self, force=False):
        if self.closed:
            return
        self.closed = True
        if not force and self.ready and not self.pending:
            try:
                for connection in self.connections:
                    connection.send(("exit", ()))
                self._collect("closed", timeout=10)
                for process in self.processes:
                    process.join(timeout=1)
            except (RuntimeError, TimeoutError, EOFError, OSError):
                pass
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        for connection in self.connections:
            connection.close()
