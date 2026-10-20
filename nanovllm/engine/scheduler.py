from collections import deque

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import SequenceStatus


class Scheduler:
    def __init__(self, config, runner, batch):
        self.config = config
        self.name = config.name
        self.runner = runner
        self.batch = min(batch, config.max_num_seqs)
        self.peer = None
        self.requests = {}
        self.prefill = deque()
        self.verify = deque()
        self.drafting = deque()
        self.pausing = deque()
        self.running = deque()
        self.queues = {status: getattr(self, status.name.lower()) for status in (
            SequenceStatus.PREFILL, SequenceStatus.VERIFY, SequenceStatus.DRAFTING,
            SequenceStatus.PAUSING, SequenceStatus.RUNNING,
        )}
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)

    def move_request(self, request_id, status):
        seq = self.requests[request_id]
        self.queues[seq.status].remove(request_id)
        seq.status = status
        self.queues[status].append(request_id)

    def select_batch(self, priority_queue, task_size):
        selected = []
        budget = self.config.max_num_batched_tokens

        def take(queue):
            nonlocal budget
            deferred = False
            for request_id in list(queue):
                if len(selected) == self.batch:
                    return True
                seq = self.requests[request_id]
                end, cost = task_size(seq)
                if cost > self.config.max_num_batched_tokens:
                    raise ValueError(f"{self.name}: request {request_id} exceeds token budget")
                if cost > budget or not self.block_manager.reserve(seq, end):
                    deferred = True
                    continue
                selected.append(seq)
                budget -= cost
                # K=0 continuations already belong to running. Rotate them for fairness.
                self.move_request(request_id, SequenceStatus.RUNNING)
            return deferred

        # New prefill may fill spare slots only after all ready high-priority work
        # has been selected. A budget-blocked continuation must not lose capacity
        # to newly admitted requests.
        deferred = take(priority_queue)
        if not deferred and len(selected) < self.batch:
            take(self.prefill)
        return selected

    def finish_request(self, request_id):
        seq = self.requests[request_id]
        self.queues[seq.status].remove(request_id)
        self.block_manager.deallocate(seq)
        seq.draft_tokens.clear()
        seq.status = SequenceStatus.FINISHED
        del self.requests[request_id]

    def is_finished(self):
        return not self.requests
