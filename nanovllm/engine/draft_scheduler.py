from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


class DraftScheduler(Scheduler):
    def __init__(self, config, runner, k, batch):
        super().__init__(config, runner, batch)
        self.k = k

    def receive_prefill(self, request_id, prompt_token_ids, first_token, sampling_params):
        """Create the draft copy only after target has produced its first token."""
        seq = Sequence(prompt_token_ids, sampling_params, self.config.kvcache_block_size, request_id)
        seq.append_token(first_token)
        self.requests[request_id] = seq
        self.prefill.append(request_id)

    def schedule(self):
        if self.runner.pending:
            return
        def task_size(seq):
            end = len(seq) + min(self.k, seq.remaining_tokens) - 1
            return end, end - seq.num_cached_tokens
        seqs = self.select_batch(self.drafting, task_size)
        if seqs:
            self.runner.submit("run_draft_batch", seqs,
                               [min(self.k, seq.remaining_tokens) for seq in seqs])

    def poll_results(self):
        results = self.runner.poll_results()
        if results is not None:
            for request_id, tokens, cached in results:
                seq = self.requests[request_id]
                seq.num_cached_tokens = cached
                seq.is_prefill = False
                seq.draft_tokens = list(tokens)
                # EOS can end the round early; return unused reserved tail blocks.
                self.block_manager.truncate(seq, cached)
                self.move_request(request_id, SequenceStatus.PAUSING)
                self.peer.receive_draft(request_id, tokens)

    def receive_target_result(self, request_id, accepted_count, next_token):
        seq = self.requests[request_id]
        end = len(seq) + accepted_count
        seq.token_ids.extend(seq.draft_tokens[:accepted_count])
        if next_token is not None:
            seq.append_token(next_token)
        self.block_manager.truncate(seq, end)
        seq.draft_tokens.clear()
        self.move_request(request_id, SequenceStatus.DRAFTING)
