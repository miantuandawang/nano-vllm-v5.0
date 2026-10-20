from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


class TargetScheduler(Scheduler):
    def __init__(self, config, runner, batch, k):
        super().__init__(config, runner, batch)
        self.k = k
        self.finished_outputs = []

    def add_request(self, prompt_token_ids, sampling_params):
        seq = Sequence(prompt_token_ids, sampling_params, self.config.kvcache_block_size)
        self.requests[seq.request_id] = seq
        self.prefill.append(seq.request_id)
        return seq.request_id

    def schedule(self):
        if self.runner.pending:
            return
        def task_size(seq):
            end = len(seq) + len(seq.draft_tokens)
            return end, end - seq.num_cached_tokens
        # In K=0 mode running requests continue normal decoding until completion.
        priority = self.verify if self.k else self.running
        seqs = self.select_batch(priority, task_size)
        if seqs:
            self.runner.submit("run_target_batch" if self.k else "run_decode", seqs)

    def poll_results(self):
        results = self.runner.poll_results()
        if results is not None:
            for request_id, accepted, token in results:
                seq = self.requests[request_id]
                if seq.is_prefill or not self.k:
                    self.on_decode_done(request_id, token)
                else:
                    self.on_verify_done(request_id, accepted, token)

    def on_decode_done(self, request_id, token):
        seq = self.requests[request_id]
        seq.num_cached_tokens = len(seq)
        seq.is_prefill = False
        seq.append_token(token)
        if seq.is_done(self.config.eos):
            self.finish_request(request_id)
        elif self.k:
            self.move_request(request_id, SequenceStatus.PAUSING)
            self.peer.receive_prefill(
                request_id, seq.token_ids[:seq.num_prompt_tokens], token,
                SamplingParams(max_tokens=seq.max_tokens, ignore_eos=seq.ignore_eos),
            )
        # K=0: keep this active request in running for its next decode step.

    def receive_draft(self, request_id, tokens):
        self.requests[request_id].draft_tokens = list(tokens)
        self.move_request(request_id, SequenceStatus.VERIFY)

    def on_verify_done(self, request_id, accepted, token):
        seq = self.requests[request_id]
        end = len(seq) + accepted
        seq.num_cached_tokens = len(seq) + len(seq.draft_tokens)
        seq.token_ids.extend(seq.draft_tokens[:accepted])
        if token is not None:
            seq.append_token(token)
        self.block_manager.truncate(seq, end)
        if seq.is_done(self.config.eos):
            self.finish_request(request_id)
        else:
            self.move_request(request_id, SequenceStatus.PAUSING)
            self.peer.receive_target_result(request_id, accepted, token)
            seq.draft_tokens.clear()

    def finish_request(self, request_id):
        self.finished_outputs.append((request_id, list(self.requests[request_id].completion_token_ids)))
        # Initial EOS/output-limit completion has no draft copy to release.
        if self.peer is not None and request_id in self.peer.requests:
            self.peer.finish_request(request_id)
        super().finish_request(request_id)

    def pop_finished_outputs(self):
        outputs, self.finished_outputs = self.finished_outputs, []
        return outputs
