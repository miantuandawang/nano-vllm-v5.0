from datetime import timedelta

import torch
import torch.distributed as dist

from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import set_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.utils.memory_budget import (
    apply_allocator_limit, check_device_budget, check_fixed_kv_capacity,
    fixed_kv_blocks, GIB,
)


class ModelRunner:
    """Eager prefill with native-style draft decode and target verify graphs."""

    def __init__(self, config, rank, init_method):
        self.config = config
        self.rank = rank
        self.world_size = config.tensor_parallel_size
        self.block_size = config.kvcache_block_size
        self.use_decode_graph = config.name == "draft" and not config.enforce_eager
        self.verify_query_len = config.speculative_k + 1
        self.verify_graph_max_bs = min(config.max_num_seqs, 512,
                                       config.max_num_batched_tokens // self.verify_query_len)
        self.use_verify_graph = (config.name == "target" and config.target_verify_graph
                                 and config.speculative_k > 0 and self.verify_graph_max_bs > 0
                                 and self.verify_query_len <= config.max_model_len)
        self.verify_scratch_blocks = ((self.verify_query_len + self.block_size - 1) // self.block_size
                                      if self.use_verify_graph else 0)
        torch.cuda.set_device(rank)
        apply_allocator_limit(config)
        dist.init_process_group("nccl", init_method=init_method,
                                world_size=self.world_size, rank=rank,
                                timeout=timedelta(seconds=config.worker_timeout))
        dtype = torch.get_default_dtype()
        torch.set_default_dtype(config.hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(config.hf_config)
        load_model(self.model, config.model)
        self.warmup_model()
        self.allocate_kv_cache()
        if self.use_decode_graph:
            self.capture_cudagraph()
        if self.use_verify_graph:
            self.capture_verify_graphs()
        check_device_budget(config, "model ready")
        torch.set_default_device("cpu")
        torch.set_default_dtype(dtype)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        length = min(self.config.max_num_batched_tokens, self.config.max_model_len)
        count = min(self.config.max_num_batched_tokens // length, self.config.max_num_seqs)
        seqs = [Sequence([0] * length, block_size=self.block_size) for _ in range(count)]
        self._predict(seqs, [seq.token_ids for seq in seqs], all_positions=True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        heads = hf.num_key_value_heads // self.world_size
        dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        block_bytes = 2 * hf.num_hidden_layers * self.block_size * heads * dim * hf.dtype.itemsize
        # Reserve graph buffers/private-pool headroom before sizing the global KV
        # pool. This is not a per-request maximum-length reservation.
        graph_reserve = config.graph_memory_reserve_mb * 1024**2 if self.use_decode_graph or self.use_verify_graph else 0
        if config.kv_cache_memory_bytes is not None:
            count = fixed_kv_blocks(config, self.verify_scratch_blocks)
            check_fixed_kv_capacity(config, count * block_bytes, graph_reserve)
        else:
            count = int(total * config.gpu_memory_utilization - used - peak + current - graph_reserve) // block_bytes
        blocks = torch.tensor(count, dtype=torch.int64, device="cuda")
        if self.world_size > 1:
            dist.all_reduce(blocks, op=dist.ReduceOp.MIN)
        total_blocks = blocks.item()
        config.num_kvcache_blocks = total_blocks - self.verify_scratch_blocks
        if config.num_kvcache_blocks <= 0:
            raise RuntimeError(f"{config.name}: insufficient memory for KV cache")
        print(f"[V5.0 KV] {config.name}: TP={self.world_size}, "
              f"allocated={total_blocks * block_bytes / GIB:.6f} GiB, "
              f"blocks={total_blocks}, usable={config.num_kvcache_blocks}, "
              f"scratch={self.verify_scratch_blocks}", flush=True)
        self.kv_cache = torch.empty(2, hf.num_hidden_layers, total_blocks,
                                    self.block_size, heads, dim,
                                    dtype=hf.dtype, device="cuda")
        if self.verify_scratch_blocks:
            self.kv_cache[:, :, config.num_kvcache_blocks:].zero_()
        layer = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer]
                module.v_cache = self.kv_cache[1, layer]
                layer += 1

    @torch.inference_mode()
    def _predict(self, seqs, token_lists, all_positions=False, prediction_starts=None):
        """Append all uncached tokens; project only each query's requested suffix.

        prediction_starts contains query-relative offsets, one per sequence.
        Without it, all_positions selects either the full query or its last row.
        """
        ids, positions, slots = [], [], []
        cu_q, cu_k = [0], [0]
        lengths = []
        for seq, tokens in zip(seqs, token_lists):
            start, end = seq.num_cached_tokens, len(tokens)
            ids.extend(tokens[start:end])
            positions.extend(range(start, end))
            lengths.append(end - start)
            cu_q.append(cu_q[-1] + end - start)
            cu_k.append(cu_k[-1] + end)
            slots.extend(seq.block_table[p // self.block_size] * self.block_size + p % self.block_size
                         if seq.block_table else -1 for p in range(start, end))
        def tensor(values, dtype=torch.int32):
            return torch.tensor(values, dtype=dtype, device="cuda")
        tables = None
        if any(seq.num_cached_tokens for seq in seqs):
            width = max(len(seq.block_table) for seq in seqs)
            tables = tensor([seq.block_table + [-1] * (width - len(seq.block_table)) for seq in seqs])
        set_context(True, tensor(cu_q), tensor(cu_k), max(lengths),
                    max(map(len, token_lists)), tensor(slots), None, tables)
        try:
            hidden = self.model(tensor(ids, torch.int64), tensor(positions, torch.int64))
            if prediction_starts is None:
                prediction_starts = [0 if all_positions else length - 1 for length in lengths]
            prediction_counts = [length - start for length, start in zip(lengths, prediction_starts)]
            if any(prediction_starts):
                rows = [row for i, start in enumerate(prediction_starts)
                        for row in range(cu_q[i] + start, cu_q[i + 1])]
                hidden = hidden[tensor(rows, torch.int64)]
            predictions = self._greedy_predictions(hidden)
            results = []
            offset = 0
            for count in prediction_counts:
                results.append(predictions[offset:offset + count])
                offset += count
            return results
        finally:
            reset_context()

    @torch.inference_mode()
    def _greedy_predictions(self, hidden):
        """Vocabulary projection, TP gather/broadcast and CPU results stay outside capture."""
        predictions = []
        for chunk in hidden.split(128):
            logits = self.model.compute_logits(chunk, all_positions=True)
            tokens = (logits.argmax(dim=-1) if self.rank == 0 else
                      torch.empty(chunk.shape[0], dtype=torch.int64, device="cuda"))
            if self.world_size > 1:
                dist.broadcast(tokens, src=0)
            predictions.extend(tokens.tolist())
        return predictions

    def _set_verify_graph_context(self, bucket):
        buffers = self.verify_graph_vars
        num_tokens = bucket * self.verify_query_len
        set_context(
            True,
            cu_seqlens_q=buffers["cu_seqlens_q"][:bucket + 1],
            cu_seqlens_k=buffers["cu_seqlens_k"][:bucket + 1],
            max_seqlen_q=self.verify_query_len,
            max_seqlen_k=self.config.max_model_len,
            slot_mapping=buffers["slot_mapping"][:num_tokens],
            block_tables=buffers["block_tables"][:bucket],
        )

    @torch.inference_mode()
    def capture_verify_graphs(self):
        """Native batch buckets/shared pool, with K+1 query tokens per request."""
        max_bs = self.verify_graph_max_bs
        qlen = self.verify_query_len
        max_tokens = max_bs * qlen
        max_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size
        sizes = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.verify_graph_bs = sorted({bs for bs in sizes if bs <= max_bs} | {max_bs})
        def zeros(*shape, dtype=torch.int32):
            return torch.zeros(*shape, dtype=dtype, device="cuda")
        # Scratch IDs are outside the allocator's [0, num_kvcache_blocks) range.
        scratch_start = self.config.num_kvcache_blocks
        block_tables = torch.full((max_bs, max_blocks), scratch_start,
                                  dtype=torch.int32, device="cuda")
        for column in range(self.verify_scratch_blocks):
            block_tables[:, column] = scratch_start + column
        self.verify_graph_vars = dict(
            input_ids=zeros(max_tokens, dtype=torch.int64),
            positions=zeros(max_tokens, dtype=torch.int64),
            slot_mapping=torch.full((max_tokens,), -1, dtype=torch.int32, device="cuda"),
            block_tables=block_tables,
            cu_seqlens_q=torch.arange(max_bs + 1, dtype=torch.int32, device="cuda") * qlen,
            cu_seqlens_k=torch.arange(max_bs + 1, dtype=torch.int32, device="cuda") * qlen,
            outputs=zeros(max_tokens, self.config.hf_config.hidden_size,
                          dtype=self.config.hf_config.dtype),
        )
        self.verify_graphs = {}
        self.verify_graph_pool = None
        buffers = self.verify_graph_vars
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        try:
            for bs in reversed(self.verify_graph_bs):
                count = bs * qlen
                self._set_verify_graph_context(bs)
                with torch.cuda.stream(warmup_stream):
                    for _ in range(3):
                        buffers["outputs"][:count] = self.model(
                            buffers["input_ids"][:count], buffers["positions"][:count])
                torch.cuda.current_stream().wait_stream(warmup_stream)
                torch.cuda.synchronize()
                # All TP ranks capture the same collective sequence and shape.
                if self.world_size > 1:
                    dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, self.verify_graph_pool):
                    buffers["outputs"][:count] = self.model(
                        buffers["input_ids"][:count], buffers["positions"][:count])
                if self.verify_graph_pool is None:
                    self.verify_graph_pool = graph.pool()
                self.verify_graphs[bs] = graph
                torch.cuda.synchronize()
        finally:
            reset_context()

    def prepare_verify_graph_inputs(self, seqs):
        bs = len(seqs)
        bucket = next(size for size in self.verify_graph_bs if size >= bs)
        qlen = self.verify_query_len
        count = bucket * qlen
        buffers = self.verify_graph_vars
        scratch_start = self.config.num_kvcache_blocks
        # Clear the WHOLE bucket to handle shrinking batches and shorter tables.
        buffers["input_ids"][:count].zero_()
        buffers["positions"][:count].zero_()
        buffers["slot_mapping"][:count].fill_(-1)
        buffers["block_tables"][:bucket].fill_(scratch_start)
        for column in range(self.verify_scratch_blocks):
            buffers["block_tables"][:bucket, column].fill_(scratch_start + column)

        ids, positions, slots = [], [], []
        cu_k = [0]
        width = max(len(seq.block_table) for seq in seqs)
        tables = []
        for seq in seqs:
            tokens = seq.token_ids + seq.draft_tokens
            start, end = seq.num_cached_tokens, len(tokens)
            ids.extend(tokens[start:end])
            positions.extend(range(start, end))
            slots.extend(seq.block_table[p // self.block_size] * self.block_size + p % self.block_size
                         for p in range(start, end))
            cu_k.append(cu_k[-1] + end)
            tables.append(seq.block_table + [scratch_start] * (width - len(seq.block_table)))
        # Dummy rows attend to qlen zero KV entries, never real requests' blocks.
        for _ in range(bucket - bs):
            cu_k.append(cu_k[-1] + qlen)

        def upload(destination, values, dtype=torch.int32):
            cpu = torch.tensor(values, dtype=dtype, device="cpu", pin_memory=True)
            destination.copy_(cpu, non_blocking=True)
        upload(buffers["input_ids"][:bs * qlen], ids, torch.int64)
        upload(buffers["positions"][:bs * qlen], positions, torch.int64)
        upload(buffers["slot_mapping"][:bs * qlen], slots)
        upload(buffers["cu_seqlens_k"][:bucket + 1], cu_k)
        upload(buffers["block_tables"][:bs, :width], tables)
        return bucket

    @torch.inference_mode()
    def replay_verify_graph(self, seqs):
        bucket = self.prepare_verify_graph_inputs(seqs)
        self._set_verify_graph_context(bucket)
        try:
            self.verify_graphs[bucket].replay()
            qlen = self.verify_query_len
            # Padding hidden states never enter the LM head or acceptance logic.
            predicted = self._greedy_predictions(self.verify_graph_vars["outputs"][:len(seqs) * qlen])
            return [predicted[i * qlen:(i + 1) * qlen] for i in range(len(seqs))]
        finally:
            reset_context()


    @torch.inference_mode()
    def capture_cudagraph(self):
        """Native-style single-token backbone capture, largest bucket first."""
        max_bs = min(self.config.max_num_seqs, 512)
        max_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size
        buckets = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graph_bs = sorted({bs for bs in buckets if bs <= max_bs} | {max_bs})
        def zeros(*shape, dtype=torch.int32):
            return torch.zeros(*shape, dtype=dtype, device="cuda")
        self.graph_vars = dict(
            input_ids=zeros(max_bs, dtype=torch.int64),
            positions=zeros(max_bs, dtype=torch.int64),
            slot_mapping=torch.full((max_bs,), -1, dtype=torch.int32, device="cuda"),
            context_lens=zeros(max_bs),
            block_tables=zeros(max_bs, max_blocks),
            outputs=zeros(max_bs, self.config.hf_config.hidden_size,
                          dtype=self.config.hf_config.dtype),
        )
        self.graphs = {}
        self.graph_pool = None
        buffers = self.graph_vars
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        try:
            for bs in reversed(self.graph_bs):
                set_context(False, slot_mapping=buffers["slot_mapping"][:bs],
                            context_lens=buffers["context_lens"][:bs],
                            block_tables=buffers["block_tables"][:bs])
                # Compile/initialize kernels outside capture, on a side stream.
                with torch.cuda.stream(warmup_stream):
                    for _ in range(3):
                        buffers["outputs"][:bs] = self.model(
                            buffers["input_ids"][:bs], buffers["positions"][:bs])
                torch.cuda.current_stream().wait_stream(warmup_stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, self.graph_pool):
                    buffers["outputs"][:bs] = self.model(
                        buffers["input_ids"][:bs], buffers["positions"][:bs])
                if self.graph_pool is None:
                    self.graph_pool = graph.pool()
                self.graphs[bs] = graph
                torch.cuda.synchronize()
        finally:
            reset_context()

    def prepare_decode(self, seqs, token_lists):
        """Fill static buffers; the caller guarantees one uncached token per row."""
        bs = len(seqs)
        bucket = next(size for size in self.graph_bs if size >= bs)
        buffers = self.graph_vars
        positions = [len(tokens) - 1 for tokens in token_lists]
        slots = [seq.block_table[pos // self.block_size] * self.block_size + pos % self.block_size
                 for seq, pos in zip(seqs, positions)]
        width = max(len(seq.block_table) for seq in seqs)
        tables = [seq.block_table + [0] * (width - len(seq.block_table)) for seq in seqs]

        # Reset the entire replay bucket so shrinking batches cannot reuse old rows.
        buffers["input_ids"][:bucket].zero_()
        buffers["positions"][:bucket].zero_()
        buffers["slot_mapping"][:bucket].fill_(-1)
        buffers["context_lens"][:bucket].zero_()
        buffers["block_tables"][:bucket].zero_()
        def copy(name, values, dtype=torch.int32):
            cpu = torch.tensor(values, dtype=dtype, device="cpu", pin_memory=True)
            if name == "block_tables":
                buffers[name][:bs, :width].copy_(cpu, non_blocking=True)
            else:
                buffers[name][:bs].copy_(cpu, non_blocking=True)
        copy("input_ids", [tokens[-1] for tokens in token_lists], torch.int64)
        copy("positions", positions, torch.int64)
        copy("slot_mapping", slots)
        copy("context_lens", [len(tokens) for tokens in token_lists])
        copy("block_tables", tables)
        return bucket

    @torch.inference_mode()
    def replay_decode_graph(self, seqs, token_lists):
        bs = len(seqs)
        bucket = self.prepare_decode(seqs, token_lists)
        buffers = self.graph_vars
        set_context(False, slot_mapping=buffers["slot_mapping"][:bucket],
                    context_lens=buffers["context_lens"][:bucket],
                    block_tables=buffers["block_tables"][:bucket])
        try:
            self.graphs[bucket].replay()
            # Match upstream capture boundaries: vocabulary projection and sampling
            # stay outside the graph and operate only on real requests.
            logits = self.model.compute_logits(buffers["outputs"][:bs], all_positions=True)
            tokens = (logits.argmax(dim=-1) if self.rank == 0 else
                      torch.empty(bs, dtype=torch.int64, device="cuda"))
            if self.world_size > 1:
                dist.broadcast(tokens, src=0)
            return [[token] for token in tokens.tolist()]
        finally:
            reset_context()


    def run_prefill(self, seqs):
        predictions = self._predict(seqs, [seq.token_ids for seq in seqs])
        return [(seq.request_id, prediction[0]) for seq, prediction in zip(seqs, predictions)]

    def run_draft_batch(self, seqs, limits):
        """Prefill plus K candidates, or a continuing K-candidate round, in one task."""
        work = {seq.request_id: list(seq.token_ids) for seq in seqs}
        candidates = {seq.request_id: [] for seq in seqs}
        budgets = {seq.request_id: limit for seq, limit in zip(seqs, limits)}
        active = list(seqs)
        while active:
            token_lists = [work[seq.request_id] for seq in active]
            single_token = all(len(tokens) - seq.num_cached_tokens == 1
                               for seq, tokens in zip(active, token_lists))
            if self.use_decode_graph and single_token and len(active) <= self.graph_bs[-1]:
                predictions = self.replay_decode_graph(active, token_lists)
            else:
                # New prompts or rollback catch-up may contain multiple tokens.
                predictions = self._predict(active, token_lists)
            next_active = []
            for seq, prediction in zip(active, predictions):
                request_id = seq.request_id
                seq.num_cached_tokens = len(work[request_id])
                token = prediction[0]
                work[request_id].append(token)
                candidates[request_id].append(token)
                if (len(candidates[request_id]) < budgets[request_id]
                        and (seq.ignore_eos or token != self.config.eos)):
                    next_active.append(seq)
            active = next_active
        return [(seq.request_id, candidates[seq.request_id], seq.num_cached_tokens) for seq in seqs]

    def run_decode(self, seqs):
        """K=0: ordinary greedy TP decode, with new prompt prefill filling spare slots."""
        predictions = self._predict(seqs, [seq.token_ids for seq in seqs])
        return [(seq.request_id, 0, predicted[0]) for seq, predicted in zip(seqs, predictions)]

    def run_target_batch(self, seqs):
        """Split an already capped dispatch into graph verify and eager work."""
        graph_seqs, eager_seqs = [], []
        for seq in seqs:
            eligible = (self.use_verify_graph and not seq.is_prefill
                        and len(seq.draft_tokens) == self.config.speculative_k
                        and len(seq) - seq.num_cached_tokens == 1)
            (graph_seqs if eligible else eager_seqs).append(seq)
        # Follow the native >512 fallback; never add requests to fill a bucket.
        if self.use_verify_graph and len(graph_seqs) > self.verify_graph_max_bs:
            eager_seqs = list(seqs)
            graph_seqs = []
        predictions = {}
        if graph_seqs:
            predictions.update(zip((seq.request_id for seq in graph_seqs),
                                   self.replay_verify_graph(graph_seqs)))
        if eager_seqs:
            inputs = [seq.token_ids + seq.draft_tokens for seq in eager_seqs]
            # Prefill needs only its final prediction. Eager verify retains all
            # query predictions, including any catch-up prefix used below.
            prediction_starts = [len(tokens) - seq.num_cached_tokens - 1 if seq.is_prefill else 0
                                 for seq, tokens in zip(eager_seqs, inputs)]
            predictions.update(zip((seq.request_id for seq in eager_seqs),
                                   self._predict(eager_seqs, inputs,
                                                 prediction_starts=prediction_starts)))
        results = []
        for seq in seqs:
            predicted = predictions[seq.request_id]
            if seq.is_prefill:
                results.append((seq.request_id, 0, predicted[-1]))
                continue
            offset = len(seq) - seq.num_cached_tokens - 1
            predicted = predicted[offset:]
            accepted = 0
            next_token = None
            for candidate, token in zip(seq.draft_tokens, predicted):
                if candidate != token:
                    next_token = token
                    break
                accepted += 1
                if accepted == seq.remaining_tokens or (not seq.ignore_eos and candidate == self.config.eos):
                    break
            else:
                if accepted < seq.remaining_tokens:
                    next_token = predicted[len(seq.draft_tokens)]
            results.append((seq.request_id, accepted, next_token))
        return results

    def run_verify(self, seqs):
        return self.run_target_batch(seqs)

    def exit(self):
        torch.cuda.synchronize()
        if self.use_decode_graph:
            self.graphs.clear()
            self.graph_pool = None
            self.graph_vars.clear()
        if self.use_verify_graph:
            self.verify_graphs.clear()
            self.verify_graph_pool = None
            self.verify_graph_vars.clear()
        dist.destroy_process_group()
