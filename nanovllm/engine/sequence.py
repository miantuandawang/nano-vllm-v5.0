from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    PREFILL = auto()
    VERIFY = auto()
    DRAFTING = auto()
    RUNNING = auto()
    PAUSING = auto()
    FINISHED = auto()


class Sequence:
    counter = count()

    def __init__(self, token_ids, sampling_params=None, block_size=256, request_id=None):
        params = sampling_params or SamplingParams()
        self.request_id = next(self.counter) if request_id is None else request_id
        self.status = SequenceStatus.PREFILL
        self.is_prefill = True
        self.token_ids = list(token_ids)
        self.draft_tokens = []
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_size = block_size
        self.block_table = []
        self.max_tokens = params.max_tokens
        self.ignore_eos = params.ignore_eos

    def __len__(self):
        return len(self.token_ids)

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def remaining_tokens(self):
        return self.max_tokens - len(self.completion_token_ids)

    def append_token(self, token):
        self.token_ids.append(token)

    def is_done(self, eos):
        return self.remaining_tokens == 0 or (not self.ignore_eos and self.token_ids[-1] == eos)

    # Default pickle transfers the complete request state, including candidate tokens.
