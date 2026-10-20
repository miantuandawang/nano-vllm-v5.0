from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 0.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature == 0, "only greedy temperature=0 is supported"
