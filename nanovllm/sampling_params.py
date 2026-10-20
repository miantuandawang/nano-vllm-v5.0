from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    # Kept for call-site compatibility; only temperature=0 (greedy) is supported.
    temperature: float = 0.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        if self.temperature != 0:
            raise ValueError("Only greedy sampling (temperature=0) is supported")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
