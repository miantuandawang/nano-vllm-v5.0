import torch
from torch import nn


class Sampler(nn.Module):
    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        return logits.argmax(dim=-1)
