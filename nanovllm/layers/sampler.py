from torch import nn


class Sampler(nn.Module):
    def forward(self, logits):
        return logits.argmax(dim=-1)
