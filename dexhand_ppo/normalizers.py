"""Running mean/std normalisers for observations and value targets (saved in checkpoints via buffers)."""
import torch
from torch import nn


class RunningMeanStd(nn.Module):
    """Per-feature running mean and variance, merged batch-wise (Chan et al. parallel variant of Welford)."""

    def __init__(self, shape, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float64))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, *self.mean.shape).to(torch.float64)
        batch_count = x.shape[0]
        if batch_count == 0:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta.pow(2) * self.count * batch_count / total
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total)
        self.count.copy_(total)

    def normalize(self, x: torch.Tensor, clip: float = 0.0) -> torch.Tensor:
        std = torch.sqrt(self.var + self.epsilon).to(x.dtype)
        out = (x - self.mean.to(x.dtype)) / std
        return out.clamp(-clip, clip) if clip > 0 else out

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(self.var + self.epsilon).to(x.dtype)
        return x * std + self.mean.to(x.dtype)


class Identity(nn.Module):
    """Stand-in when normalisation is disabled, with the same interface."""

    def update(self, x: torch.Tensor) -> None:
        pass

    def normalize(self, x: torch.Tensor, clip: float = 0.0) -> torch.Tensor:
        return x.clamp(-clip, clip) if clip > 0 else x

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x
