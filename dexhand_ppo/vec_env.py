"""Vectorised env interface used by the trainer, plus per-episode statistics."""
from typing import Optional, Protocol

import torch


class VecEnv(Protocol):
    """All tensors live on the env's device. Finished envs are reset inside ``step`` (same-step autoreset):
    the returned obs of a finished env is the first obs of its next episode.

    ``info`` may contain:
      - ``final_obs`` (N, obs_dim) and ``final_obs_mask`` (N,): true terminal obs of envs that finished (needed only
        for truncation bootstrapping);
      - ``episode``: dict of 1-D tensors with one entry per episode that ended in this step (from EpisodeTracker).
    """

    num_envs: int
    obs_dim: int
    action_dim: int
    device: torch.device

    def reset(self) -> torch.Tensor: ...

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]: ...

    def on_iteration(self, global_step: int) -> dict: ...

    def close(self) -> None: ...


class EpisodeTracker:
    """Accumulates per-env episode sums (return, length and optional extra terms) and emits finished episodes."""

    def __init__(self, num_envs: int, device, extra_keys: tuple[str, ...] = ()):
        self.extra_keys = extra_keys
        self.returns = torch.zeros(num_envs, device=device)
        self.lengths = torch.zeros(num_envs, device=device)
        self.extras = {k: torch.zeros(num_envs, device=device) for k in extra_keys}

    def step(self, rewards: torch.Tensor, dones: torch.Tensor, extras: Optional[dict] = None) -> dict:
        self.returns += rewards
        self.lengths += 1
        for k in self.extra_keys:
            if extras is not None and k in extras:
                self.extras[k] += extras[k]
        done_idx = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_idx.numel() == 0:
            return {}
        episode = {"return": self.returns[done_idx].clone(), "length": self.lengths[done_idx].clone()}
        for k in self.extra_keys:  # per-step mean of each extra term over the episode
            episode[k] = self.extras[k][done_idx] / self.lengths[done_idx]
            self.extras[k][done_idx] = 0.0
        self.returns[done_idx] = 0.0
        self.lengths[done_idx] = 0.0
        return episode
