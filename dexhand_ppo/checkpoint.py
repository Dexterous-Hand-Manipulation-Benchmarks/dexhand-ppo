"""Checkpoints: model, optimiser, normalisers, progress counters and env state (e.g. curriculum)."""
from pathlib import Path

import torch


def save_checkpoint(path, *, model, optimizer, obs_norm, value_norm, iteration: int, global_step: int,
                    learning_rate: float, num_updates: int, env_state: dict, resolved_cfg: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "obs_norm": obs_norm.state_dict(), "value_norm": value_norm.state_dict(),
        "iteration": iteration, "global_step": global_step, "learning_rate": learning_rate,
        "num_updates": num_updates, "env_state": env_state, "config": resolved_cfg,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic: a job killed mid-save never leaves a truncated checkpoint


def load_checkpoint(path, device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)
