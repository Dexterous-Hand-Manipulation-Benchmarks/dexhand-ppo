"""Env adapters. Imported lazily by ``make_envs`` so the core never needs genesis/dexmachina or gymnasium."""


def make_envs(cfg):
    """Returns (train_env, evaluator). The evaluator has ``evaluate(policy) -> dict`` and ``close()``."""
    if cfg.env.kind == "gym":
        from dexhand_ppo.envs.gym_env import make_gym_envs
        return make_gym_envs(cfg)
    if cfg.env.kind == "dexmachina":
        from dexhand_ppo.envs.dexmachina import make_dexmachina_envs
        return make_dexmachina_envs(cfg)
    raise ValueError(f"unknown env.kind: {cfg.env.kind}")
