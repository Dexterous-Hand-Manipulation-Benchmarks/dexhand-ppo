"""Structured Hydra schema.

Every key a run can set is declared here. Hydra validates the YAML files and command-line overrides against
these dataclasses, so a misspelled or unknown key is an error instead of being silently ignored.
"""
from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore


@dataclass
class PPOConfig:
    num_steps: int = 32                 # rollout length T per env and iteration
    num_minibatches: int = 16           # minibatches per epoch, independent of num_envs (unlike rl_games)
    num_epochs: int = 5                 # passes over each rollout batch
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 1.0             # loss = value_coef * 0.5 * MSE (rl_games: critic_coef 4 with the same 0.5)
    value_clip: bool = False            # rl_games-style clipping of the value prediction around the old value
    entropy_coef: float = 0.0
    bounds_coef: float = 1e-4           # soft penalty on action means beyond +-bounds_soft (rl_games bounds loss)
    bounds_soft: float = 1.1
    max_grad_norm: float = 1.0
    learning_rate: float = 3e-4
    lr_schedule: str = "adaptive_kl"    # adaptive_kl | constant | linear
    kl_target: float = 0.008            # adaptive_kl: divide lr by 1.5 if KL > 2*target, multiply if KL < target/2
    lr_min: float = 1e-6
    lr_max: float = 1e-2
    kl_stop: float = 0.0                # > 0: stop the epoch loop once a minibatch KL exceeds kl_stop * kl_target
    reward_scale: float = 0.1
    normalize_obs: bool = True
    obs_clip: float = 5.0
    normalize_value: bool = True
    normalize_advantage: bool = True    # per minibatch
    bootstrap_truncation: bool = True   # add gamma*V(final_obs) on truncation; only if the env reports truncations


@dataclass
class NetConfig:
    shared: bool = False                # shared actor/critic trunk (rl_games baseline) vs separate networks
    actor_hidden: list[int] = field(default_factory=lambda: [512, 256, 128])
    critic_hidden: list[int] = field(default_factory=lambda: [512, 256, 128])
    shared_hidden: list[int] = field(default_factory=lambda: [512, 512, 256, 128])
    activation: str = "elu"
    layer_norm: bool = False
    init_std: float = 1.0               # sigma_0 of the state-independent Gaussian std
    min_std: float = 0.0                # lower bound on the std (0 = none)


@dataclass
class GymEnvConfig:
    id: str = "Pendulum-v1"
    num_eval_episodes: int = 10


@dataclass
class DexMachinaEnvConfig:
    # What a run may choose of dexmachina's benchmark task; every other env setting is fixed there
    # (dexmachina.envs.benchmark_task). benchmark_spec refuses a clip that is not one of its benchmark clips.
    hand: str = "allegro_hand"
    clip: str = "box-30-230"


@dataclass
class EnvConfig:
    kind: str = "dexmachina"            # dexmachina | gym
    num_envs: int = 4096
    gym: GymEnvConfig = field(default_factory=GymEnvConfig)
    dexmachina: DexMachinaEnvConfig = field(default_factory=DexMachinaEnvConfig)


@dataclass
class EvalConfig:
    interval_steps: int = 5_000_000     # evaluate roughly every this many env steps (and at the end)
    enabled: bool = True


@dataclass
class LogConfig:
    wandb: bool = False
    project: str = "dexmachina_clean"
    entity: Optional[str] = None
    group: Optional[str] = None
    episode_window: int = 100           # train/return is the mean over the last N finished episodes


@dataclass
class CheckpointConfig:
    interval_steps: int = 5_000_000
    resume: Optional[str] = None        # path to a checkpoint to resume from


@dataclass
class Config:
    preset_name: str = "default"
    seed: int = 0
    device: str = "cuda"
    total_steps: int = 50_000_000
    exp_name: str = "ppo"
    run_dir: str = "???"
    ppo: PPOConfig = field(default_factory=PPOConfig)
    net: NetConfig = field(default_factory=NetConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    log: LogConfig = field(default_factory=LogConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


def register_configs() -> None:
    ConfigStore.instance().store(name="schema", node=Config)
