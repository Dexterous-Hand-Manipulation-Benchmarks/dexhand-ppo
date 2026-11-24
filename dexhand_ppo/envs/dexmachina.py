"""DexMachina adapter: the benchmark task, done handling, curriculum, metrics and the shared evaluation.

The env is dexmachina's benchmark task (``dexmachina.envs.benchmark_task``): a run chooses hand, clip, seed and the
number of envs; every other env setting, including the share of random start frames, is fixed there, so every
method trains on the same task. Training steps go through dexmachina's ``DexMachinaVecEnv``, evaluation through the
team's shared protocol (``make_eval_env``, ``rollout_episode``, ``score``, ``zero_action_floor``).

DexMachina specifics this module handles (see docs/implementation-notes.md):
- ``DexMachinaVecEnv.step`` resets finished envs inside the step and returns the post-reset obs; ``final_obs`` holds
  the obs before the reset.
- Every episode end of the benchmark task (demo end, drop, early reset, NaN) is terminated; it has no truncations.
- The curriculum only advances if the learner calls ``set_curriculum(epoch)``.
"""
import json
from dataclasses import fields
from pathlib import Path
from types import MappingProxyType

import torch

from dexhand_ppo.vec_env import EpisodeTracker

EPISODE_TERMS = ("task_rew", "imi_rew", "bc_rew", "con_rew")   # per-episode means of these rew_dict entries
STEP_METRICS = ("pos_dist", "rot_dist", "arti_dist", "success_2cm", "success_5cm")  # assisted, per step
CURRICULUM_TERMS = ("kp", "kv", "fr")
# EvalScore fields that are mappings: the per-part AUCs are logged one key per part; the caller-chosen part
# aggregations are not requested here, so that field stays empty.
EVAL_SCORE_MAPPINGS = ("auc_add_part", "auc_add_aggregated")
# The benchmark task sets every reward weight through its EnvSpec.
NO_REWARD_OVERRIDES = MappingProxyType({})
TRAJECTORY_DIRNAME = "eval_trajectories"            # in the run dir
TRAJECTORY_FILENAME = "trajectory_step_{step}.npz"  # env steps at the evaluation; the last one is final.pt's

_GENESIS_READY = False


def init_genesis(device: torch.device, seed: int) -> None:
    """gs.init is process-global and may only run once (training and eval env share the process)."""
    global _GENESIS_READY
    import genesis as gs
    if _GENESIS_READY or getattr(gs, "_initialized", False):  # also initialised elsewhere (e.g. a test fixture)
        _GENESIS_READY = True
        return
    gs.init(backend=gs.cpu if device.type == "cpu" else gs.gpu, logging_level="warning", seed=seed)
    _GENESIS_READY = True


def dexmachina_device(device: torch.device) -> str:
    return "cpu" if device.type == "cpu" else "cuda"


def run_spec(cfg):
    """The benchmark task's EnvSpec for this run."""
    from dexmachina.envs.benchmark_task import RunChoices, benchmark_spec

    d = cfg.env.dexmachina
    return benchmark_spec(RunChoices(hand=d.hand, clip=d.clip, seed=cfg.seed, num_envs=cfg.env.num_envs))


class DexMachinaTrainEnv:
    """Training env (assisted physics, curriculum driven by the normalised progress counter)."""

    def __init__(self, cfg, vec_env=None):
        """``vec_env``: an already built DexMachinaVecEnv-like object with ``spec`` and the BaseEnv as ``env``
        (tests); otherwise the benchmark env is built from cfg."""
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        if vec_env is None:
            from dexmachina.envs.vec_env import DexMachinaVecEnv

            init_genesis(self.device, cfg.seed)
            vec_env = DexMachinaVecEnv.from_spec(run_spec(cfg), dexmachina_device(self.device), NO_REWARD_OVERRIDES)
        self.vec_env = vec_env
        self.spec = vec_env.spec
        self.env = vec_env.env
        self.num_envs = self.env.num_envs
        self.obs_dim = int(self.env.num_obs)
        self.action_dim = int(self.env.action_dim)
        self.max_episode_length = int(self.env.max_episode_length)
        self.tracker = EpisodeTracker(self.num_envs, self.device, extra_keys=EPISODE_TERMS)

        steps_per_iteration = cfg.env.num_envs * cfg.ppo.num_steps
        self.total_env_steps = (cfg.total_steps // steps_per_iteration) * steps_per_iteration
        # Emax of the normalised progress counter; the env derives its zero_epoch from the same spec value.
        self.max_epochs = int(self.spec.max_epochs)
        self.current_epoch = -1
        self.num_reductions = 0
        self.num_dialbacks = 0
        self.hard_stop_epoch = None
        self._step_sums = {}
        self._num_steps = 0
        self._nan_obs = 0
        self._nan_rew = 0

    # ---- VecEnv protocol ----
    def reset(self) -> torch.Tensor:
        obs, _ = self.vec_env.reset()
        return obs

    def step(self, actions: torch.Tensor):
        obs, rew, terminated, truncated, vec_info = self.vec_env.step(actions)  # BaseEnv clamps to its action_clip
        final_obs = vec_info["final_obs"]
        nan_obs, nan_rew = ~torch.isfinite(obs), ~torch.isfinite(rew)
        if nan_obs.any() or nan_rew.any():
            self._nan_obs += int(nan_obs.any(dim=-1).sum())
            self._nan_rew += int(nan_rew.sum())
            obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
        # only read for truncation bootstrapping; sanitised like obs so that a bootstrap never reads NaN
        final_obs = torch.nan_to_num(final_obs, nan=0.0, posinf=0.0, neginf=0.0)
        done = terminated | truncated
        rew_dict = self.env.rew_dict
        terms = {k: rew_dict[k].float() for k in EPISODE_TERMS if k in rew_dict}
        for key in STEP_METRICS:
            if key in rew_dict:
                self._step_sums[key] = self._step_sums.get(key, 0.0) + rew_dict[key].float().mean()
        self._num_steps += 1
        info = {"episode": self.tracker.step(rew, done, terms), "final_obs": final_obs, "final_obs_mask": done}
        return obs, rew, terminated, truncated, info

    def on_iteration(self, global_step: int) -> dict:
        """Advance the curriculum on the normalised progress counter and return env metrics of the last rollout."""
        # sums are accumulated on the device; one sync per iteration instead of several per step
        metrics = ({f"env/{k}": float(v) / self._num_steps for k, v in self._step_sums.items()}
                   if self._num_steps else {})
        metrics.update({"train/nan_obs_envs": self._nan_obs, "train/nan_rewards": self._nan_rew})
        self._step_sums, self._num_steps = {}, 0

        epoch = min(global_step * self.max_epochs // self.total_env_steps, self.max_epochs)
        curriculum = self._curriculum()
        if curriculum is not None and epoch > self.current_epoch:
            before = dict(curriculum.curr_gains)
            self.env.set_curriculum(epoch)
            after = curriculum.curr_gains
            changed = [k for k in curriculum.decay_terms if after[k] != before[k]]
            if changed and epoch > curriculum.zero_epoch and all(after[k] == 0.0 for k in curriculum.decay_terms):
                self.hard_stop_epoch = epoch           # remaining assistance removed in one step
            elif changed and all(after[k] <= before[k] for k in changed):
                self.num_reductions += 1
            elif changed:
                self.num_dialbacks += 1
            self.current_epoch = epoch
        if curriculum is not None:
            metrics.update({f"env/{k}": float(curriculum.curr_gains[k]) for k in CURRICULUM_TERMS
                            if k in curriculum.curr_gains})
            metrics.update({"curriculum/epoch": self.current_epoch, "curriculum/num_reductions": self.num_reductions,
                            "curriculum/num_dialbacks": self.num_dialbacks,
                            "curriculum/hard_stop_applied": float(self.hard_stop_epoch is not None)})
        return metrics

    def close(self) -> None:
        pass  # Genesis has no per-scene teardown; the process exit cleans up

    def _curriculum(self):
        return self.env.curriculum if getattr(self.env, "use_curriculum", False) else None

    # ---- provenance and checkpoints ----
    def describe(self) -> dict:
        from dexmachina.envs.factory import task_hash

        return {"kind": "dexmachina", "spec": json.loads(self.spec.to_json()), "task_hash": task_hash(self.spec),
                "obs_dim": self.obs_dim, "action_dim": self.action_dim,
                "max_episode_length": self.max_episode_length, "curriculum_total_env_steps": self.total_env_steps,
                "curriculum_zero_epoch": self.env.curriculum.zero_epoch}

    def save_run_spec(self, directory: Path) -> Path:
        """``env_spec.json`` next to the checkpoints, from which evaluation rebuilds this run's task."""
        from dexmachina.envs.run_spec import save_run_spec

        return save_run_spec(directory, self.spec, NO_REWARD_OVERRIDES)

    def state_dict(self) -> dict:
        curriculum = self._curriculum()
        state = {"current_epoch": self.current_epoch, "num_reductions": self.num_reductions,
                 "num_dialbacks": self.num_dialbacks, "hard_stop_epoch": self.hard_stop_epoch}
        if curriculum is not None:
            state.update(curr_gains=dict(curriculum.curr_gains), curr_gains_lower=dict(curriculum.curr_gains_lower),
                         gain_history=list(curriculum.gain_history),
                         last_decay_epoch=getattr(curriculum, "last_decay_epoch", 0))
        return state

    def load_state_dict(self, state: dict) -> None:
        """Restores counters and gains. The curriculum's reward/length deques start empty again (documented)."""
        for key in ("current_epoch", "num_reductions", "num_dialbacks", "hard_stop_epoch"):
            setattr(self, key, state[key])
        curriculum = self._curriculum()
        if curriculum is not None and "curr_gains" in state:
            curriculum.curr_gains = dict(state["curr_gains"])
            curriculum.curr_gains_lower = dict(state["curr_gains_lower"])
            curriculum.gain_history = list(state["gain_history"])
            curriculum.last_decay_epoch = state["last_decay_epoch"]
            curriculum.reset_object_gains()


def score_metrics(prefix: str, scored) -> dict:
    """Every scalar of a dexmachina ``EvalScore`` as ``<prefix><name>``, the AUC of each object part as
    ``<prefix>auc_add_part/<part>`` (FlashSAC's names for the same scores)."""
    metrics = {f"{prefix}{f.name}": float(getattr(scored, f.name)) for f in fields(scored)
               if f.name not in EVAL_SCORE_MAPPINGS}
    metrics.update({f"{prefix}auc_add_part/{part}": auc for part, auc in scored.auc_add_part.items()})
    return metrics


class DexMachinaEvaluator:
    """The shared evaluation: per call, the zero-action floor and one deterministic, unassisted, full-length policy
    episode on the same single-env eval env. The policy episode's trajectory is saved, so its scores can be
    recomputed from the file."""

    selection_metric = "eval/auc_add"

    def __init__(self, spec, device: torch.device, trajectory_dir: Path):
        from dexmachina.envs.factory import make_eval_env, task_hash
        from dexmachina.envs.run_spec import current_dexmachina_state

        self.spec = spec
        self.trajectory_dir = trajectory_dir
        self.task_hash = task_hash(spec)
        self.dexmachina_sha = current_dexmachina_state().sha
        self.env = make_eval_env(spec, dexmachina_device(device))

    @torch.no_grad()
    def evaluate(self, policy, global_step: int) -> dict:
        from dexmachina.envs.evaluation import rollout_episode, save_trajectory, score, zero_action_floor

        floor = zero_action_floor(self.env, self.spec.hand, self.spec.clip)
        trajectory = rollout_episode(self.env, policy, self.spec.hand, self.spec.clip, self.task_hash,
                                     self.dexmachina_sha)
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        save_trajectory(self.trajectory_dir / TRAJECTORY_FILENAME.format(step=global_step), trajectory)
        return {**score_metrics("eval/", score(trajectory)), **score_metrics("eval/floor/", floor)}

    def close(self) -> None:
        pass


def make_dexmachina_envs(cfg):
    train_env = DexMachinaTrainEnv(cfg)
    evaluator = (DexMachinaEvaluator(train_env.spec, train_env.device, Path(cfg.run_dir) / TRAJECTORY_DIRNAME)
                 if cfg.eval.enabled else None)
    return train_env, evaluator
