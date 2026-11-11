"""DexMachina adapter: benchmark env construction, done handling, curriculum, metrics and unassisted evaluation.

DexMachina specifics this module handles (see docs/implementation-notes.md):
- ``BaseEnv.step`` resets finished envs internally and returns the post-reset obs; there is no terminal obs.
- With ``use_rl_games=True`` it returns (obs_dict, rew, terminated, time_outs, extras), where *both* flags equal the
  reset mask for every reset (time-out, early reset, NaN). All episode ends are therefore terminal here.
- ``rew_buf`` and the flags are persistent buffers updated in place: they are cloned every step.
- The curriculum only advances if the learner calls ``set_curriculum(epoch)``.
- With ``is_eval`` and more than one env, the last env is a dummy (not stepped, zero init qpos); it is excluded
  from evaluation metrics.
"""
import numpy as np
import torch

from dexhand_ppo.vec_env import EpisodeTracker

EPISODE_TERMS = ("task_rew", "imi_rew", "bc_rew", "con_rew")   # per-episode means of these rew_dict entries
STEP_METRICS = ("pos_dist", "rot_dist", "arti_dist", "success_2cm", "success_5cm")  # assisted, per step
ADD_PARTS = ("add_top", "add_bottom")
SUCCESS_THRESHOLDS = {"2cm": 0.02, "5cm": 0.05}
CURRICULUM_TERMS = ("kp", "kv", "fr")

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


def build_argv(cfg, num_envs: int, eval_mode: bool) -> list[str]:
    d = cfg.env.dexmachina
    rand_init = 0.0 if eval_mode else d.rand_init_ratio
    return [*map(str, d.args), "-B", str(num_envs), "--hand", d.hand, "--clip", d.clip, "--seed", str(cfg.seed),
            "--max_epochs", str(d.curriculum_max_epochs), "--rand_init_ratio", str(rand_init)]


def build_env_kwargs(cfg, num_envs: int, eval_mode: bool) -> tuple[list[str], dict]:
    from dexmachina.envs.constructors import get_all_env_cfg, get_common_argparser

    argv = build_argv(cfg, num_envs, eval_mode)
    args, unknown = get_common_argparser().parse_known_args(argv)
    if unknown:  # a typo in env.dexmachina.args must not be silently ignored
        raise ValueError(f"unknown dexmachina arguments: {unknown}")
    device = "cpu" if torch.device(cfg.device).type == "cpu" else "cuda"
    kwargs = get_all_env_cfg(args, device=device)
    env_cfg = kwargs["env_cfg"]
    env_cfg["use_rl_games"] = True
    env_cfg["record_video"] = False
    if eval_mode:  # permanently unassisted, full-length, deterministic episodes (REPPO's eval_mode recipe)
        env_cfg["is_eval"] = True
        env_cfg["early_reset_threshold"] = 0.0
        env_cfg["early_reset_aux_thres"] = dict(con=0.0, imi=0.0, bc=0.0)
        env_cfg["rand_init_ratio"] = 0.0
        env_cfg["log_add_metric"] = True
        env_cfg["add_num_verts"] = 500
        rand_cfg = kwargs.setdefault("rand_cfg", {})
        if isinstance(rand_cfg, dict):
            rand_cfg["randomize"] = False
    else:
        env_cfg["log_add_metric"] = False
    return argv, kwargs


def _jsonable(value, depth: int = 0):
    """Small, JSON-serialisable view of a config (drops tensors/arrays and deep nesting) for provenance."""
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, (list, tuple)) and len(value) <= 64 and all(
            isinstance(v, (bool, int, float, str)) or v is None for v in value):
        return list(value)
    if isinstance(value, dict) and depth < 3:
        out = {str(k): _jsonable(v, depth + 1) for k, v in value.items()}
        return {k: v for k, v in out.items() if v is not None or k in value and value[k] is None}
    return None


def _zero_gains(env) -> None:
    if getattr(env, "use_curriculum", False) and env.curriculum is not None:
        curriculum = env.curriculum
        for key in curriculum.decay_terms:
            curriculum.curr_gains[key] = 0.0
            curriculum.curr_gains_lower[key] = 0.0
        curriculum.reset_object_gains()
        env.record_curriculum = False  # eval episodes must never enter the curriculum statistics


class DexMachinaVecEnv:
    """Training env (assisted physics, curriculum driven by the normalised progress counter)."""

    def __init__(self, cfg, env=None):
        """``env``: an already built BaseEnv-like object (tests); otherwise the benchmark env is built from cfg."""
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        if env is None:
            from dexmachina.envs.base_env import BaseEnv

            init_genesis(self.device, cfg.seed)
            self.argv, kwargs = build_env_kwargs(cfg, cfg.env.num_envs, eval_mode=False)
            self.env_kwargs_summary = {k: _jsonable(kwargs.get(k)) for k in
                                       ("env_cfg", "reward_cfg", "curriculum_cfg", "rand_cfg", "object_cfgs")}
            env = BaseEnv(**kwargs)
        else:
            self.argv, self.env_kwargs_summary = [], {}
        self.env = env
        self.num_envs = self.env.num_envs
        self.obs_dim = int(self.env.num_obs)
        self.action_dim = int(self.env.action_dim)
        self.max_episode_length = int(self.env.max_episode_length)
        self.tracker = EpisodeTracker(self.num_envs, self.device, extra_keys=EPISODE_TERMS)

        steps_per_iteration = cfg.env.num_envs * cfg.ppo.num_steps
        self.total_env_steps = (cfg.total_steps // steps_per_iteration) * steps_per_iteration
        self.max_epochs = int(cfg.env.dexmachina.curriculum_max_epochs)
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
        obs_dict, _ = self.env.reset()
        return obs_dict["policy"].clone()

    def step(self, actions: torch.Tensor):
        obs_dict, rew, terminated, _time_outs, _extras = self.env.step(actions.clamp(-1.0, 1.0))
        obs = obs_dict["policy"].clone()
        rew = rew.clone()
        done = terminated.clone().bool()
        nan_obs, nan_rew = ~torch.isfinite(obs), ~torch.isfinite(rew)
        if nan_obs.any() or nan_rew.any():
            self._nan_obs += int(nan_obs.any(dim=-1).sum())
            self._nan_rew += int(nan_rew.sum())
            obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
        rew_dict = self.env.rew_dict
        terms = {k: rew_dict[k].float() for k in EPISODE_TERMS if k in rew_dict}
        for key in STEP_METRICS:
            if key in rew_dict:
                self._step_sums[key] = self._step_sums.get(key, 0.0) + rew_dict[key].float().mean()
        self._num_steps += 1
        info = {"episode": self.tracker.step(rew, done, terms)}
        # All ends are terminal: dexmachina cannot tell time-outs from failures, and the obs contains t/T.
        truncated = torch.zeros_like(done)
        return obs, rew, done, truncated, info

    def on_iteration(self, global_step: int) -> dict:
        """Advance the curriculum on the normalised progress counter and return env metrics of the last rollout."""
        # sums are accumulated on the device; one sync per iteration instead of several per step
        metrics = ({f"env/{k}": float(v) / self._num_steps for k, v in self._step_sums.items()}
                   if self._num_steps else {})
        metrics.update({"train/nan_obs_envs": self._nan_obs, "train/nan_rewards": self._nan_rew})
        self._step_sums, self._num_steps = {}, 0

        epoch = min(global_step * self.max_epochs // self.total_env_steps, self.max_epochs)
        curriculum = self.env.curriculum if getattr(self.env, "use_curriculum", False) else None
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

    # ---- provenance and checkpoints ----
    def describe(self) -> dict:
        return {"kind": "dexmachina", "argv": self.argv, "obs_dim": self.obs_dim, "action_dim": self.action_dim,
                "max_episode_length": self.max_episode_length, "curriculum_total_env_steps": self.total_env_steps,
                "env_kwargs": self.env_kwargs_summary}

    def state_dict(self) -> dict:
        curriculum = self.env.curriculum if getattr(self.env, "use_curriculum", False) else None
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
        curriculum = self.env.curriculum if getattr(self.env, "use_curriculum", False) else None
        if curriculum is not None and "curr_gains" in state:
            curriculum.curr_gains = dict(state["curr_gains"])
            curriculum.curr_gains_lower = dict(state["curr_gains_lower"])
            curriculum.gain_history = list(state["gain_history"])
            curriculum.last_decay_epoch = state["last_decay_epoch"]
            curriculum.reset_object_gains()


class DexMachinaEvaluator:
    """Separate, permanently unassisted env: one full-length deterministic episode per real env."""

    selection_metric = "eval/auc_add"

    def __init__(self, cfg, env=None):
        """``env``: an already built eval env whose last env is the dummy (tests); otherwise built from cfg."""
        if env is None:
            from dexmachina.envs.base_env import BaseEnv

            init_genesis(torch.device(cfg.device), cfg.seed)
            num_real = int(cfg.env.dexmachina.num_eval_envs)
            self.argv, kwargs = build_env_kwargs(cfg, num_real + 1, eval_mode=True)  # +1: dexmachina's dummy env
            env = BaseEnv(**kwargs)
        else:
            self.argv = []
        self.env = env
        _zero_gains(self.env)
        self.num_real = self.env.num_envs - 1
        self.max_episode_length = int(self.env.max_episode_length)

    @torch.no_grad()
    def evaluate(self, policy) -> dict:
        from dexmachina.envs.add_metric import compute_auc

        obs_dict, _ = self.env.reset()
        obs = obs_dict["policy"]
        per_step = {k: [] for k in (*ADD_PARTS, "add", "success_2cm", "success_5cm", "pos_dist")}
        for _ in range(self.max_episode_length):
            obs_dict, _rew, _term, _trunc, extras = self.env.step(policy(obs).clamp(-1.0, 1.0))
            obs = obs_dict["policy"]
            log = extras["log"]
            for key in (*ADD_PARTS, "add"):
                per_step[key].append(log[key][: self.num_real].float().cpu().numpy())
            for key in ("success_2cm", "success_5cm", "pos_dist"):
                per_step[key].append(self.env.rew_dict[key][: self.num_real].float().cpu().numpy())
        arrays = {k: np.stack(v) for k, v in per_step.items()}   # (T, num_real)
        parts = [arrays[p] for p in ADD_PARTS]
        metrics = {
            # compute_add.py's definition: AUC per object part, then averaged
            "eval/auc_add": float(np.mean([compute_auc(p.ravel()) for p in parts])),
            # REPPO's online definition (AUC of the part-averaged ADD), for comparison
            "eval/auc_add_part_mean": compute_auc(arrays["add"].ravel()),
            "eval/add": float(arrays["add"].mean()),
            "eval/success_2cm": float(arrays["success_2cm"].mean()),   # pos/rot/arti thresholds (FastTD3-style)
            "eval/success_5cm": float(arrays["success_5cm"].mean()),
            "eval/pos_dist": float(arrays["pos_dist"].mean()),
            "eval/num_envs": self.num_real,
        }
        for name, threshold in SUCCESS_THRESHOLDS.items():  # ADD-based, per part then averaged (like the AUC)
            metrics[f"eval/add_success_{name}"] = float(np.mean([(p < threshold).mean() for p in parts]))
        return metrics

    def close(self) -> None:
        pass


def make_dexmachina_envs(cfg):
    train_env = DexMachinaVecEnv(cfg)
    evaluator = DexMachinaEvaluator(cfg) if cfg.eval.enabled else None
    return train_env, evaluator
