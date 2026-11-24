"""gymnasium vector env adapter for CPU sanity tasks (e.g. Pendulum)."""
import gymnasium as gym
import numpy as np
import torch

from dexhand_ppo.vec_env import EpisodeTracker


def _make_vec(env_id: str, num_envs: int, seed: int):
    envs = gym.make_vec(env_id, num_envs=num_envs, vectorization_mode="sync",
                        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})
    envs.reset(seed=seed)
    return envs


class GymVecEnv:
    """Maps gymnasium's same-step autoreset onto the VecEnv protocol. Actions in [-1, 1] are rescaled to the
    env's action bounds."""

    def __init__(self, env_id: str, num_envs: int, seed: int, device):
        self.envs = _make_vec(env_id, num_envs, seed)
        self.num_envs = num_envs
        self.device = torch.device(device)
        space = self.envs.single_action_space
        self.low = torch.as_tensor(space.low, dtype=torch.float32, device=self.device)
        self.high = torch.as_tensor(space.high, dtype=torch.float32, device=self.device)
        self.obs_dim = int(np.prod(self.envs.single_observation_space.shape))
        self.action_dim = int(np.prod(space.shape))
        self.tracker = EpisodeTracker(num_envs, self.device)
        self._seed = seed

    def _t(self, x) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)

    def reset(self) -> torch.Tensor:
        obs, _ = self.envs.reset(seed=self._seed)
        return self._t(obs)

    def step(self, actions: torch.Tensor):
        scaled = self.low + (actions.clamp(-1.0, 1.0) + 1.0) * 0.5 * (self.high - self.low)
        obs, rew, term, trunc, info = self.envs.step(scaled.cpu().numpy())
        rew_t, term_t, trunc_t = self._t(rew), self._t(term).bool(), self._t(trunc).bool()
        out = {"episode": self.tracker.step(rew_t, term_t | trunc_t)}
        if "final_obs" in info:
            mask = np.asarray(info["_final_obs"])
            final = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
            for i in np.nonzero(mask)[0]:
                final[i] = info["final_obs"][i]
            out["final_obs"], out["final_obs_mask"] = self._t(final), self._t(mask).bool()
        return self._t(obs), rew_t, term_t, trunc_t, out

    def on_iteration(self, global_step: int) -> dict:
        return {}

    def close(self) -> None:
        self.envs.close()


class GymEvaluator:
    """Deterministic (mean-action) episodes on a separate env instance."""

    selection_metric = "eval/return"

    def __init__(self, env_id: str, num_episodes: int, seed: int, device):
        self.env = GymVecEnv(env_id, num_episodes, seed + 10_000, device)
        self.num_episodes = num_episodes

    @torch.no_grad()
    def evaluate(self, policy, global_step: int) -> dict:
        """``global_step`` names a saved evaluation in the DexMachina evaluator; this one saves nothing."""
        obs = self.env.reset()
        returns = torch.zeros(self.num_episodes, device=self.env.device)
        finished = torch.zeros(self.num_episodes, dtype=torch.bool, device=self.env.device)
        while not finished.all():
            obs, rew, term, trunc, _ = self.env.step(policy(obs))
            returns += rew * (~finished)
            finished |= term | trunc
        return {"eval/return": returns.mean().item(), "eval/return_std": returns.std().item()}

    def close(self) -> None:
        self.env.close()


def make_gym_envs(cfg):
    train = GymVecEnv(cfg.env.gym.id, cfg.env.num_envs, cfg.seed, cfg.device)
    evaluator = (GymEvaluator(cfg.env.gym.id, cfg.env.gym.num_eval_episodes, cfg.seed, cfg.device)
                 if cfg.eval.enabled else None)
    return train, evaluator
