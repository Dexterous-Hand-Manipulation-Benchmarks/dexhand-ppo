"""Rollout storage, generalised advantage estimation and minibatching."""
import torch


def compute_gae(rewards, values, dones, last_values, gamma: float, lam: float):
    """GAE over a (T, N) rollout.

    ``dones[t]`` is 1 if the episode ended after the action at step t; the value of the following state is then
    not bootstrapped and the advantage trace is cut. ``last_values`` is V(s_T) for the state after the rollout.
    Truncations must already be folded into ``rewards`` (gamma * V(final_obs)) and marked in ``dones``.
    Returns (advantages, returns), both (T, N).
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(last_values)
    for t in reversed(range(num_steps)):
        next_values = last_values if t == num_steps - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values * not_done - values[t]
        last_gae = delta + gamma * lam * not_done * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


def minibatch_indices(batch_size: int, num_minibatches: int, generator: torch.Generator, device):
    """Random partition of range(batch_size) into num_minibatches equal parts (each index exactly once)."""
    if batch_size % num_minibatches != 0:
        raise ValueError(f"batch size {batch_size} is not divisible by num_minibatches {num_minibatches}")
    perm = torch.randperm(batch_size, generator=generator, device="cpu").to(device)
    return perm.view(num_minibatches, -1)


class RolloutBuffer:
    """(T, N) rollout of normalised observations, actions, log-probs, raw values, rewards and dones."""

    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, action_dim: int, device):
        shape = (num_steps, num_envs)
        self.num_steps, self.num_envs = num_steps, num_envs
        self.obs = torch.zeros(*shape, obs_dim, device=device)
        self.actions = torch.zeros(*shape, action_dim, device=device)
        self.log_probs = torch.zeros(shape, device=device)
        self.mu = torch.zeros(*shape, action_dim, device=device)
        self.sigma = torch.zeros(*shape, action_dim, device=device)
        self.values = torch.zeros(shape, device=device)       # raw (denormalised) values
        self.rewards = torch.zeros(shape, device=device)      # scaled rewards incl. truncation bootstrap
        self.dones = torch.zeros(shape, device=device)
        self.advantages = torch.zeros(shape, device=device)
        self.returns = torch.zeros(shape, device=device)
        self.step = 0

    def add(self, obs, actions, log_probs, mu, sigma, values, rewards, dones) -> None:
        t = self.step
        self.obs[t], self.actions[t], self.log_probs[t] = obs, actions, log_probs
        self.mu[t], self.sigma[t], self.values[t] = mu, sigma, values
        self.rewards[t], self.dones[t] = rewards, dones
        self.step += 1

    def finish(self, last_values, gamma: float, lam: float) -> None:
        assert self.step == self.num_steps, "rollout incomplete"
        self.advantages, self.returns = compute_gae(self.rewards, self.values, self.dones, last_values, gamma, lam)
        self.step = 0

    def flat(self) -> dict:
        n = self.num_steps * self.num_envs
        return {
            "obs": self.obs.reshape(n, -1), "actions": self.actions.reshape(n, -1),
            "log_probs": self.log_probs.reshape(n), "mu": self.mu.reshape(n, -1), "sigma": self.sigma.reshape(n, -1),
            "values": self.values.reshape(n), "advantages": self.advantages.reshape(n),
            "returns": self.returns.reshape(n),
        }
