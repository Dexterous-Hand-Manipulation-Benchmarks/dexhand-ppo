"""Actor-critic networks: Gaussian policy with a state-independent log std, and a scalar value head."""
import math

import torch
from torch import nn
from torch.distributions import Normal

ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU}


def mlp(in_dim: int, hidden: list[int], activation: str, layer_norm: bool) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    dim = in_dim
    for width in hidden:
        linear = nn.Linear(dim, width)
        nn.init.orthogonal_(linear.weight, gain=math.sqrt(2))
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if layer_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(ACTIVATIONS[activation]())
        dim = width
    return nn.Sequential(*layers), dim


def head(in_dim: int, out_dim: int, gain: float) -> nn.Linear:
    linear = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(linear.weight, gain=gain)
    nn.init.zeros_(linear.bias)
    return linear


class ActorCritic(nn.Module):
    """Separate actor and critic MLPs, or one shared trunk with two heads (rl_games baseline)."""

    def __init__(self, obs_dim: int, action_dim: int, cfg):
        super().__init__()
        self.shared = cfg.shared
        if self.shared:
            self.trunk, dim = mlp(obs_dim, list(cfg.shared_hidden), cfg.activation, cfg.layer_norm)
            self.mu_head = head(dim, action_dim, gain=0.01)
            self.value_head = head(dim, 1, gain=1.0)
        else:
            self.actor_body, actor_dim = mlp(obs_dim, list(cfg.actor_hidden), cfg.activation, cfg.layer_norm)
            self.critic_body, critic_dim = mlp(obs_dim, list(cfg.critic_hidden), cfg.activation, cfg.layer_norm)
            self.mu_head = head(actor_dim, action_dim, gain=0.01)
            self.value_head = head(critic_dim, 1, gain=1.0)
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(cfg.init_std)))
        self.min_std = float(cfg.min_std)

    def std(self) -> torch.Tensor:
        std = self.log_std.exp()
        return std.clamp_min(self.min_std) if self.min_std > 0 else std

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Action mean and (normalised) value."""
        if self.shared:
            features = self.trunk(obs)
            return self.mu_head(features), self.value_head(features).squeeze(-1)
        return self.mu_head(self.actor_body(obs)), self.value_head(self.critic_body(obs)).squeeze(-1)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        if self.shared:
            return self.value_head(self.trunk(obs)).squeeze(-1)
        return self.value_head(self.critic_body(obs)).squeeze(-1)

    def act_mean(self, obs: torch.Tensor) -> torch.Tensor:
        if self.shared:
            return self.mu_head(self.trunk(obs))
        return self.mu_head(self.actor_body(obs))

    def distribution(self, mu: torch.Tensor) -> Normal:
        return Normal(mu, self.std().expand_as(mu))
