"""PPO update: clipped surrogate, value loss, entropy and bounds terms, KL-controlled learning rate."""
import torch
from torch import nn

from dexhand_ppo.storage import minibatch_indices


def gaussian_kl(mu_old, sigma_old, mu_new, sigma_new) -> torch.Tensor:
    """KL(old || new) of diagonal Gaussians, summed over action dims, per sample (as rl_games' policy_kl)."""
    return (torch.log(sigma_new / sigma_old)
            + (sigma_old.pow(2) + (mu_old - mu_new).pow(2)) / (2.0 * sigma_new.pow(2)) - 0.5).sum(-1)


def bounds_loss(mu: torch.Tensor, soft_bound: float) -> torch.Tensor:
    """rl_games bounds loss: squared excess of the action mean beyond +-soft_bound."""
    high = (mu - soft_bound).clamp_min(0.0).pow(2)
    low = (mu + soft_bound).clamp_max(0.0).pow(2)
    return (high + low).sum(-1).mean()


def adapt_learning_rate(lr: float, kl: float, cfg) -> float:
    """rl_games 'adaptive' schedule: steer the measured KL towards kl_target."""
    if kl > 2.0 * cfg.kl_target:
        lr = lr / 1.5
    elif kl < 0.5 * cfg.kl_target:
        lr = lr * 1.5
    return min(max(lr, cfg.lr_min), cfg.lr_max)


class PPO:
    def __init__(self, model: nn.Module, value_normalizer, cfg, total_iterations: int, device):
        self.model, self.value_norm, self.cfg = model, value_normalizer, cfg
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        self.learning_rate = cfg.learning_rate
        self.total_iterations = total_iterations
        self.generator = torch.Generator()
        self.num_updates = 0  # gradient steps so far

    def set_lr(self, lr: float) -> None:
        self.learning_rate = lr
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def update(self, buffer, iteration: int) -> dict:
        cfg = self.cfg
        if cfg.lr_schedule == "linear":
            self.set_lr(cfg.learning_rate * max(1.0 - iteration / self.total_iterations, 0.0))
        data = buffer.flat()
        # Value targets in normalised units: update the normaliser with this batch's returns first (rl_games).
        self.value_norm.update(data["returns"])
        returns_n = self.value_norm.normalize(data["returns"])
        old_values_n = self.value_norm.normalize(data["values"])

        batch_size = data["obs"].shape[0]
        stats = {k: 0.0 for k in ("policy_loss", "value_loss", "entropy", "bounds_loss", "kl", "clip_fraction",
                                  "grad_norm")}
        num_minibatch_updates, stopped_early, epochs_done = 0, False, 0
        for _ in range(cfg.num_epochs):
            epoch_kls = []
            for idx in minibatch_indices(batch_size, cfg.num_minibatches, self.generator, self.device):
                mu, values_n = self.model(data["obs"][idx])
                dist = self.model.distribution(mu)
                log_probs = dist.log_prob(data["actions"][idx]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                adv = data["advantages"][idx]
                if cfg.normalize_advantage:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                ratio = torch.exp(log_probs - data["log_probs"][idx])
                surrogate = torch.min(ratio * adv, ratio.clamp(1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv)
                policy_loss = -surrogate.mean()

                if cfg.value_clip:
                    v_clipped = old_values_n[idx] + (values_n - old_values_n[idx]).clamp(-cfg.clip_ratio,
                                                                                         cfg.clip_ratio)
                    value_err = torch.max((values_n - returns_n[idx]).pow(2), (v_clipped - returns_n[idx]).pow(2))
                else:
                    value_err = (values_n - returns_n[idx]).pow(2)
                value_loss = 0.5 * value_err.mean()
                b_loss = bounds_loss(mu, cfg.bounds_soft)

                loss = (policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                        + cfg.bounds_coef * b_loss)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                self.num_updates += 1

                with torch.no_grad():
                    kl = gaussian_kl(data["mu"][idx], data["sigma"][idx], mu, dist.stddev).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_ratio).float().mean().item()
                epoch_kls.append(kl)
                for key, val in (("policy_loss", policy_loss.item()), ("value_loss", value_loss.item()),
                                 ("entropy", entropy.item()), ("bounds_loss", b_loss.item()), ("kl", kl),
                                 ("clip_fraction", clip_frac), ("grad_norm", float(grad_norm))):
                    stats[key] += val
                num_minibatch_updates += 1
                if cfg.kl_stop > 0 and kl > cfg.kl_stop * cfg.kl_target:
                    stopped_early = True
                    break
            epochs_done += 1
            if cfg.lr_schedule == "adaptive_kl":
                self.set_lr(adapt_learning_rate(self.learning_rate, sum(epoch_kls) / len(epoch_kls), cfg))
            if stopped_early:
                break

        stats = {k: v / max(num_minibatch_updates, 1) for k, v in stats.items()}
        with torch.no_grad():
            var_ret = data["returns"].var()
            explained_var = 1.0 - (data["returns"] - data["values"]).var() / (var_ret + 1e-8)
        stats.update(learning_rate=self.learning_rate, explained_variance=explained_var.item(),
                     action_std=self.model.std().mean().item(), minibatch_updates=num_minibatch_updates,
                     epochs_done=epochs_done, stopped_early=float(stopped_early), num_updates=self.num_updates)
        return stats
