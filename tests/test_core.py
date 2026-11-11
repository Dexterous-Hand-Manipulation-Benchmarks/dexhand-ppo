"""Unit tests of the env-independent core: GAE, minibatching, normalisers, loss helpers, networks, checkpoints."""
import math

import pytest
import torch
from omegaconf import OmegaConf
from torch.distributions import Normal, kl_divergence

from dexhand_ppo.checkpoint import load_checkpoint, save_checkpoint
from dexhand_ppo.config import NetConfig, PPOConfig
from dexhand_ppo.networks import ActorCritic
from dexhand_ppo.normalizers import Identity, RunningMeanStd
from dexhand_ppo.ppo import PPO, adapt_learning_rate, bounds_loss, gaussian_kl
from dexhand_ppo.storage import RolloutBuffer, compute_gae, minibatch_indices
from dexhand_ppo.vec_env import EpisodeTracker

GAMMA, LAM = 0.9, 0.8


def test_gae_matches_hand_computation():
    # one env, 3 steps, episode ends after step 1
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.5], [0.4], [0.3]])
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    last = torch.tensor([0.2])
    adv, ret = compute_gae(rewards, values, dones, last, GAMMA, LAM)
    d2 = 3.0 + GAMMA * 0.2 - 0.3
    d1 = 2.0 - 0.4                     # terminal: no bootstrap
    d0 = 1.0 + GAMMA * 0.4 - 0.5
    expected = torch.tensor([[d0 + GAMMA * LAM * d1], [d1], [d2]])
    torch.testing.assert_close(adv, expected)
    torch.testing.assert_close(ret, expected + values)


def test_gae_envs_are_independent():
    torch.manual_seed(0)
    rewards, values = torch.randn(5, 3), torch.randn(5, 3)
    dones = (torch.rand(5, 3) < 0.3).float()
    last = torch.randn(3)
    adv, _ = compute_gae(rewards, values, dones, last, GAMMA, LAM)
    for i in range(3):
        adv_i, _ = compute_gae(rewards[:, i:i + 1], values[:, i:i + 1], dones[:, i:i + 1], last[i:i + 1], GAMMA, LAM)
        torch.testing.assert_close(adv[:, i:i + 1], adv_i)


def test_minibatches_partition_the_batch():
    gen = torch.Generator().manual_seed(0)
    idx = minibatch_indices(64, 8, gen, "cpu")
    assert idx.shape == (8, 8)
    assert sorted(idx.flatten().tolist()) == list(range(64))
    with pytest.raises(ValueError):
        minibatch_indices(65, 8, gen, "cpu")


def test_running_mean_std_matches_full_batch():
    torch.manual_seed(0)
    data = torch.randn(1000, 4) * torch.tensor([1.0, 2.0, 0.1, 5.0]) + 3.0
    rms = RunningMeanStd(4)
    for chunk in data.split(137):
        rms.update(chunk)
    torch.testing.assert_close(rms.mean.float(), data.mean(0), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(rms.var.float(), data.var(0, unbiased=False), atol=1e-4, rtol=1e-4)
    x = torch.randn(10, 4)
    torch.testing.assert_close(rms.denormalize(rms.normalize(x)), x, atol=1e-5, rtol=1e-5)
    assert rms.normalize(torch.full((1, 4), 1e6), clip=5.0).max() == 5.0
    restored = RunningMeanStd(4)
    restored.load_state_dict(rms.state_dict())
    torch.testing.assert_close(restored.normalize(x), rms.normalize(x))
    assert torch.equal(Identity().normalize(x), x)


def test_gaussian_kl_matches_torch():
    torch.manual_seed(0)
    mu0, mu1 = torch.randn(7, 3), torch.randn(7, 3)
    s0, s1 = torch.rand(7, 3) + 0.1, torch.rand(7, 3) + 0.1
    expected = kl_divergence(Normal(mu0, s0), Normal(mu1, s1)).sum(-1)
    torch.testing.assert_close(gaussian_kl(mu0, s0, mu1, s1), expected)
    assert gaussian_kl(mu0, s0, mu0, s0).abs().max() < 1e-6


def test_bounds_loss_only_beyond_soft_bound():
    assert bounds_loss(torch.tensor([[0.5, -1.0]]), 1.1) == 0.0
    assert math.isclose(bounds_loss(torch.tensor([[1.6, -1.3]]), 1.1).item(), 0.5**2 + 0.2**2, rel_tol=1e-5)


def test_adaptive_lr_rule():
    cfg = PPOConfig(kl_target=0.01, lr_min=1e-6, lr_max=1e-2)
    assert adapt_learning_rate(3e-4, 0.03, cfg) == pytest.approx(2e-4)
    assert adapt_learning_rate(3e-4, 0.001, cfg) == pytest.approx(4.5e-4)
    assert adapt_learning_rate(3e-4, 0.01, cfg) == 3e-4
    assert adapt_learning_rate(9e-3, 0.0, cfg) == 1e-2


@pytest.mark.parametrize("shared", [False, True])
def test_actor_critic_shapes_and_std(shared):
    net_cfg = OmegaConf.structured(NetConfig(shared=shared, init_std=1.0, min_std=0.5))
    model = ActorCritic(6, 3, net_cfg)
    mu, value = model(torch.randn(4, 6))
    assert mu.shape == (4, 3) and value.shape == (4,)
    with torch.no_grad():
        model.log_std.fill_(math.log(0.1))
    assert torch.allclose(model.std(), torch.full((3,), 0.5))  # lower bound
    assert torch.allclose(model.act_mean(torch.zeros(1, 6)), mu.new_zeros(1, 3), atol=1.0)


def _filled_buffer(num_steps=8, num_envs=4, obs_dim=6, action_dim=3, seed=0):
    torch.manual_seed(seed)
    buf = RolloutBuffer(num_steps, num_envs, obs_dim, action_dim, "cpu")
    for _ in range(num_steps):
        mu = torch.randn(num_envs, action_dim) * 0.1
        sigma = torch.ones(num_envs, action_dim)
        actions = mu + sigma * torch.randn_like(mu)
        log_probs = Normal(mu, sigma).log_prob(actions).sum(-1)
        buf.add(torch.randn(num_envs, obs_dim), actions, log_probs, mu, sigma, torch.zeros(num_envs),
                torch.randn(num_envs), (torch.rand(num_envs) < 0.1).float())
    buf.finish(torch.zeros(num_envs), 0.99, 0.95)
    return buf


def _ppo(seed=0, **overrides):
    torch.manual_seed(seed)
    model = ActorCritic(6, 3, OmegaConf.structured(NetConfig(actor_hidden=[16], critic_hidden=[16])))
    cfg = OmegaConf.structured(PPOConfig(num_minibatches=4, num_epochs=2, **overrides))
    algo = PPO(model, RunningMeanStd(()), cfg, total_iterations=10, device="cpu")
    algo.generator.manual_seed(seed)
    return algo


def test_ppo_update_counts_and_reproducibility():
    stats_a = _ppo().update(_filled_buffer(), iteration=0)
    stats_b = _ppo().update(_filled_buffer(), iteration=0)
    assert stats_a["minibatch_updates"] == 8 and stats_a["num_updates"] == 8
    assert stats_a == stats_b
    assert all(math.isfinite(v) for v in stats_a.values())


def test_kl_stop_ends_update_early():
    stats = _ppo(kl_stop=1e-9, lr_schedule="constant").update(_filled_buffer(), iteration=0)
    assert stats["stopped_early"] == 1.0 and stats["minibatch_updates"] < 8


def test_episode_tracker_emits_finished_episodes():
    tracker = EpisodeTracker(2, "cpu", extra_keys=("task",))
    assert tracker.step(torch.tensor([1.0, 2.0]), torch.tensor([False, False]), {"task": torch.tensor([1.0, 0.0])}) == {}
    ep = tracker.step(torch.tensor([1.0, 2.0]), torch.tensor([True, False]), {"task": torch.tensor([0.0, 0.0])})
    assert ep["return"].tolist() == [2.0] and ep["length"].tolist() == [2.0] and ep["task"].tolist() == [0.5]
    ep = tracker.step(torch.tensor([5.0, 1.0]), torch.tensor([True, True]), None)
    assert ep["return"].tolist() == [5.0, 5.0] and ep["length"].tolist() == [1.0, 3.0]


def test_checkpoint_round_trip(tmp_path):
    algo = _ppo()
    rms = RunningMeanStd(6)
    rms.update(torch.randn(10, 6))
    save_checkpoint(tmp_path / "c.pt", model=algo.model, optimizer=algo.optimizer, obs_norm=rms,
                    value_norm=algo.value_norm, iteration=3, global_step=96, learning_rate=1e-4, num_updates=24,
                    env_state={"epoch": 7}, resolved_cfg={"a": 1})
    ckpt = load_checkpoint(tmp_path / "c.pt", "cpu")
    assert ckpt["iteration"] == 3 and ckpt["env_state"] == {"epoch": 7}
    restored = RunningMeanStd(6)
    restored.load_state_dict(ckpt["obs_norm"])
    assert torch.equal(restored.mean, rms.mean)
