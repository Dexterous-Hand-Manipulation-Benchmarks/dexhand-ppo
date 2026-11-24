"""Hydra configs (presets, schema validation) and a short end-to-end training run on Pendulum."""
import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.errors import ConfigCompositionException
from omegaconf import open_dict

from dexhand_ppo.config import register_configs

register_configs()

PRESET_UPDATES_PER_ITERATION = {"baseline": 4 * 5, "default": 16 * 5, "reppo_budget": 256 * 4}


def _compose(overrides):
    GlobalHydra.instance().clear()
    with initialize_config_module(config_module="dexhand_ppo.configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)
    with open_dict(cfg):
        cfg.run_dir = "unused"  # ${hydra:...} is only resolvable inside a Hydra run
    return cfg


@pytest.mark.parametrize("preset", PRESET_UPDATES_PER_ITERATION)
def test_presets_compose_with_dexmachina(preset):
    cfg = _compose([f"preset={preset}"])
    assert cfg.preset_name == preset
    assert cfg.ppo.num_minibatches * cfg.ppo.num_epochs == PRESET_UPDATES_PER_ITERATION[preset]
    batch = cfg.env.num_envs * cfg.ppo.num_steps
    assert batch % cfg.ppo.num_minibatches == 0
    assert cfg.total_steps // batch == 381  # lab report Table I: 50M steps = 381 iterations of 4096 x 32


@pytest.mark.parametrize("override", ["env.dexmachina.rand_init_ratio=0.02", "env.dexmachina.args=[-act]",
                                      "env.dexmachina.curriculum_max_epochs=5000"])
def test_task_settings_cannot_be_chosen_per_run(override):
    # dexmachina's benchmark task fixes them for every method; a run may choose only hand and clip
    with pytest.raises(ConfigCompositionException):
        _compose([override])


def test_env_and_preset_files_override_base_values():
    cfg = _compose(["env=pendulum"])
    assert cfg.device == "cpu" and cfg.total_steps == 400_000 and cfg.ppo.num_steps == 256
    assert _compose(["preset=baseline"]).net.shared is True


def test_unknown_key_is_an_error():
    with pytest.raises(ConfigCompositionException):
        _compose(["ppo.actor_min_std=0.05"])


@pytest.fixture
def pendulum_cfg(tmp_path):
    cfg = _compose(["env=pendulum", "total_steps=8192", "eval.interval_steps=4096", "checkpoint.interval_steps=4096"])
    with open_dict(cfg):
        cfg.run_dir = str(tmp_path / "run")
    return cfg


def test_short_pendulum_run_writes_provenance_metrics_and_checkpoints(pendulum_cfg):
    pytest.importorskip("gymnasium")
    from dexhand_ppo.train import train

    result = train(pendulum_cfg)
    run_dir = pendulum_cfg.run_dir
    assert result["global_step"] == 8192
    provenance = json.loads(open(f"{run_dir}/provenance.json").read())
    assert provenance["repos"]["dexhand_ppo"]["commit"] is not None or provenance["repos"]["dexhand_ppo"]["root"]
    assert provenance["derived"]["iterations"] == 2
    lines = [json.loads(line) for line in open(f"{run_dir}/metrics.jsonl")]
    assert len(lines) == 2 and "eval/return" in lines[-1] and "ppo/kl" in lines[-1]
    assert lines[-1]["train/truncation_bootstraps"] >= 0
    for name in ("final.pt", "best.pt", "step_4096.pt"):
        assert Path(run_dir, "checkpoints", name).exists()


def test_resume_continues_from_checkpoint(pendulum_cfg, tmp_path):
    pytest.importorskip("gymnasium")
    from dexhand_ppo.train import train

    train(pendulum_cfg)
    with open_dict(pendulum_cfg):
        pendulum_cfg.checkpoint.resume = f"{pendulum_cfg.run_dir}/checkpoints/step_4096.pt"
        pendulum_cfg.run_dir = str(tmp_path / "resumed")
    result = train(pendulum_cfg)
    assert result["global_step"] == 8192
    lines = [json.loads(line) for line in open(f"{pendulum_cfg.run_dir}/metrics.jsonl")]
    assert [line["train/iteration"] for line in lines] == [1]


def test_seed_decides_initial_weights_even_if_env_construction_reseeds(pendulum_cfg, monkeypatch):
    """DexMachina's env construction calls np.random.seed(42) / torch.manual_seed(42); the run seed must still win."""
    pytest.importorskip("gymnasium")
    import numpy as np
    import torch

    import dexhand_ppo.train as train_module

    real_make_envs, real_model = train_module.make_envs, train_module.ActorCritic

    def make_envs_that_reseed(cfg):
        envs = real_make_envs(cfg)
        np.random.seed(42)
        torch.manual_seed(42)
        return envs

    class Captured(Exception):
        pass

    def capture_initial_weights(*args, **kwargs):
        model = real_model(*args, **kwargs)
        raise Captured(torch.cat([p.detach().flatten().clone() for p in model.parameters()]))

    monkeypatch.setattr(train_module, "make_envs", make_envs_that_reseed)
    monkeypatch.setattr(train_module, "ActorCritic", capture_initial_weights)
    weights = {}
    for seed in (0, 1, 0):
        with open_dict(pendulum_cfg):
            pendulum_cfg.seed = seed
        with pytest.raises(Captured) as captured:
            train_module.train(pendulum_cfg)
        weights.setdefault(seed, []).append(captured.value.args[0])
    assert not torch.equal(weights[0][0], weights[1][0])
    assert torch.equal(weights[0][0], weights[0][1])
