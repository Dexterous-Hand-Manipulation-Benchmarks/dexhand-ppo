"""Training entry point: ``python -m dexhand_ppo.train [preset=...] [env=...] [key=value ...]``."""
import random
import time
from collections import defaultdict, deque
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from dexhand_ppo.checkpoint import CHECKPOINT_DIRNAME, load_checkpoint, save_checkpoint
from dexhand_ppo.config import register_configs
from dexhand_ppo.envs import make_envs
from dexhand_ppo.metrics import MetricLogger
from dexhand_ppo.networks import ActorCritic
from dexhand_ppo.normalizers import Identity, RunningMeanStd
from dexhand_ppo.ppo import PPO
from dexhand_ppo.provenance import write_provenance
from dexhand_ppo.storage import RolloutBuffer

register_configs()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def num_iterations(cfg) -> int:
    per_iteration = cfg.env.num_envs * cfg.ppo.num_steps
    iterations = cfg.total_steps // per_iteration
    if iterations < 1:
        raise ValueError(f"total_steps {cfg.total_steps} < one iteration ({per_iteration} steps)")
    return iterations


def crossed(prev_step: int, step: int, interval: int) -> bool:
    return interval > 0 and prev_step // interval != step // interval


def train(cfg) -> dict:
    resolved = OmegaConf.to_container(cfg, resolve=True)
    run_dir = Path(cfg.run_dir)
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    ppo_cfg = cfg.ppo

    train_env, evaluator = make_envs(cfg)
    # Seed again: building a DexMachina env reseeds the global numpy/torch RNGs with 42 (curriculum, ADD vertex
    # sampling), which would give every run seed the same initial weights and sampling stream.
    seed_everything(cfg.seed)
    iterations = num_iterations(cfg)
    if hasattr(train_env, "save_run_spec"):
        train_env.save_run_spec(run_dir / CHECKPOINT_DIRNAME)
    env_description = train_env.describe() if hasattr(train_env, "describe") else {}
    provenance = write_provenance(str(run_dir), resolved, extra={
        "env": env_description,
        "derived": {"iterations": iterations, "env_steps": iterations * cfg.env.num_envs * ppo_cfg.num_steps,
                    "batch_size": cfg.env.num_envs * ppo_cfg.num_steps,
                    "minibatch_size": cfg.env.num_envs * ppo_cfg.num_steps // ppo_cfg.num_minibatches,
                    "max_updates": iterations * ppo_cfg.num_epochs * ppo_cfg.num_minibatches},
    })

    model = ActorCritic(train_env.obs_dim, train_env.action_dim, cfg.net).to(device)
    obs_norm = (RunningMeanStd(train_env.obs_dim) if ppo_cfg.normalize_obs else Identity()).to(device)
    value_norm = (RunningMeanStd(()) if ppo_cfg.normalize_value else Identity()).to(device)
    algo = PPO(model, value_norm, ppo_cfg, iterations, device)
    algo.generator.manual_seed(cfg.seed)

    start_iteration, global_step, best_score = 0, 0, -float("inf")
    if cfg.checkpoint.resume:
        ckpt = load_checkpoint(cfg.checkpoint.resume, device)
        model.load_state_dict(ckpt["model"])
        algo.optimizer.load_state_dict(ckpt["optimizer"])
        obs_norm.load_state_dict(ckpt["obs_norm"])
        value_norm.load_state_dict(ckpt["value_norm"])
        algo.set_lr(ckpt["learning_rate"])
        algo.num_updates = ckpt["num_updates"]
        start_iteration, global_step = ckpt["iteration"], ckpt["global_step"]
        if hasattr(train_env, "load_state_dict"):
            train_env.load_state_dict(ckpt["env_state"])
        print(f"[INFO] resumed from {cfg.checkpoint.resume} at iteration {start_iteration}")

    def save(name: str, iteration: int) -> None:
        save_checkpoint(run_dir / CHECKPOINT_DIRNAME / name, model=model, optimizer=algo.optimizer, obs_norm=obs_norm,
                        value_norm=value_norm, iteration=iteration, global_step=global_step,
                        learning_rate=algo.learning_rate, num_updates=algo.num_updates,
                        env_state=train_env.state_dict() if hasattr(train_env, "state_dict") else {},
                        resolved_cfg=resolved)

    @torch.no_grad()
    def deterministic_policy(obs: torch.Tensor) -> torch.Tensor:
        return model.act_mean(obs_norm.normalize(obs, ppo_cfg.obs_clip))

    def run_eval() -> dict:
        model.eval()
        metrics = evaluator.evaluate(deterministic_policy, global_step)
        model.train()
        return metrics

    logger = MetricLogger(cfg, str(run_dir), resolved, provenance)
    buffer = RolloutBuffer(ppo_cfg.num_steps, train_env.num_envs, train_env.obs_dim, train_env.action_dim, device)
    episode_window = defaultdict(lambda: deque(maxlen=cfg.log.episode_window))
    obs = train_env.reset()
    print(f"[INFO] {iterations} iterations x {train_env.num_envs} envs x {ppo_cfg.num_steps} steps; "
          f"run dir {run_dir}")

    for iteration in range(start_iteration, iterations):
        env_metrics = train_env.on_iteration(global_step)
        t_sim = t_policy = 0.0
        t_rollout = time.perf_counter()
        num_truncation_bootstraps = 0
        for _ in range(ppo_cfg.num_steps):
            t0 = time.perf_counter()
            with torch.no_grad():
                obs_norm.update(obs)
                obs_n = obs_norm.normalize(obs, ppo_cfg.obs_clip)
                mu, values_n = model(obs_n)
                dist = model.distribution(mu)
                actions = dist.sample()
                log_probs = dist.log_prob(actions).sum(-1)
                values = value_norm.denormalize(values_n)
            t1 = time.perf_counter()
            next_obs, rewards, terminated, truncated, info = train_env.step(actions)
            t2 = time.perf_counter()
            t_policy += t1 - t0
            t_sim += t2 - t1
            with torch.no_grad():
                scaled = rewards * ppo_cfg.reward_scale
                if ppo_cfg.bootstrap_truncation and "final_obs" in info:
                    mask = truncated & ~terminated & info["final_obs_mask"]
                    if mask.any():
                        final_n = obs_norm.normalize(info["final_obs"][mask], ppo_cfg.obs_clip)
                        scaled[mask] += ppo_cfg.gamma * value_norm.denormalize(model.value(final_n))
                        num_truncation_bootstraps += int(mask.sum())
                dones = (terminated | truncated).float()
                buffer.add(obs_n, actions, log_probs, mu, dist.stddev, values, scaled, dones)
            for key, vals in info.get("episode", {}).items():
                episode_window[key].extend(vals.tolist())
            obs = next_obs
            global_step += train_env.num_envs
        with torch.no_grad():
            last_values = value_norm.denormalize(model.value(obs_norm.normalize(obs, ppo_cfg.obs_clip)))
        buffer.finish(last_values, ppo_cfg.gamma, ppo_cfg.gae_lambda)
        rollout_time = time.perf_counter() - t_rollout

        t_update = time.perf_counter()
        update_stats = algo.update(buffer, iteration)
        update_time = time.perf_counter() - t_update

        steps_per_iter = train_env.num_envs * ppo_cfg.num_steps
        metrics = {f"ppo/{k}": v for k, v in update_stats.items()}
        metrics.update({f"train/episode_{k}" if k not in ("return", "length") else f"train/{k}": float(np.mean(v))
                        for k, v in episode_window.items() if v})
        metrics.update(env_metrics)
        metrics.update({"train/iteration": iteration, "train/truncation_bootstraps": num_truncation_bootstraps,
                        "perf/fps": steps_per_iter / (rollout_time + update_time),
                        "perf/sim_fps": steps_per_iter / max(t_sim, 1e-9), "perf/rollout_s": rollout_time,
                        "perf/policy_s": t_policy, "perf/update_s": update_time})
        prev_step = global_step - steps_per_iter
        last_iteration = iteration == iterations - 1
        if evaluator is not None and (crossed(prev_step, global_step, cfg.eval.interval_steps) or last_iteration):
            eval_metrics = run_eval()
            metrics.update(eval_metrics)
            score = eval_metrics.get(evaluator.selection_metric)
            if score is not None and score > best_score:
                best_score = score
                save("best.pt", iteration + 1)
                metrics["best/" + evaluator.selection_metric.split("/", 1)[-1]] = score
        if crossed(prev_step, global_step, cfg.checkpoint.interval_steps):
            save(f"step_{global_step}.pt", iteration + 1)
        logger.log(metrics, step=global_step)
        print(f"[it {iteration + 1}/{iterations}] step {global_step} "
              + " ".join(f"{k}={metrics[k]:.4g}" for k in ("train/return", "eval/return", "eval/auc_add",
                                                           "ppo/kl", "ppo/learning_rate", "perf/fps")
                         if k in metrics))

    save("final.pt", iterations)
    logger.close()
    train_env.close()
    if evaluator is not None:
        evaluator.close()
    return {"global_step": global_step, "best_score": best_score}


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
