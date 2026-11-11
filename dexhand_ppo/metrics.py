"""Metric logging: always to metrics.jsonl in the run dir, optionally to W&B (via the umbrella's UniversalLogger)."""
import json
import time
from pathlib import Path

try:  # unified W&B metric names when run from the dexhand-bench umbrella
    from dexhand_bench.wandb_logger import UniversalLogger
except ModuleNotFoundError as e:
    if e.name != "dexhand_bench":
        raise
    UniversalLogger = None


class MetricLogger:
    def __init__(self, cfg, run_dir: str, resolved_cfg: dict, provenance: dict):
        self.path = Path(run_dir) / "metrics.jsonl"
        self.start = time.time()
        self.wandb = None
        if cfg.log.wandb:
            import wandb
            if cfg.env.kind == "dexmachina":
                if UniversalLogger is None:
                    print("[INFO] dexhand_bench not installed: unified W&B metric logging disabled")
                else:
                    d = cfg.env.dexmachina
                    UniversalLogger.install(algo="ppo", hand=d.hand, clip=d.clip, seed=cfg.seed,
                                            num_envs=cfg.env.num_envs, num_eval_envs=d.num_eval_envs,
                                            horizon_length=cfg.ppo.num_steps)
            self.wandb = wandb.init(project=cfg.log.project, entity=cfg.log.entity, group=cfg.log.group,
                                    name=cfg.exp_name, dir=run_dir,
                                    config={**resolved_cfg, "provenance": provenance})

    def log(self, metrics: dict, step: int) -> None:
        record = {"global/env_step": step, "time/elapsed_s": time.time() - self.start, **metrics}
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=float) + "\n")
        if self.wandb is not None:
            self.wandb.log(record, step=step)

    def close(self) -> None:
        if self.wandb is not None:
            self.wandb.finish()
