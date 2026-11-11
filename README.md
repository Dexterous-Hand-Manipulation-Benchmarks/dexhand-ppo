# dexhand-ppo

PPO for bimanual dexterous manipulation on [DexMachina](https://github.com/MandiZhao/dexmachina), part of
[dexhand-bench](https://github.com/Dexterous-Hand-Manipulation-Benchmarks/dexhand-bench).

Compared with dexmachina's rl_games baseline, the number of gradient updates per environment step is a free
parameter (minibatches and epochs no longer scale with the number of environments), and every deviation from the
baseline is a separate configuration flag. See `docs/implementation-notes.md` for all implementation details.

## Presets
| Preset | Updates per iteration (4096 envs x 32 steps) | Purpose |
|---|---|---|
| `baseline` | 4 x 5 = 20 | reproduces the rl_games configuration (shared trunk, no obs normalisation, ...) |
| `default` | 16 x 5 = 80 | proposed defaults: separate actor/critic, obs normalisation, std lower bound |
| `reppo_budget` | 256 x 4 = 1,024 | about REPPO's update budget (~390k updates in 50M steps) |

## Usage
Inside the dexhand-bench umbrella (installs genesis and dexmachina):
```bash
uv run --exact --extra dexhand-ppo python -m dexhand_ppo.train preset=default env.dexmachina.hand=allegro_hand
```
CPU sanity check on Pendulum (standalone):
```bash
uv sync --group dev
uv run python -m dexhand_ppo.train env=pendulum
uv run pytest
```
Every run writes `config.json`, `provenance.json` (git commits and diffs of all involved repos, package versions,
hardware), `metrics.jsonl` and checkpoints to its run dir (`runs/<exp_name>/<timestamp>` unless `run_dir` is set).
