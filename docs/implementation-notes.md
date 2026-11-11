# Implementation notes

Every implementation detail that could matter for a report, especially where we deviate from the PPO paper
(Schulman et al., 2017), from dexmachina's rl_games configuration (lab report Table I) or from the common
"37 implementation details of PPO" (Huang et al., 2022). Updated in the same commit as the code it describes.

## Configuration
- Hydra with a structured schema (`dexhand_ppo/config.py`): unknown or misspelled keys are errors.
- Defaults-list order: base values, then `preset/`, then `env/`. An env file may override preset values (Pendulum
  sets its own rollout length and update layout).
- Presets: `baseline` (rl_games equivalent), `default`, `reppo_budget`. See the README table and `configs/preset/`.

## Rollout
- Same-step autoreset: the obs returned for a finished env is the first obs of its next episode (DexMachina and
  gymnasium `AutoresetMode.SAME_STEP`).
- Observation normaliser is updated with every rollout step's raw obs *before* normalising it; stored obs are
  normalised with the statistics of that moment and are not re-normalised during the update (frozen normaliser).
  Normalised obs are clipped to ±`obs_clip` (5). With `normalize_obs=false` (baseline), obs are only clipped, as
  rl_games did after dexmachina's own ±5 clip.
- Actions are sampled from an unsquashed diagonal Gaussian; the env clamps them to [−1, 1]. The stored log-prob is
  that of the unclipped sample (as rl_games and most PPO implementations).
- Rewards are scaled by `reward_scale` (0.1, as rl_games) before GAE.
- `done = terminated | truncated` cuts the GAE trace. On a truncation (not a termination) with a reported final obs,
  `gamma * V(final_obs)` is added to that step's (scaled) reward (`bootstrap_truncation`). DexMachina reports no
  truncations, so all its episode ends are terminal (decided: the obs contains t/T, so the finite-horizon return is
  exact; same as rl_games and FastTD3).

## Returns and value targets
- GAE(γ, λ) with V(s_T) bootstrap after the last rollout step (`storage.compute_gae`).
- Value normalisation (`normalize_value`): the critic predicts normalised values; GAE uses denormalised values. In
  the update, the return normaliser is first updated with the batch's returns, then targets and old values are
  normalised with the new statistics (as rl_games' `value_mean_std`).
- Advantages are normalised per minibatch (as rl_games).

## Update
- `num_epochs` passes; each pass is a fresh random partition into `num_minibatches` equal minibatches (batch size
  must be divisible). Unlike rl_games, neither depends on the number of environments.
- Loss = −clipped surrogate + `value_coef` · 0.5 · MSE (optionally clipped around the old value by ±`clip_ratio` in
  normalised units, as rl_games) − `entropy_coef` · entropy + `bounds_coef` · bounds loss. rl_games' `critic_coef 4`
  corresponds to `value_coef 4` here (same 0.5 factor).
- Bounds loss: squared excess of the action mean beyond ±1.1, summed over action dims, averaged over the batch
  (rl_games).
- One Adam optimiser over all parameters (eps 1e-8), global gradient-norm clipping (`max_grad_norm`).
- KL: analytic KL(old ‖ new) of the diagonal Gaussians, summed over action dims, computed per minibatch from the
  pre-step outputs (as rl_games' `policy_kl`).
- `adaptive_kl` learning rate: once per epoch, from the mean KL of that epoch's minibatches: ÷1.5 if KL > 2·target,
  ×1.5 if KL < target/2, clamped to [1e-6, 1e-2] (rl_games' rule). `linear` decays to 0 over the run.
- Optional KL early stop (`kl_stop` > 0): the update stops as soon as a minibatch's KL exceeds `kl_stop · kl_target`
  (used by `reppo_budget`).

## Networks
- MLPs with orthogonal init (gain √2), zero biases; policy mean head gain 0.01, value head gain 1.0 (37 details).
- State-independent learned log std, initialised to log σ0 (1.0); optional lower bound `min_std` via clamping
  (gradient is zero below the bound).
- `net.shared=true`: one trunk with two heads (rl_games); otherwise separate actor and critic.

## Training loop and budget
- Iterations = `total_steps // (num_envs · num_steps)` (integer; 50M → 381 iterations of 4096 × 32 = 49.9M steps).
- Evaluation and periodic checkpoints whenever the env-step counter crosses a multiple of their interval, plus at
  the end. `best.pt` by the evaluator's selection metric (for inspection); `final.pt` is the one to report.

## Provenance (every run)
- `config.json`: fully resolved config. `provenance.json`: command, git commit/branch/dirty flag of dexhand-ppo,
  dexmachina, Genesis and dexhand-bench (diffs of dirty repos saved as `git_diff_<repo>.patch`), package versions,
  hardware, relevant env vars, derived quantities (iterations, batch size, minibatch size, max updates), env
  description. Also attached to the W&B run config.
- `metrics.jsonl`: every logged metric of every iteration, independent of W&B.

## Validation log
- 2026-11-11, Pendulum-v1 (16 envs × 256 steps, 10 epochs × 16 minibatches, γ 0.95, lr 1e-3, `default` network
  settings, CPU): deterministic eval return −130 after 53k steps, −113 ± 1 from 100k to 400k steps (solved;
  random ≈ −1200).
