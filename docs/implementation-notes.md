# Implementation notes

Every implementation detail that could matter for a report, especially where we deviate from the PPO paper
(Schulman et al., 2017), from dexmachina's rl_games configuration (lab report Table I) or from the common
"37 implementation details of PPO" (Huang et al., 2022). Updated in the same commit as the code it describes.

## Design goals
- A PPO for DexMachina that can be trusted and improved in measurable steps. The lab report attributes PPO's deficit
  mainly to its update budget (7,620 gradient updates in 50M env steps, against REPPO's 389k), so the update budget
  is a free parameter, and every deviation from the rl_games baseline is a separate flag that can be ablated.
- Env handling, evaluation and logging correct by construction (terminal handling, unassisted evaluation, true env
  steps).
- Not a goal: a general RL library, multi-GPU, vision, changing the env or the benchmark protocol, or rl_games
  compatibility (checkpoints, config format).

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

## DexMachina adapter (`envs/dexmachina.py`)
- The env is dexmachina's benchmark task: `benchmark_spec(RunChoices(hand, clip, seed, num_envs))`
  (`dexmachina.envs.benchmark_task`), built with `DexMachinaVecEnv.from_spec` and no reward overrides. A run chooses
  only `env.dexmachina.hand`, `env.dexmachina.clip` (one of the benchmark clips), `seed` and `env.num_envs`; every
  other env setting is fixed by the task, the same for every method. dexhand-ppo has no env settings of its own.
- Change from the earlier adapter, which built the env from its own copy of the lab report's Table VI arg list: the
  arg list matched the benchmark task in every field (compared field by field and on the full `get_all_env_cfg`
  output, 2026-11-23) except the share of random start frames, `rand_init_ratio`: 0.02 before (`default`,
  `reppo_budget`), now the task's value (0.0 as of 2026-11-23, the `EnvSpec` default). This changes training results
  of the `default` and `reppo_budget` presets; `baseline` had 0.0 already.
- The run's `EnvSpec` and its `task_hash` go into `provenance.json`; `env_spec.json` (dexmachina's `save_run_spec`:
  spec, reward overrides, dexmachina commit) is written into `checkpoints/` before training, so evaluation can
  rebuild the task from a checkpoint's directory. The file is never overwritten, so a resumed run needs a new run dir
  (the default run dir has a timestamp).
- Verified on the built env (2026-11-11, CPU): obs 510, actions 44, episode length 200; zero_epoch 9000, warm-up 100,
  decay ratios 0.9/0.8 (fr 1.0), thresholds task 0.6 / others 0.01, kp/kv init 80/5; rewards 1/0.3/0.3/3, task betas
  10/1/5, contact beta 10, bc beta 500.
- `gs.init` once per process (backend from `device`, seed = run seed). Building a DexMachina env reseeds the global
  numpy/torch RNGs with 42 (curriculum constructor; numpy also in the ADD vertex sampling). `train.py` therefore
  seeds random/numpy/torch with the run seed again right after `make_envs`, before the networks are created, so
  the run seed decides the initial weights, action sampling and the env's own draws from the global generators
  (e.g. the uniform curriculum's per-env gains). Before this fix every seed started from identical weights and an
  identical sampling stream. Test: `test_seed_decides_initial_weights_even_if_env_construction_reseeds`.
- Every step: actions clamped to [−1, 1] by the env; obs and reward come as copies from `DexMachinaVecEnv`; NaN/inf in obs,
  reward and final obs replaced by 0 and counted (`train/nan_obs_envs`, `train/nan_rewards`). `terminated` and
  `truncated` are `DexMachinaVecEnv`'s, from the cause of each episode end: in the benchmark task every end (demo
  end, drop, early reset, NaN) is terminated and none truncated, so no value is bootstrapped (consistent because the
  obs contains t/T). `final_obs` (the obs before the reset) is passed on for truncation bootstrapping in tasks that
  have truncations.
- Episode statistics: return, length and per-step means of `task_rew`, `imi_rew`, `bc_rew`, `con_rew` per finished
  episode (`train/episode_*`, window of the last 100 episodes). Per-step means of `pos_dist`, `rot_dist`, `arti_dist`,
  `success_2cm/5cm` over each rollout (`env/*`, **assisted** physics).
- Curriculum: before each rollout, `epoch = min(global_step · Emax // total_env_steps, Emax)` (Emax = the task's
  `max_epochs`, 10,000) with
  `total_env_steps = iterations · num_envs · num_steps`; `set_curriculum(epoch)` only when the epoch increases (381
  calls over a 50M run; the spacing between reductions is counted in epochs since dexmachina `38ff004`). Logged:
  `curriculum/epoch`, `curriculum/num_reductions` (calls that lowered the gains), `curriculum/num_dialbacks` (raised
  them), `curriculum/hard_stop_applied` (gains zeroed by the hard stop after zero_epoch), `env/kp|kv|fr` (upper bounds).
- Resume restores the counters and gains (and re-applies them to the object); the curriculum's reward and length
  deques start empty again.

## DexMachina evaluation (`DexMachinaEvaluator`)
- The team's shared evaluation (dexmachina `envs/evaluation.py`): one env from `make_eval_env(spec)` (`is_eval`, no
  assistance, object not actuated, no early reset, no randomisation, ADD metric on, outside the curriculum
  statistics). Each evaluation runs `zero_action_floor` (one episode with zero actions) and then one full-length
  episode with the deterministic (mean) action via `rollout_episode`, both on that env, and scores both with `score`.
- Logged: every scalar of the policy's `EvalScore` as `eval/<name>` (`auc_add`: AUC per object part over 1–9 cm, then
  the mean, the paper's definition and the selection metric; `auc_add_of_mean`: REPPO's online definition;
  `mean_add`; ADD-based `success_2cm|5cm`; `nan_rate`; `drop_rate`), the AUC of each part as
  `eval/auc_add_part/<part>`, and the same for the floor as `eval/floor/<name>` (FlashSAC's names for these scores).
  Scores are raw, not normalised by the floor.
- The policy episode's trajectory (dexmachina `save_trajectory`: object states, ADD per part, NaN, reset and drop
  flags, paired demo frames) is saved as `eval_trajectories/trajectory_step_<env steps>.npz`, so its scores can be
  recomputed with `score(load_trajectory(path))`; the last one is the evaluation of `final.pt`. The online
  evaluation sets no seed: nothing random reaches it (one env, no randomisation, mean action), and reseeding the
  global generators would change the training run's random stream.
- Change from the earlier evaluator: it ran 16 deterministic envs (plus dexmachina's dummy env) with the object
  actuated at zero gains and its own ADD bookkeeping; `eval/success_2cm|5cm` were dexmachina's pos/rot/arti flags and
  are now ADD-based (the earlier `eval/add_success_*`), `eval/auc_add_part_mean` is now `eval/auc_add_of_mean`,
  `eval/add` is `eval/mean_add`, `eval/pos_dist` is gone. The shared recorder pairs the state after step t with demo
  frame t + 1 and scores the last step before its reset. Scores of the earlier evaluator (and REPPO's online
  evaluation, also actuated at zero gains) are therefore not comparable with these.

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
- 2026-11-11, same Pendulum setup with the `baseline` preset (shared trunk, no obs normalisation, value coef 4 with
  clipping): eval −206 at 53k steps, −115 at 100k, −113 to −131 until 400k (solved; slightly slower start than
  `default`).
