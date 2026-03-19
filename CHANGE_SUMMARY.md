# Change Summary (Current vs Initial Baseline)

This file lists the meaningful experiment changes introduced after the initial baseline.

## 1) Runtime/Setup Stability

- `backbone_setup.py`
  - Added Windows-safe deletion for read-only files/folders.
  - Exclusion list expanded to avoid copying bulky/non-essential artifacts.
  - Setup is now no-fail when `backbone/` or patch files are absent.
- `train.py`
  - Added `KMP_DUPLICATE_LIB_OK=TRUE` to bypass OpenMP runtime conflict on Windows.
- `GRPO.py`
  - Added `KMP_DUPLICATE_LIB_OK=TRUE` for the same runtime issue.

## 2) PPO Configuration Changes

- `hyperparams/ppo.yml`
  - Added `MISOEnv-antenna-2` PPO block (was absent in baseline).
  - Current learning rate policy is `lin_5e-5` (decay schedule for fine-tuning stability).

## 3) Environment/Reward Logic Changes

- `custom_envs/MISOenv.py`
  - Reward changed from terminal-only style to dense per-step reward.
  - Current formula:
    - `r_t = (sum_rate - 10 * step_penalty) / max_steps`

## 4) GRPO Training Logic Changes

- `GRPO.py`
  - Trajectory collection now records actual per-step reward on every step (dense reward consistent).
  - Added total-step control argument:
    - `--grpo-total-timesteps`
  - Training loop now targets collected environment steps for fairer PPO/GRPO comparisons.
  - Eval logging available via `--eval`, with TensorBoard tags used for plotting.

## 5) Environment Registration / Channel Model

- `rl_zoo3/import_envs.py`
  - Added dynamic registration for `MISOEnv-antenna-*` configs.
- `custom_envs/field_response_channel.py`
  - Channel gain scale adjusted (`Sigma` amplitude factor increased from `0.0001` to `0.01`).

## 6) Repository Slimming (for GitHub upload)

Removed non-essential artifacts for reproducibility-focused upload:

- training outputs/logs/checkpoints
- docs/tests/images/docker helper folders
- large backbone mirror and bundled pretrained agents
- non-PPO hyperparameter files

Core runnable path retained:

- `backbone_setup.py`
- `train.py`
- `GRPO.py`
- `custom_envs/`
- `rl_zoo3/`
- `hyperparams/ppo.yml`
- dependency/build files
