# Parameter Configuration Table

## A) Command-Line Reproduction Parameters

| Stage | Command | Key Parameters | Value |
|---|---|---|---|
| Setup | `python backbone_setup.py` | mode | safe setup (no-backbone compatible) |
| PPO pretrain | `python train.py ...` | `--algo` | `ppo` |
| PPO pretrain | `python train.py ...` | `--env` | `MISOEnv-antenna-2` |
| PPO pretrain | `python train.py ...` | `--tensorboard-log` | `./logs/` |
| PPO pretrain | `python train.py ...` | `--n-timesteps` | `25000` (quick run) |
| GRPO train | `python3 GRPO.py ...` | `--env` | `MISOEnv-antenna-2` |
| GRPO train | `python3 GRPO.py ...` | `--exp-id` | `1` / `11` |
| GRPO train | `python3 GRPO.py ...` | `--algo` | `ppo` |
| GRPO train | `python3 GRPO.py ...` | `--folder` | `./logs/` |
| GRPO train | `python3 GRPO.py ...` | `--device` | `cuda` (recommended) |
| GRPO train | `python3 GRPO.py ...` | `--grpo-total-timesteps` | `3000000` |
| GRPO eval | `python3 GRPO.py ... --eval` | mode | evaluation only |

## B) PPO Hyperparameters (`hyperparams/ppo.yml`, `MISOEnv-antenna-2`)

| Parameter | Value |
|---|---|
| `normalize` | `true` |
| `n_envs` | `8` |
| `n_timesteps` | `3e5` |
| `policy` | `MlpPolicy` |
| `n_steps` | `128` |
| `batch_size` | `256` |
| `gamma` | `0.95` |
| `gae_lambda` | `0.99` |
| `n_epochs` | `10` |
| `learning_rate` | `lin_5e-5` |
| `clip_range` | `0.2` |
| `ent_coef` | `1e-3` |
| `target_kl` | `0.02` |
| `vf_coef` | `0.8855255956624888` |
| `max_grad_norm` | `0.3` |
| `sde_sample_freq` | `4` |
| `policy_kwargs` | `pi/vf=[256,256], ReLU, log_std_init=-2.0, ortho_init=False` |

## C) GRPO Internal Training Parameters (`GRPO.py`)

| Parameter | Value |
|---|---|
| `trajectories_per_update` (group size) | `50` |
| `trajectory_len` | `50` |
| `beta` | `1e-5` |
| `gamma` | `0.99` |
| `value_coef` | `0.5` |
| `entropy_coef` | `0.01` |
| `eval_every` | `10` |
| `eval_episodes` | `10` |
| `episode_horizon` | `50` |

## D) Environment/Reward Definition (`custom_envs/MISOenv.py`)

| Item | Current Definition |
|---|---|
| Reward type | dense (per-step) |
| Reward formula | `reward = (sum_rate - 10 * step_penalty) / max_steps` |
| `max_steps` | `50` |
| Channel error model | enabled (robust rate uses estimated channel + sampled errors) |
