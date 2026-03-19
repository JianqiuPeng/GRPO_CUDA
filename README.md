# rt_grpo_ICC_Rate (Reproducible Minimal Version)

This repository is trimmed to reproduce the current PPO/GRPO experiments on `MISOEnv-antenna-2`.

## 1) Environment

Recommended Python: `3.10`

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2) Reproduction Commands

```bash
python backbone_setup.py
python train.py --algo ppo --env MISOEnv-antenna-2 --tensorboard-log ./logs/ --n-timesteps 25000
python3 GRPO.py --env MISOEnv-antenna-2 --exp-id 1 --algo ppo --folder ./logs/
python3 GRPO.py --env MISOEnv-antenna-2 --exp-id 1 --algo ppo --folder ./logs/ --eval
```

Full run (300w steps, GPU):

```bash
python3 GRPO.py --env MISOEnv-antenna-2 --exp-id 11 --algo ppo --folder ./logs/ --device cuda --grpo-total-timesteps 3000000
```

## 3) Notes

- `backbone_setup.py` is now backward-compatible and safe to run even if `backbone/` is absent.
- Logs and trained checkpoints are intentionally not tracked in this minimal repo.
- Main experiment parameters are documented in `PARAMETER_CONFIG.md`.
- Code changes relative to the initial baseline are documented in `CHANGE_SUMMARY.md`.
