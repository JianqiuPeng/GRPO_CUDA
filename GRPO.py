# NOTE: Run PPO pretraining on MISOEnv-custom and save the checkpoint before using GRPO for fine-tuning.
# -*- coding: utf-8 -*-
import argparse
import importlib
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import copy

import numpy as np
import torch as th
import yaml
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.callbacks import tqdm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed

from torch.utils.tensorboard import SummaryWriter

import rl_zoo3.import_envs  # noqa: F401 pylint: disable=unused-import
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.load_from_hub import download_from_hub
from rl_zoo3.utils import StoreDict, get_model_path

from tqdm import tqdm

device = th.device("cuda" if th.cuda.is_available() else "cpu")

group_num = 1  # This should be 1 in default
trajectories_per_update = 50  # This is the group size
trajectory_len = 50  # This is the horizon length
beta = 1e-5
gamma = 0.99
value_coef = 0.5
entropy_coef = 0.01
eval_every = 10  # Evaluate policy every N GRPO epochs
eval_episodes = 10  # Number of evaluation episodes
episode_horizon = 50  # Max episode length for reward normalization
eval_every = 10  # Evaluate policy every N GRPO epochs
eval_episodes = 10  # Number of evaluation episodes


def ensure_finite_parameters(policy):
    """Guard against NaNs/inf values when loading pretrained policies."""
    with th.no_grad():
        for param in policy.parameters():
            param.data = th.nan_to_num(param.data)
        if hasattr(policy, "log_std"):
            policy.log_std.data = th.nan_to_num(policy.log_std.data)


def sanitize_observation(obs):
    """Clamp observations to a safe numeric range."""
    arr = np.asarray(obs, dtype=np.float32)
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False), -100.0, 100.0)


def sanitize_action(action):
    """Clamp agent actions to avoid extreme values."""
    return np.clip(np.nan_to_num(action, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)


def sanitize_reward(reward):
    """Convert reward to a finite scalar."""
    reward_arr = np.asarray(reward, dtype=np.float64)
    reward_arr = np.nan_to_num(reward_arr, nan=0.0, posinf=0.0, neginf=0.0)
    return float(reward_arr.item()) if reward_arr.size == 1 else float(np.mean(reward_arr))


def squeeze_env_axis(array):
    """Drop the leading VecEnv axis when it corresponds to a single environment."""
    arr = np.asarray(array)
    if arr.ndim >= 1 and arr.shape[0] == 1:
        return np.squeeze(arr, axis=0)
    return arr


def collect_trajectory(env, policy, trajectory_len=10, deterministic=False):
    """Roll out a single trajectory and gather data for policy updates."""
    reset_output = env.reset()
    obs = sanitize_observation(reset_output)
    observations = []
    log_probs = []
    chosen_actions = []
    rewards = []
    values = []
    dones = []

    for _ in range(trajectory_len):
        obs = sanitize_observation(obs)
        obs_tensor = th.nan_to_num(th.as_tensor(obs, device=device, dtype=th.float32)).clamp_(-100.0, 100.0)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        with th.no_grad():
            action_tensor, value_tensor, log_prob = policy(obs_tensor, deterministic=deterministic)

        action = action_tensor.cpu().numpy()
        env_action = sanitize_action(action)
        stored_obs = np.array(squeeze_env_axis(obs), copy=True)
        stored_action = np.array(squeeze_env_axis(env_action), copy=True)
        value = float(value_tensor.squeeze().cpu().item())
        log_prob_scalar = float(log_prob.sum().cpu().item())

        observations.append(stored_obs)
        log_probs.append(log_prob_scalar)
        chosen_actions.append(stored_action)
        values.append(value)

        obs, reward, done, infos = env.step(env_action)
        obs = sanitize_observation(obs)

        reward_scalar = reward[0] if isinstance(reward, (np.ndarray, list, tuple)) else reward
        done_flag = bool(done[0]) if isinstance(done, (np.ndarray, list, tuple)) else bool(done)
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        dones.append(done_flag)

        # Dense reward: keep the actual per-step reward for GRPO updates.
        rewards.append(sanitize_reward(reward_scalar))

        if done_flag:
            break

    if dones and dones[-1]:
        next_value = 0.0
    else:
        obs = sanitize_observation(obs)
        obs_tensor = th.nan_to_num(th.as_tensor(obs, device=device, dtype=th.float32)).clamp_(-100.0, 100.0)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        with th.no_grad():
            next_value = float(policy.predict_values(obs_tensor).cpu().item())

    total_reward = float(np.sum(rewards)) if rewards else 0.0
    return {
        "observations": observations,
        "log_probs": log_probs,
        "actions": chosen_actions,
        "rewards": rewards,
        "values": values,
        "dones": dones,
        "next_value": next_value,
        "return_sum": total_reward,
    }


def grpo_update(
    trajectories,
    policy,
    eps=0.05,
    n_epochs=1,
    batch_size=128,
    max_grad_norm=None,
    ref_policy=None,
    writer=None,
    epi=None,
):
    """Apply GRPO-style updates using trajectory-level relative advantages."""
    policy.set_training_mode(True)

    rewards = np.array([traj["return_sum"] for traj in trajectories], dtype=np.float32)
    if rewards.size == 0:
        return
    std = rewards.std()
    if std < 1e-8:
        return
    advantages = (rewards - rewards.mean()) / (std + 1e-8)
    advantages_tensor = th.as_tensor(advantages, device=device, dtype=th.float32)

    obs_list = []
    action_list = []
    lengths = []
    old_log_sums = []
    for traj in trajectories:
        observations = np.stack(traj["observations"])
        actions = np.stack(traj["actions"])
        obs_list.append(observations)
        action_list.append(actions)
        lengths.append(len(traj["log_probs"]))
        old_log_sums.append(np.sum(traj["log_probs"]))

    if not obs_list:
        return

    obs_tensor = th.as_tensor(np.concatenate(obs_list, axis=0), device=device, dtype=th.float32)
    actions_tensor = th.as_tensor(np.concatenate(action_list, axis=0), device=device, dtype=th.float32)
    old_log_sums_tensor = th.as_tensor(np.array(old_log_sums), device=device, dtype=th.float32)
    values_pred, new_log_prob, entropy = policy.evaluate_actions(obs_tensor, actions_tensor)
    if new_log_prob.dim() > 1:
        new_log_prob = new_log_prob.sum(dim=-1)
    if entropy.dim() > 1:
        entropy = entropy.mean(dim=-1)
    if not th.isfinite(new_log_prob).all():
        return

    ref_log_prob = None
    if ref_policy is not None and beta > 0:
        with th.no_grad():
            _, ref_log_prob, _ = ref_policy.evaluate_actions(obs_tensor, actions_tensor)
        if ref_log_prob.dim() > 1:
            ref_log_prob = ref_log_prob.sum(dim=-1)

    policy_losses = []
    entropy_terms = []
    kl_terms = []

    idx_start = 0
    for traj_idx, (adv_value, traj_len) in enumerate(zip(advantages_tensor, lengths)):
        idx_end = idx_start + traj_len
        new_log_prob_sum = new_log_prob[idx_start:idx_end].sum()
        ratio = th.exp(new_log_prob_sum - old_log_sums_tensor[traj_idx])
        clipped_ratio = th.clamp(ratio, min=1 - eps, max=1 + eps)
        policy_losses.append(-th.min(ratio * adv_value, clipped_ratio * adv_value))
        entropy_terms.append(entropy[idx_start:idx_end].mean())

        if ref_log_prob is not None:
            log_ratio = th.clamp(ref_log_prob[idx_start:idx_end].sum() - new_log_prob_sum, max=10)
            kl_terms.append((th.exp(log_ratio) - 1) - log_ratio)

        idx_start = idx_end

    if not policy_losses:
        return

    policy_loss = th.stack(policy_losses).mean()
    entropy_loss = th.stack(entropy_terms).mean() if entropy_terms else th.tensor(0.0, device=device)

    kl_term = None
    if kl_terms:
        kl_term = th.stack(kl_terms).mean()
        policy_loss = policy_loss + beta * kl_term

    total_loss = policy_loss - entropy_coef * entropy_loss
    if not th.isfinite(total_loss):
        return

    policy.optimizer.zero_grad()
    total_loss.backward()
    if max_grad_norm is not None:
        th.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    policy.optimizer.step()

    if writer is not None and epi is not None:
        writer.add_scalar('Training/PolicyLoss', policy_loss.item(), epi)
        writer.add_scalar('Training/Entropy', entropy_loss.item(), epi)
        if kl_term is not None:
            writer.add_scalar('Training/KL', kl_term.item(), epi)



def log_dir_gen(folder, alg):

    import re
    if not os.path.exists(folder):
        os.makedirs(folder)

    existing_dirs = os.listdir(folder)
    pattern = re.compile(rf"^{alg}_(\d+)$")

    max_x = 0
    for d in existing_dirs:
        match = pattern.match(d)
        if match:
            max_x = max(max_x, int(match.group(1)))

    return os.path.join(folder, f"{alg}_{max_x + 1}")


def gen_env_model_path():

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=EnvironmentName, default="CartPole-v1")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False,
                        choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="auto", type=str)
    parser.add_argument(
        "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, "
             "you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument(
        "--load-last-checkpoint",
        action="store_true",
        default=False,
        help="Load last checkpoint instead of last model if available",
    )
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument(
        "--norm-reward", action="store_true", default=False,
        help="Normalize reward if applicable (trained with VecNormalize)"
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environment package modules to import",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict,
        help="Optional keyword argument to pass to the env constructor"
    )
    parser.add_argument(
        "--custom-objects", action="store_true", default=False, help="Use custom objects to solve loading issues"
    )
    parser.add_argument(
        "-P",
        "--progress",
        action="store_true",
        default=False,
        help="if toggled, display a progress bar using tqdm and rich",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="if toggled, run evaluation instead of training",
    )
    parser.add_argument(
        "--grpo-total-timesteps",
        type=int,
        default=3000000,
        help="Total environment timesteps to collect during GRPO fine-tuning",
    )
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_name: EnvironmentName = args.env
    algo = args.algo
    folder = args.folder

    try:
        _, model_path, log_path = get_model_path(
            args.exp_id,
            folder,
            algo,
            env_name,
            args.load_best,
            args.load_checkpoint,
            args.load_last_checkpoint,
        )
    except (AssertionError, ValueError) as e:
        # Special case for rl-trained agents
        # auto-download from the hub
        if "rl-trained-agents" not in folder:
            raise e
        else:
            print(
                "Pretrained model not found, trying to download it from sb3 Huggingface hub: https://huggingface.co/sb3")
            # Auto-download
            download_from_hub(
                algo=algo,
                env_name=env_name,
                exp_id=args.exp_id,
                folder=folder,
                organization="sb3",
                repo_name=None,
                force=False,
            )
            # Try again
            _, model_path, log_path = get_model_path(
                args.exp_id,
                folder,
                algo,
                env_name,
                args.load_best,
                args.load_checkpoint,
                args.load_last_checkpoint,
            )

    print(f"Loading {model_path}")

    # Off-policy algorithm only support one env for now
    set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    log_dir = args.reward_log if args.reward_log != "" else None

    try:
        env = create_test_env(
            env_name.gym_id,
            n_envs=args.n_envs,
            stats_path=maybe_stats_path,
            seed=args.seed,
            log_dir=log_dir,
            should_render=not args.no_render,
            hyperparams=hyperparams,
            env_kwargs=env_kwargs,
        )
    except AssertionError as err:
        if "spaces must have the same shape" not in str(err):
            raise
        print("[GRPO] VecNormalize stats shape mismatch, falling back to raw observations.")
        env = create_test_env(
            env_name.gym_id,
            n_envs=args.n_envs,
            stats_path=None,
            seed=args.seed,
            log_dir=log_dir,
            should_render=not args.no_render,
            hyperparams=hyperparams,
            env_kwargs=env_kwargs,
        )

    return model_path, log_path, env, env_name, args


def enjoy() -> None:  # noqa: C901

    model_path, log_path, env, env_name, args = gen_env_model_path()
    # Ensure GRPO checkpoints are written explicitly alongside the PPO logs.
    grpo_filename = f'grpo_model_gs_{trajectories_per_update}_beta_{beta}.zip'
    GRPO_model_path = os.path.join(log_path, grpo_filename)

    if args.eval:
        model = ALGOS['ppo'].load(GRPO_model_path)
        policy = model.policy

        eval_traj = collect_trajectory(env, policy, trajectory_len, deterministic=True)
        print(f"Eval Reward: {eval_traj['return_sum']:.5f}")
        exit()

    # Load the pretrained PPO policy and make sure the weights are numerically safe.
    model = ALGOS['ppo'].load(model_path)
    policy = model.policy
    ensure_finite_parameters(policy)
    for group in policy.optimizer.param_groups:
        group["lr"] = 2e-5
        group["weight_decay"] = 1e-4
    maximum_norm = model.max_grad_norm if model.max_grad_norm is not None else 0.5

    # Clone a reference policy for KL regularisation and freeze its parameters.
    ref_policy = copy.deepcopy(policy)
    ensure_finite_parameters(ref_policy)
    ref_policy.to(device)
    for param in ref_policy.parameters():
        param.requires_grad_(False)


    folder = f'./logs/{env_name.gym_id}'
    log_dir = log_dir_gen(folder, 'GRPO')
    writer = SummaryWriter(log_dir=log_dir)

    target_total_steps = max(1, int(args.grpo_total_timesteps))
    collected_steps = 0
    traj_global_idx = 0
    max_running_reward = float("-inf")
    i_episode = 0
    while collected_steps < target_total_steps:
        trajectories = []
        episode_rewards = []

        # Collect a batch of trajectories and log per-trajectory reward.
        for idx in range(trajectories_per_update):
            if collected_steps >= target_total_steps:
                break
            trajectory = collect_trajectory(env, policy, trajectory_len=trajectory_len)
            trajectories.append(trajectory)
            episode_rewards.append(trajectory["return_sum"])
            collected_steps += len(trajectory["rewards"])
            writer.add_scalar("Episode/Reward", trajectory["return_sum"], traj_global_idx)
            writer.add_scalar("Training/CollectedSteps", collected_steps, traj_global_idx)
            traj_global_idx += 1

        if not trajectories:
            break

        avg_reward = np.mean(episode_rewards)
        max_reward = np.max(episode_rewards)
        normalized_avg = avg_reward * (episode_horizon / trajectory_len)
        normalized_max = max_reward * (episode_horizon / trajectory_len)

        writer.add_scalar("Training/Avg Reward", avg_reward, i_episode)
        writer.add_scalar("Training/Max Reward", max_reward, i_episode)
        writer.add_scalar("Training/Avg Reward normalized", normalized_avg, i_episode)
        writer.add_scalar("Training/Max Reward normalized", normalized_max, i_episode)

        # Track and snapshot the best performing policy.
        if avg_reward > max_running_reward:
            max_running_reward = avg_reward
            model.save(GRPO_model_path)

        grpo_update(
            trajectories,
            policy,
            max_grad_norm=maximum_norm,
            ref_policy=ref_policy,
            writer=writer,
            epi=i_episode,
        )

        print(
            f"Episode {i_episode}, Steps {collected_steps}/{target_total_steps}, "
            f"Avg Reward: {avg_reward:.5f} (normalized {normalized_avg:.5f})"
        )
        print(f"Episode {i_episode}, Max Reward: {max_reward:.5f} (normalized {normalized_max:.5f})")

        if (i_episode + 1) % eval_every == 0:
            eval_mean, eval_std = evaluate_policy(
                model,
                env,
                n_eval_episodes=eval_episodes,
                deterministic=True,
                return_episode_rewards=False,
            )
            writer.add_scalar("Eval/ep_rew_mean", eval_mean, i_episode)
            writer.add_scalar("Eval/ep_rew_std", eval_std, i_episode)
            print(f"[Eval] Episodes {i_episode - eval_every + 2}-{i_episode + 1}, mean reward: {eval_mean:.5f} ± {eval_std:.5f}")
        i_episode += 1

    writer.close()
    env.close()
    # Persist final policy parameters for evaluation convenience.
    if not os.path.isfile(GRPO_model_path):
        model.save(GRPO_model_path)


if __name__ == "__main__":
    enjoy()
