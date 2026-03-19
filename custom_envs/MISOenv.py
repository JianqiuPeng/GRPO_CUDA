"""Fluid-antenna MISO environment used for Blind IA optimisation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from .field_response_channel import FieldResponseChannel


@dataclass
class RegionConfig:
    """Cubic feasible region for fluid antennas."""

    center: np.ndarray
    size: float

    @property
    def lower(self) -> np.ndarray:
        return self.center - self.size / 2.0

    @property
    def upper(self) -> np.ndarray:
        return self.center + self.size / 2.0


class MISOEnv:
    """Core environment without Gym wrapper."""

    def __init__(
        self,
        num_users: int = 1,
        wavelength: float = 0.005,
        num_rx_paths: int = 4,
        num_tx_paths: int = 4,
        tx_positions: Optional[np.ndarray] = None,
        region: Optional[RegionConfig] = None,
        min_distance: float = 0.0025,
        tx_power: float = 0.25,
        noise_power: float = 1e-9,
        seed: int = 42,
        max_steps: int = 50,
        step_scale: Optional[float] = 0.25,
        error_covariance: Optional[np.ndarray] = None,
        regenerate_each_step: bool = False,
    ) -> None:
        # --- Physical configuration -------------------------------------------------------- #
        # Core scenario dimensions: number of users, BIA requires two candidate positions
        # per user, and basic channel/antenna parameters.
        self.num_users = num_users
        self.positions_per_user = 2
        self.wavelength = wavelength
        self.num_rx_paths = num_rx_paths
        self.num_tx_paths = num_tx_paths
        self.min_distance = min_distance
        self.tx_power = tx_power
        self.noise_power = noise_power

        # RNG drives initial placement and any stochastic sampling inside the environment.
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.regenerate_each_step = regenerate_each_step

        # Region definitions: if none provided fall back to a centred cube. The step scale
        # controls how raw actions translate to physical displacements.
        self.region = region or RegionConfig(center=np.ones(3) * 3, size=0.5)
        self.step_scale = step_scale if step_scale is not None else 0.05 * self.region.size

        # Base station antenna coordinates. Default: two-element ULA along x-axis.
        tx_positions = (
            np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32)
            if tx_positions is None
            else tx_positions
        )
        self.tx_positions = torch.as_tensor(tx_positions, dtype=torch.float32)

        # FieldResponseChannel encapsulates the deterministic propagation model that maps
        # antenna locations to complex-valued channel vectors.
        self.channel_model = FieldResponseChannel(
            wavelength=self.wavelength,
            num_users=self.num_users,
            num_rx_paths=self.num_rx_paths,
            num_tx_paths=self.num_tx_paths,
            num_tx_antennas=self.tx_positions.shape[0],
            seed=seed,
        )

        # Buffers updated on every reset/step call.
        self.current_positions: Optional[np.ndarray] = None
        self.current_channels: Optional[np.ndarray] = None  # actual channels with estimation errors
        self.current_estimated: Optional[np.ndarray] = None  # nominal (estimated) channels
        self.channel_errors: Optional[np.ndarray] = None
        self.t = 0
        base_cov = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64) * 0.1 * (10.0 ** -2)
        chosen_cov = error_covariance if error_covariance is not None else base_cov
        self.error_covariance = np.asarray(chosen_cov, dtype=np.float64)
        self.best_sum_rate: float = float("-inf")
        self.penalty_accumulator: float = 0.0

    def _clip_to_region(self, positions: np.ndarray) -> Tuple[np.ndarray, float]:
        """Project candidate positions onto the feasible cube and accumulate violation cost."""
        lower, upper = self.region.lower, self.region.upper
        clipped = np.clip(positions, lower, upper)
        return clipped, float(np.abs(positions - clipped).sum())

    def _check_min_distance(self, positions: np.ndarray) -> float:
        """Additive penalty when the two user positions fall below the minimum separation."""
        penalty = 0.0
        for pts in positions:
            dist = np.linalg.norm(pts[0] - pts[1])
            if dist < self.min_distance:
                penalty += self.min_distance - dist
        return penalty

    def _compute_estimated_channels(self, positions: np.ndarray) -> np.ndarray:
        """Evaluate FieldResponseChannel for every (user, position) pair (nominal channels)."""
        return np.array(
            [
                [
                    self.channel_model.compute_channel(
                        user_idx,
                        rx_position=torch.tensor(pos, dtype=torch.float32),
                        tx_positions=self.tx_positions,
                    )
                    .numpy()
                    .ravel()
                    for pos in positions[user_idx]
                ]
                for user_idx in range(self.num_users)
            ],
            dtype=np.complex128,
        )

    def _apply_channel_uncertainty(self, estimated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sample channel estimation errors and construct actual channels."""
        cov = self.error_covariance
        if cov.ndim == 2:
            cov_broadcast = np.broadcast_to(cov, (self.num_users, self.positions_per_user, 2, 2))
        elif cov.ndim == 4:
            cov_broadcast = cov
        else:
            raise ValueError("error_covariance must have shape (2,2) or (num_users, positions, 2,2).")

        errors = np.zeros_like(estimated, dtype=np.complex128)
        for user_idx in range(self.num_users):
            for pos_idx in range(self.positions_per_user):
                cov_matrix = cov_broadcast[user_idx, pos_idx]
                real_part = self.rng.multivariate_normal(np.zeros(2), 0.5 * cov_matrix)
                imag_part = self.rng.multivariate_normal(np.zeros(2), 0.5 * cov_matrix)
                errors[user_idx, pos_idx] = real_part + 1j * imag_part
        actual = estimated + errors
        return actual, errors

    def _robust_sum_rate(self, estimated: np.ndarray, errors: np.ndarray) -> float:
        """Robust sum-rate objective incorporating estimation errors."""
        identity = np.eye(2, dtype=np.complex128)
        total_rate = 0.0
        for user_idx in range(self.num_users):
            tilde_h1 = estimated[user_idx, 0].reshape(1, -1)
            tilde_h2 = estimated[user_idx, 1].reshape(1, -1)
            tilde_H = np.vstack([tilde_h1, tilde_h2])

            delta_h1 = errors[user_idx, 0].reshape(1, -1)
            delta_h2 = errors[user_idx, 1].reshape(1, -1)
            delta_matrix = np.vstack([delta_h1, delta_h2])

            noise_cov = np.array(
                [[4 * self.noise_power, 0.0], [0.0, self.noise_power]], dtype=np.complex128
            )
            error_cov = self.tx_power * (delta_matrix @ delta_matrix.conj().T)
            robust_cov = error_cov + noise_cov + 1e-12 * identity
            inv_cov = np.linalg.inv(robust_cov)
            gram = identity + self.tx_power * (tilde_H @ tilde_H.conj().T @ inv_cov)
            rate = np.log2(np.linalg.det(gram)).real
            total_rate += float(max(rate, 0.0))
        return total_rate

    def _nominal_sum_rate(self, estimated: np.ndarray) -> float:
        """Nominal sum-rate using estimated channels only (no estimation error)."""
        identity = np.eye(2, dtype=np.complex128)
        total_rate = 0.0
        for user_idx in range(self.num_users):
            tilde_h1 = estimated[user_idx, 0].reshape(1, -1)
            tilde_h2 = estimated[user_idx, 1].reshape(1, -1)
            tilde_H = np.vstack([tilde_h1, tilde_h2])
            noise_cov = np.array(
                [[self.num_users * self.noise_power, 0.0], [0.0, self.noise_power]], dtype=np.complex128
            )
            inv_cov = np.linalg.inv(noise_cov + 1e-12 * identity)
            gram = identity + self.tx_power * (tilde_H @ tilde_H.conj().T @ inv_cov)
            rate = np.log2(np.linalg.det(gram)).real
            total_rate += float(max(rate, 0.0))
        return total_rate

    def reset(self) -> Dict[str, np.ndarray]:
        """Initialise state, sample starting antenna locations, and build observation."""
        self.t = 0
        self.best_sum_rate = float("-inf")
        self.penalty_accumulator = 0.0
        self.current_positions = self._sample_initial_positions()
        estimated = self._compute_estimated_channels(self.current_positions)
        actual, errors = self._apply_channel_uncertainty(estimated)
        self.current_estimated = estimated
        self.current_channels = actual
        self.channel_errors = errors
        obs = self._build_observation(estimated, actual, errors, self.current_positions)
        sum_rate = self._robust_sum_rate(estimated, errors)
        self.best_sum_rate = max(self.best_sum_rate, sum_rate)
        return {
            "observation": obs,
            "positions": self.current_positions.copy(),
            "estimated_channels": self.current_estimated.copy(),
            "channels": self.current_channels.copy(),
            "channel_errors": self.channel_errors.copy(),
        }

    def step(self, delta_positions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Translate scaled action into movement, update channels, and report reward."""
        assert delta_positions.shape == (self.num_users, self.positions_per_user, 3)

        proposed = self.current_positions + delta_positions
        clipped, boundary_penalty = self._clip_to_region(proposed)
        distance_penalty = self._check_min_distance(clipped)

        self.current_positions = clipped
        estimated = self._compute_estimated_channels(clipped)
        actual, errors = self._apply_channel_uncertainty(estimated)
        self.current_estimated = estimated
        self.current_channels = actual
        self.channel_errors = errors

        sum_rate = self._robust_sum_rate(estimated, errors)
        self.best_sum_rate = max(self.best_sum_rate, sum_rate)
        step_penalty = boundary_penalty + distance_penalty
        self.penalty_accumulator += step_penalty
        # Dense reward: per-step signal normalized by episode length for stable scale.
        reward = (sum_rate - 10.0 * step_penalty) / float(self.max_steps)

        if self.regenerate_each_step:
            self.current_channels, self.channel_errors = self._apply_channel_uncertainty(self.current_estimated)

        obs = self._build_observation(self.current_estimated, self.current_channels, self.channel_errors, self.current_positions)
        self.t += 1
        done = self.t >= self.max_steps
        info = {
            "sum_rate": sum_rate,
            "best_sum_rate": self.best_sum_rate,
            "boundary_penalty": boundary_penalty,
            "distance_penalty": distance_penalty,
            "positions": clipped.copy(),
            "estimated_channels": self.current_estimated.copy(),
            "channels": self.current_channels.copy(),
            "channel_errors": self.channel_errors.copy(),
            "step": self.t,
        }
        return obs, reward, done, info

    def _sample_initial_positions(self) -> np.ndarray:
        """Uniformly draw user positions while satisfying the minimum distance constraint."""
        lower, upper = self.region.lower, self.region.upper
        positions = self.rng.uniform(lower, upper, size=(self.num_users, self.positions_per_user, 3))
        for idx in range(self.num_users):
            diff = positions[idx, 0] - positions[idx, 1]
            while np.linalg.norm(diff) < self.min_distance:
                positions[idx, 1] = self.rng.uniform(lower, upper, size=3)
                diff = positions[idx, 0] - positions[idx, 1]
        return positions

    def _build_observation(
        self,
        estimated: np.ndarray,
        actual: np.ndarray,
        errors: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray:
        """Flatten nominal/actual channel components, errors, and positions into a single vector."""
        est_features = np.concatenate([estimated.real, estimated.imag], axis=2).reshape(self.num_users, -1)
        act_features = np.concatenate([actual.real, actual.imag], axis=2).reshape(self.num_users, -1)
        err_features = np.concatenate([errors.real, errors.imag], axis=2).reshape(self.num_users, -1)
        normalized = ((positions - self.region.center) / (self.region.size / 2.0)).reshape(self.num_users, -1)
        concatenated = np.concatenate([est_features, act_features, err_features, normalized], axis=1)
        return concatenated.ravel().astype(np.float32)

    def evaluate_positions(self, positions: np.ndarray) -> dict[str, np.ndarray | float]:
        """Utility helper for heuristics: evaluate robust sum-rate at arbitrary positions."""
        estimated = self._compute_estimated_channels(positions)
        actual, errors = self._apply_channel_uncertainty(estimated)
        sum_rate = self._robust_sum_rate(estimated, errors)
        return {
            "sum_rate": float(sum_rate),
            "estimated": estimated,
            "actual": actual,
            "errors": errors,
        }

    def evaluate_positions_nominal(self, positions: np.ndarray) -> dict[str, np.ndarray | float]:
        """Evaluate nominal (estimated) sum-rate at arbitrary positions without sampling errors."""
        estimated = self._compute_estimated_channels(positions)
        sum_rate = self._nominal_sum_rate(estimated)
        return {
            "sum_rate": float(sum_rate),
            "estimated": estimated,
        }


class MISOEnvWrapper(gym.Env):
    """Gymnasium wrapper exposing vectorised Box spaces."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: Optional[str] = None, **kwargs):
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode '{render_mode}'. Supported: {self.metadata['render_modes']}")
        self.render_mode = render_mode
        # Gather kwargs so downstream code can pass arbitrary overrides without breaking.
        params = {
            "num_users": kwargs.get("num_users", 1),
            "wavelength": kwargs.get("wavelength", 0.005),
            "num_rx_paths": kwargs.get("num_rx_paths", 4),
            "num_tx_paths": kwargs.get("num_tx_paths", 4),
            "tx_positions": kwargs.get("tx_positions"),
            "region": kwargs.get("region", RegionConfig(center=np.ones(3) * 3, size=0.5)),
            "min_distance": kwargs.get("min_distance", 0.0025),
            "tx_power": kwargs.get("tx_power", 0.25),
            "noise_power": kwargs.get("noise_power", 1e-9),
            "seed": kwargs.get("seed", 42),
            "max_steps": kwargs.get("max_steps", 50),
            "step_scale": kwargs.get("step_scale", 0.25),
            "error_covariance": kwargs.get("error_covariance"),
            "regenerate_each_step": kwargs.get("regenerate_each_step", False),
        }
        self.env = MISOEnv(**params)

        # Mirror a few convenience attributes for downstream agents.
        self.num_users = self.env.num_users
        self.positions_per_user = self.env.positions_per_user

        # Action comprises x/y/z offsets for each of the fluid antenna positions.
        action_dim = self.num_users * self.positions_per_user * 3
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        # Observation dimension is derived from the underlying environment helper.
        tx_antennas = self.env.tx_positions.shape[0]
        dummy_estimated = np.zeros((self.num_users, self.positions_per_user, tx_antennas), dtype=np.complex128)
        dummy_actual = np.zeros_like(dummy_estimated)
        dummy_errors = np.zeros_like(dummy_estimated)
        dummy_positions = np.zeros((self.num_users, self.positions_per_user, 3), dtype=np.float32)
        obs_dim = self.env._build_observation(dummy_estimated, dummy_actual, dummy_errors, dummy_positions).shape[0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Bridge Gymnasium reset signature to the underlying environment."""
        if seed is not None:
            self.env.rng = np.random.default_rng(seed)
        obs = self.env.reset()
        return obs["observation"], {}

    def step(self, action: np.ndarray):
        """Map flat action vector to 3-D displacements before calling the core env."""
        reshaped = action.astype(np.float32).reshape(self.num_users, self.positions_per_user, 3)
        obs, reward, done, info = self.env.step(reshaped * self.env.step_scale)
        return obs, reward, done, False, info

    def render(self):
        if self.render_mode == "human":
            # Environment does not support visualisation; comply with Gymnasium by returning None.
            return None
        raise RuntimeError("Environment was not initialised with a supported render_mode.")

    def close(self):
        pass
