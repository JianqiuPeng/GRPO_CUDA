"""Field-response channel primitives shared by Blind IA / MISO environments.

This module isolates the physics-inspired antenna response model so that
`custom_envs.MISOenv` (and any future Blind IA variants) can reuse the same
implementation without depending on the older geometry-based MapGenerator.
"""
import math
from typing import Dict, Tuple

import numpy as np
import torch


def set_seed(seed: int = 42):
    """设置所有随机数生成器的种子以确保可重复性"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _direction_cosines(theta: torch.Tensor, phi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direction cosines (xi, eta, zeta) following the field response model."""
    cos_theta = torch.cos(theta)
    # Convert spherical angles (theta, phi) into Cartesian direction cosines
    return (
        cos_theta * torch.cos(phi),
        cos_theta * torch.sin(phi),
        torch.sin(theta),
    )


class FieldResponseChannel:
    """Implements h_k(u_{k,n}) = f_k(u_{k,n})^H Σ_k G_k for fluid antenna systems."""

    def __init__(
        self,
        wavelength: float,
        num_users: int,
        num_rx_paths: int,
        num_tx_paths: int,
        num_tx_antennas: int,
        seed: int = 42  # 添加种子参数
    ) -> None:
        self.wavelength = wavelength
        self.K = num_users
        self.L_r = num_rx_paths
        self.L_t = num_tx_paths
        self.M = num_tx_antennas
        self.seed = seed

        # 设置种子
        set_seed(self.seed)

        # Random angles: theta, phi ∈ [-π/2, π/2] -- used to sample path directions
        self.rx_angles = self._init_angles(self.K, self.L_r)
        self.tx_angles = self._init_angles(self.K, self.L_t)

        # Path-response matrices Σ_k sampled from circularly symmetric complex Gaussian
        #   (each entry stores complex gain between RX and TX paths)
        sigma_real = torch.randn(self.K, self.L_r, self.L_t)
        sigma_imag = torch.randn(self.K, self.L_r, self.L_t)
        self.Sigma = 0.01 * torch.complex(sigma_real, sigma_imag) / math.sqrt(2.0)

    def _init_angles(self, num_entities: int, num_paths: int) -> Dict[str, torch.Tensor]:
        # 使用固定种子生成随机角度
        theta = torch.rand(num_entities, num_paths) * math.pi - math.pi / 2.0
        phi = torch.rand(num_entities, num_paths) * math.pi - math.pi / 2.0
        return {"theta": theta, "phi": phi}

    def compute_frv(self, user_idx: int, position: torch.Tensor) -> torch.Tensor:
        """Compute f_k(u_{k,n}) with shape [L_r, 1]."""
        theta = self.rx_angles["theta"][user_idx]
        phi = self.rx_angles["phi"][user_idx]
        xi, eta, zeta = _direction_cosines(theta, phi)

        x, y, z = position
        coeff = 2.0 * math.pi / self.wavelength
        phase = x * xi + y * eta + z * zeta
        return torch.exp(1j * coeff * phase).unsqueeze(1)

    def compute_frm(self, user_idx: int, tx_positions: torch.Tensor) -> torch.Tensor:
        """Compute G_k = [g_k(v_1), …, g_k(v_M)] with shape [L_t, M]."""
        theta = self.tx_angles["theta"][user_idx]
        phi = self.tx_angles["phi"][user_idx]
        xi, eta, zeta = _direction_cosines(theta, phi)

        coeff = 2.0 * math.pi / self.wavelength
        vectors = []
        for m in range(self.M):
            x, y, z = tx_positions[m]
            phase = x * xi + y * eta + z * zeta
            vectors.append(torch.exp(1j * coeff * phase))

        return torch.stack(vectors, dim=1)

    def compute_channel(
        self,
        user_idx: int,
        rx_position: torch.Tensor,
        tx_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute h_k(u_{k,n}) with shape [1, M]."""
        frv = self.compute_frv(user_idx, rx_position)
        frm = self.compute_frm(user_idx, tx_positions)
        prm = self.Sigma[user_idx]
        return frv.conj().transpose(0, 1) @ prm @ frm


# -------- Optional NumPy helpers for non-PyTorch pipelines -------- #

def numpy_direction_cosines(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    cos_theta = np.cos(theta)
    return np.stack([
        cos_theta * np.cos(phi),
        cos_theta * np.sin(phi),
        np.sin(theta),
    ], axis=0)


def build_numpy_frv(
    rx_position: np.ndarray,
    theta_r: np.ndarray,
    phi_r: np.ndarray,
    wavelength: float,
) -> np.ndarray:
    xi, eta, zeta = numpy_direction_cosines(theta_r, phi_r)
    coeff = 2.0 * np.pi / wavelength
    phase = rx_position @ np.vstack([xi, eta, zeta])
    return np.exp(1j * coeff * phase)


def build_numpy_frm(
    tx_positions: np.ndarray,
    theta_t: np.ndarray,
    phi_t: np.ndarray,
    wavelength: float,
) -> np.ndarray:
    xi, eta, zeta = numpy_direction_cosines(theta_t, phi_t)
    coeff = 2.0 * np.pi / wavelength
    k_vec = np.vstack([xi, eta, zeta])
    phase = tx_positions @ k_vec
    return np.exp(1j * coeff * phase.T)


# 全局种子设置函数，可以在主程序开始时调用
def initialize_deterministic(seed: int = 42):
    """全局初始化函数，确保整个系统的可重复性"""
    set_seed(seed)
    print(f"Random seed set to {seed} for reproducible results")