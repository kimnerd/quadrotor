import numpy as np
from typing import Tuple
from simulation import (
    so3_log,
    orientation_from_accel,
    exp_SO3,
    vee,
)
from plants_ab import NominalQuadrotor


class NominalInverseController:
    """Inverse controller for the nominal plant (A)."""

    def __init__(self) -> None:
        self.quad = NominalQuadrotor()
        self._last_x1: np.ndarray | None = None
        self._last_a_cmd: np.ndarray | None = None

    def sync_state(
        self, x: np.ndarray, v: np.ndarray, R: np.ndarray, omega: np.ndarray
    ) -> None:
        self.quad.x = x.copy()
        self.quad.v = v.copy()
        self.quad.R = R.copy()
        self.quad.omega = omega.copy()

    def rollout_nominal_slots(
        self,
        x_ref: np.ndarray,
        v_ref: np.ndarray,
        a_ref: np.ndarray,
        yaw_ref: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x1, x2, x3, x4, a_cmd = self.quad.f_x(x_ref, v_ref, a_ref)
        R1, R2 = self.quad.f_R(a_cmd, yaw_ref)
        self._last_x1 = x1
        self._last_a_cmd = a_cmd
        return x1, x2, x3, x4, R1, R2, a_cmd

    def block_inverse(
        self,
        x2: np.ndarray,
        x3: np.ndarray,
        x4_slot: np.ndarray,
        R1: np.ndarray,
        R2_slot: np.ndarray,
    ) -> Tuple[float, np.ndarray, float, float]:
        x1 = self._last_x1
        if x1 is None:
            raise RuntimeError("rollout_nominal_slots must be called before block_inverse")
        m, dt, g = self.quad.m, self.quad.dt, self.quad.g
        y0 = m / (dt**2) * (x2 - 2 * x1 + self.quad.x) - g
        y1 = m / (dt**2) * (x3 - 2 * x2 + x1) - g
        y2 = m / (dt**2) * (x4_slot - 2 * x3 + x2) - g
        ez = np.array([0.0, 0.0, 1.0])
        A = np.column_stack((self.quad.R @ ez, R1 @ ez, R2_slot @ ez))
        Y = np.column_stack((y0, y1, y2))
        lam = self.quad.lam
        condA = float("inf")
        off_diag_norm = float("nan")
        try:
            condA = np.linalg.cond(A)
            if not np.isfinite(condA) or condA > 1e3:
                raise np.linalg.LinAlgError
            Ainv_damped = np.linalg.inv(A.T @ A + (lam**2) * np.eye(3)) @ A.T
            D = Ainv_damped @ Y
            T0, T1, T2 = np.diag(D)
            T = float(np.clip(T0, 0.0, 4.0 * self.quad.max_force))
            off_diag_norm = float(
                np.linalg.norm(D - np.diag(np.diag(D)), ord="fro")
            )
            alpha_hat_raw = R1.T @ R2_slot - self.quad.R.T @ R1
            alpha_hat = 0.5 * (alpha_hat_raw - alpha_hat_raw.T)
            alpha = vee(alpha_hat) / (dt**2)
            M = self.quad.I @ alpha - np.cross(self.quad.I @ self.quad.omega, self.quad.omega)
        except np.linalg.LinAlgError:
            a_cmd = self._last_a_cmd
            T = float(
                np.clip((self.quad.m * a_cmd - g) @ (self.quad.R @ ez), 0.0, 4.0 * self.quad.max_force)
            )
            R_ref = orientation_from_accel(a_cmd, 0.0, self.quad.m, self.quad.g)
            xi = so3_log(self.quad.R.T @ R_ref)
            self.quad.e_R_int = self.quad.leak_R * self.quad.e_R_int + self.quad.dt * xi
            self.quad.e_R_int = np.clip(self.quad.e_R_int, -1.0, 1.0)
            M = (
                self.quad.k_Rp * xi
                - self.quad.k_Rd * self.quad.omega
                + self.quad.k_Ri * self.quad.e_R_int
            )
        return T, M, condA, off_diag_norm
