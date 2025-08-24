"""MR-GPR controller with reference-side slot corrections."""

from __future__ import annotations

import numpy as np
from typing import Tuple

from controller_C_nominal import NominalInverseController
from models.residual_slots_gp import ResidualYSlotsGP, ResidualR2GP
from plants_ab import RealQuadrotor
from simulation import exp_SO3, finite_diff_y, hat, so3_log


class MRGPBlockController(RealQuadrotor):
    def __init__(
        self,
        gp_y: ResidualYSlotsGP,
        gp_r: ResidualR2GP,
        trust_scale_y: float = 1.0,
        trust_scale_r: float = 1.0,
        var_clip_y: float = 1e6,
        var_clip_r: float = 1e6,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.ctrl = NominalInverseController()
        self.gp_y = gp_y
        self.gp_r = gp_r
        self.ts_y = trust_scale_y
        self.ts_r = trust_scale_r
        self.clip_y = var_clip_y
        self.clip_r = var_clip_r
        self.x_hist: list[np.ndarray] = []
        self.R_hist: list[np.ndarray] = []
        self.u_hist: list[np.ndarray] = []
        self.u_log: list[np.ndarray] = []
        self.last_u: np.ndarray | None = None

    def _zeta1(self) -> np.ndarray | None:
        if len(self.x_hist) < 4 or len(self.R_hist) < 2 or len(self.u_hist) < 3:
            return None
        x_part = np.array(self.x_hist[-4:]).reshape(-1)
        logR_tm1 = so3_log(self.R_hist[-2])
        logR_t = so3_log(self.R_hist[-1])
        u_part = np.array(self.u_hist[-3:]).reshape(-1)
        return np.concatenate([x_part, logR_tm1, logR_t, u_part])

    def step(
        self,
        x_ref: np.ndarray,
        v_ref: np.ndarray,
        a_ref: np.ndarray,
        yaw_ref: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
        # history update
        self.x_hist.append(self.x.copy())
        self.R_hist.append(self.R.copy())
        if self.last_u is not None:
            self.u_hist.append(self.last_u.copy())
        if len(self.x_hist) > 4:
            self.x_hist = self.x_hist[-4:]
        if len(self.R_hist) > 2:
            self.R_hist = self.R_hist[-2:]
        if len(self.u_hist) > 3:
            self.u_hist = self.u_hist[-3:]

        # nominal rollout
        self.ctrl.sync_state(self.x, self.v, self.R, self.omega)
        x1, x2, x3, x4_nom, R1, R2_nom, a_cmd = self.ctrl.rollout_nominal_slots(
            x_ref, v_ref, a_ref, yaw_ref
        )

        y0_nom = finite_diff_y(self.m, self.dt, self.g, self.x, x1, x2)
        y1_nom = finite_diff_y(self.m, self.dt, self.g, x1, x2, x3)
        y2_nom = finite_diff_y(self.m, self.dt, self.g, x2, x3, x4_nom)
        y_nom = np.concatenate([y0_nom, y1_nom, y2_nom])

        z1 = self._zeta1()
        if z1 is None:
            T, M, condA, off_diag = self.ctrl.block_inverse(x2, x3, x4_nom, R1, R2_nom)
        else:
            Xy_run = np.concatenate([z1, y_nom])
            xi2_nom = so3_log(R1.T @ R2_nom)
            xr_run = np.concatenate([z1, xi2_nom])
            dy, vy = self.gp_y.predict(Xy_run[None, :])
            wy = self.ts_y * (1.0 / (1.0 + vy[0] / self.clip_y))
            dxi, vr = self.gp_r.predict(xr_run[None, :])
            wr = self.ts_r * (1.0 / (1.0 + vr[0] / self.clip_r))
            y_corr_flat = y_nom + wy * dy[0]
            y0_corr = y_corr_flat[0:3]
            y1_corr = y_corr_flat[3:6]
            y2_corr = y_corr_flat[6:9]
            R2 = R2_nom @ exp_SO3((wr * dxi)[0])
            Y_override = np.column_stack((y0_corr, y1_corr, y2_corr))
            T, M, condA, off_diag = self.ctrl.block_inverse(
                x2, x3, x4_nom, R1, R2, Y_override=Y_override
            )

        forces, T_act, M_act = self.rotor_forces(T, M)

        self.last_u = np.concatenate(([T_act], M_act)).astype(float)
        self.u_log.append(self.last_u.copy())
        if len(self.u_log) > 20000:
            self.u_log.pop(0)

        dt = self.dt
        ez = np.array([0.0, 0.0, 1.0])
        self.x += dt * self.v
        self.v += dt * (self.g + (self.R @ ez * T_act) / self.m - self.k_v * self.v + self.wind)
        self.omega += dt * np.linalg.inv(self.I) @ (
            np.cross(self.I @ self.omega, self.omega) + M_act
        )
        self.R += dt * self.R @ hat(self.omega)
        u, _, vh = np.linalg.svd(self.R)
        self.R = u @ vh

        angle_error = np.arccos(np.clip((np.trace(R1.T @ self.R) - 1.0) / 2.0, -1.0, 1.0))
        return (
            forces,
            self.R.copy(),
            x_ref.copy(),
            R1,
            float(angle_error),
            float(condA),
            off_diag,
        )

