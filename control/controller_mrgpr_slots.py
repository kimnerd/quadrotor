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
        ignore_var_trust: bool = False,
        cond_aware: bool = False,
        cond_ref: float = 1e3,
        ood_aware: bool = False,
        ood_ref: float = 3.0,
        cap_y_ratio: float = 0.5,
        cap_y_abs: float = 2.0,
        cap_r_ratio: float = 0.5,
        cap_r_abs: float = 0.5,
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
        self.ignore_var_trust = ignore_var_trust
        self.cond_aware = cond_aware
        self.cond_ref = cond_ref
        self.ood_aware = ood_aware
        self.ood_ref = ood_ref
        self.cap_y_ratio = cap_y_ratio
        self.cap_y_abs = cap_y_abs
        self.cap_r_ratio = cap_r_ratio
        self.cap_r_abs = cap_r_abs

        self.x_hist: list[np.ndarray] = []
        self.R_hist: list[np.ndarray] = []
        self.u_hist: list[np.ndarray] = []
        self.u_log: list[np.ndarray] = []
        self.last_u: np.ndarray | None = None
        self.debug = bool(kwargs.pop("debug", False))
        self.applied_stats = {
            "dy_norm": [],
            "wy_l2": [],
            "wy_mean": [],
            "dxi_norm": [],
            "wr_l2": [],
            "wr_mean": [],
            "vy_mean": [],
            "vr_mean": [],
            "cond_pre": [],
        }
        self.step_total = 0
        self.feature_steps = 0
        self.applied_steps = 0

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
        self.step_total += 1
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
            self.feature_steps += 1
            Xy_run = np.concatenate([z1, y_nom])
            xi2_nom = so3_log(R1.T @ R2_nom)
            xr_run = np.concatenate([z1, xi2_nom])

            ez = np.array([0.0, 0.0, 1.0])
            A_pre = np.column_stack((self.R @ ez, R1 @ ez, R2_nom @ ez))
            cond_pre = float(np.linalg.cond(A_pre))
            if not np.isfinite(cond_pre):
                cond_pre = 1e12

            dy, vy = self.gp_y.predict(Xy_run[None, :])
            dxi, vr = self.gp_r.predict(xr_run[None, :])

            if self.ignore_var_trust:
                wy = self.ts_y
                wr = self.ts_r
            else:
                wy = self.ts_y * (1.0 / (1.0 + vy[0] / self.clip_y))
                wr = self.ts_r * (1.0 / (1.0 + vr[0] / self.clip_r))

            cond_scale = 1.0
            if self.cond_aware:
                c = max(0.0, cond_pre - self.cond_ref) / max(self.cond_ref, 1e-9)
                cond_scale = 1.0 / (1.0 + c)

            ood_scale_y = ood_scale_r = 1.0
            if self.ood_aware:
                z_y = self.gp_y.scaler.transform(Xy_run[None, :])
                z_r = self.gp_r.scaler.transform(xr_run[None, :])
                dz_y = float(np.mean(np.abs(z_y)))
                dz_r = float(np.mean(np.abs(z_r)))

                def _ood_s(d: float, ref: float) -> float:
                    k = max(0.0, d - ref) / max(ref, 1e-9)
                    return 1.0 / (1.0 + k)

                ood_scale_y = _ood_s(dz_y, self.ood_ref)
                ood_scale_r = _ood_s(dz_r, self.ood_ref)

            wy = wy * cond_scale * ood_scale_y
            wr = wr * cond_scale * ood_scale_r

            dy_apply = wy * dy[0]
            y0c, y1c, y2c = dy_apply[0:3], dy_apply[3:6], dy_apply[6:9]

            def _cap_vec(vec: np.ndarray, ref_norm: float, ratio: float, abs_cap: float) -> np.ndarray:
                n = float(np.linalg.norm(vec))
                lim = min(ratio * max(ref_norm, 1e-9), abs_cap)
                if n > lim and n > 0:
                    vec = vec * (lim / n)
                return vec

            y0c = _cap_vec(y0c, np.linalg.norm(y0_nom), self.cap_y_ratio, self.cap_y_abs)
            y1c = _cap_vec(y1c, np.linalg.norm(y1_nom), self.cap_y_ratio, self.cap_y_abs)
            y2c = _cap_vec(y2c, np.linalg.norm(y2_nom), self.cap_y_ratio, self.cap_y_abs)
            dy_apply = np.concatenate([y0c, y1c, y2c])
            y_corr_flat = y_nom + dy_apply
            y0_corr, y1_corr, y2_corr = y_corr_flat[0:3], y_corr_flat[3:6], y_corr_flat[6:9]

            dxi_apply = (wr * dxi)[0]
            dxi_apply = _cap_vec(dxi_apply, np.linalg.norm(xi2_nom), self.cap_r_ratio, self.cap_r_abs)

            R2 = R2_nom @ exp_SO3(dxi_apply)
            Y_override = np.column_stack((y0_corr, y1_corr, y2_corr))
            T, M, condA, off_diag = self.ctrl.block_inverse(
                x2, x3, x4_nom, R1, R2, Y_override=Y_override
            )
            applied = (self.ts_y > 0.0) or (self.ts_r > 0.0)
            if applied:
                self.applied_steps += 1
                self.applied_stats["dy_norm"].append(float(np.linalg.norm(dy[0])))
                self.applied_stats["wy_l2"].append(float(np.linalg.norm(wy)))
                self.applied_stats["wy_mean"].append(float(np.mean(wy)))
                self.applied_stats["dxi_norm"].append(float(np.linalg.norm(dxi[0])))
                self.applied_stats["wr_l2"].append(float(np.linalg.norm(wr)))
                self.applied_stats["wr_mean"].append(float(np.mean(wr)))
                self.applied_stats["vy_mean"].append(float(np.mean(vy)))
                self.applied_stats["vr_mean"].append(float(np.mean(vr)))
                self.applied_stats["cond_pre"].append(cond_pre)
                if self.debug and len(self.applied_stats["dy_norm"]) % 50 == 0:
                    print(
                        "[MRGP] mean||dy||=%.3e  mean||wy||=%.3e  mean wy=%.3e  mean||dxi||=%.3e  mean||wr||=%.3e  mean wr=%.3e"
                        % (
                            np.mean(self.applied_stats["dy_norm"]),
                            np.mean(self.applied_stats["wy_l2"]),
                            np.mean(self.applied_stats["wy_mean"]),
                            np.mean(self.applied_stats["dxi_norm"]),
                            np.mean(self.applied_stats["wr_l2"]),
                            np.mean(self.applied_stats["wr_mean"]),
                        )
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

    def get_debug_stats(self) -> dict:
        total = max(self.step_total, 1)
        feature_ratio = self.feature_steps / total
        applied_ratio = self.applied_steps / total
        mean = lambda lst: float(np.mean(lst)) if lst else 0.0
        cond_list = self.applied_stats["cond_pre"]
        if cond_list:
            cp = np.percentile(cond_list, [50, 90, 99])
        else:
            cp = [0.0, 0.0, 0.0]
        return {
            "applied_steps": self.applied_steps,
            "applied_ratio": applied_ratio,
            "mean_norm_dy": mean(self.applied_stats["dy_norm"]),
            "mean_norm_wy": mean(self.applied_stats["wy_l2"]),
            "mean_wy_mean": mean(self.applied_stats["wy_mean"]),
            "mean_vy": mean(self.applied_stats["vy_mean"]),
            "mean_norm_dxi": mean(self.applied_stats["dxi_norm"]),
            "mean_norm_wr": mean(self.applied_stats["wr_l2"]),
            "mean_wr_mean": mean(self.applied_stats["wr_mean"]),
            "mean_vr": mean(self.applied_stats["vr_mean"]),
            "z1_ratio": feature_ratio,
            "cond_pre_p50": float(cp[0]),
            "cond_pre_p90": float(cp[1]),
            "cond_pre_p99": float(cp[2]),
        }
