"""LEGACY path: directly correcting slot outcomes (Δx4/Δξ2).
Paper-aligned controller is control/controller_mrgpr_slots.py using
reference-side correction [ζ1; s] → s+μ → nominal inverse."""

import numpy as np
from typing import Tuple
from simulation import hat, so3_log, exp_SO3
from plants_ab import RealQuadrotor
from controller_C_nominal import NominalInverseController
from residual_models import ResidualFxGP, ResidualFrGP

try:  # optional dependency guard
    from gpr_inverse import predict_forces
except Exception:  # pragma: no cover
    predict_forces = None


class SlotCorrectedController(RealQuadrotor):
    def __init__(
        self,
        gp_fx: ResidualFxGP | None,
        gp_fr: ResidualFrGP | None,
        trust_scale_fx: float = 1.0,
        trust_scale_fr: float = 1.0,
        var_clip_fx: float = 1e6,
        var_clip_fr: float = 1e6,
        debug_residual: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.gp_fx = gp_fx
        self.gp_fr = gp_fr
        self.trust_scale_fx = trust_scale_fx
        self.trust_scale_fr = trust_scale_fr
        self.var_clip_fx = var_clip_fx
        self.var_clip_fr = var_clip_fr
        self.debug_residual = debug_residual
        self.ctrl = NominalInverseController()
        self._x_hist: list[np.ndarray] = []
        self._R_hist: list[np.ndarray] = []
        self.u_log: list[np.ndarray] = []
        self.last_u: np.ndarray | None = None

    def _phi(self) -> np.ndarray | None:
        if len(self._x_hist) < 4 or len(self._R_hist) < 2:
            return None
        x_hist = np.array(self._x_hist[-4:]).reshape(-1)
        logR_tm1 = so3_log(self._R_hist[-2])
        logR_t = so3_log(self._R_hist[-1])
        return np.concatenate([x_hist, logR_tm1, logR_t])

    def step(
        self,
        x_ref: np.ndarray,
        v_ref: np.ndarray,
        a_ref: np.ndarray,
        yaw_ref: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
        self._x_hist.append(self.x.copy())
        self._R_hist.append(self.R.copy())
        if len(self._x_hist) > 8:
            self._x_hist = self._x_hist[-8:]
        if len(self._R_hist) > 4:
            self._R_hist = self._R_hist[-4:]

        self.ctrl.sync_state(self.x, self.v, self.R, self.omega)
        x1, x2, x3, x4_nom, R1, R2_nom, a_cmd = self.ctrl.rollout_nominal_slots(
            x_ref, v_ref, a_ref, yaw_ref
        )

        phi = self._phi()
        x4 = x4_nom
        R2 = R2_nom
        if phi is not None and self.gp_fx is not None and self.gp_fr is not None:
            mu_x, var_x = self.gp_fx.predict(phi.reshape(1, -1))
            w_x = self.trust_scale_fx * (1.0 / (1.0 + var_x[0] / self.var_clip_fx))
            mu_r, var_r = self.gp_fr.predict(phi.reshape(1, -1))
            w_r = self.trust_scale_fr * (1.0 / (1.0 + var_r[0] / self.var_clip_fr))
            x4 = x4_nom + w_x * mu_x[0]
            R2 = R2_nom @ exp_SO3(w_r * mu_r[0])
            if self.debug_residual:
                print(
                    "||mu_x||", np.linalg.norm(mu_x[0]), "w_x", w_x,
                    "||mu_R||", np.linalg.norm(mu_r[0]), "w_r", w_r,
                )

        T, M, condA, off_diag = self.ctrl.block_inverse(x2, x3, x4, R1, R2)

        if self.use_gpr and self.gpr_model is not None and predict_forces is not None:
            feat = np.concatenate(([T], M))
            forces = predict_forces(self.gpr_model, feat.reshape(1, -1))[0]
            forces = np.clip(forces, 0.0, self.max_force)
            A_alloc = np.array(
                [
                    [1, 1, 1, 1],
                    [0, -self.l, 0, self.l],
                    [self.l, 0, -self.l, 0],
                    [-self.c_t_asym[0], self.c_t_asym[1], -self.c_t_asym[2], self.c_t_asym[3]],
                ]
            )
            TM_act = A_alloc @ forces
            T_act = float(TM_act[0])
            M_act = TM_act[1:]
        else:
            forces, T_act, M_act = self.rotor_forces(T, M)

        self.last_u = np.concatenate(([T_act], M_act)).astype(float)
        self.u_log.append(self.last_u.copy())
        if len(self.u_log) > 20000:
            self.u_log.pop(0)

        dt, m, g = self.dt, self.m, self.g
        ez = np.array([0.0, 0.0, 1.0])
        self.x += dt * self.v
        self.v += dt * (g + (self.R @ ez * T_act) / m - self.k_v * self.v + self.wind)
        self.omega += dt * np.linalg.inv(self.I) @ (
            np.cross(self.I @ self.omega, self.omega) + M_act
        )
        self.R += dt * self.R @ hat(self.omega)
        u, _, vh = np.linalg.svd(self.R)
        self.R = u @ vh

        angle_error = np.arccos(
            np.clip((np.trace(R1.T @ self.R) - 1.0) / 2.0, -1.0, 1.0)
        )
        return (
            forces,
            self.R.copy(),
            x_ref.copy(),
            R1,
            float(angle_error),
            float(condA),
            off_diag,
        )
