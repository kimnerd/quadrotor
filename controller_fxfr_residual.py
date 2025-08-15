import numpy as np
from typing import Tuple
from simulation import (
    Quadrotor,
    hat,
    so3_log,
    exp_SO3,
    orientation_from_accel,
    vee,
)
from mrgpr_fxfr_residual import ResidualFxGP, ResidualFrGP

try:  # optional dependency guard
    from gpr_inverse import predict_forces
except Exception:  # pragma: no cover
    predict_forces = None


class FxFrResidualQuad(Quadrotor):
    def __init__(
        self,
        gp_fx: ResidualFxGP,
        gp_fr: ResidualFrGP,
        trust_scale_fx: float = 1.0,
        trust_scale_fr: float = 1.0,
        var_clip_fx: float = 5.0,
        var_clip_fr: float = 5.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.gp_fx = gp_fx
        self.gp_fr = gp_fr
        self.trust_scale_fx = trust_scale_fx
        self.trust_scale_fr = trust_scale_fr
        self.var_clip_fx = var_clip_fx
        self.var_clip_fr = var_clip_fr
        self._x_hist: list[np.ndarray] = []
        self._R_hist: list[np.ndarray] = []

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
        # maintain history with current state before computing features
        self._x_hist.append(self.x.copy())
        self._R_hist.append(self.R.copy())
        if len(self._x_hist) > 8:
            self._x_hist = self._x_hist[-8:]
        if len(self._R_hist) > 4:
            self._R_hist = self._R_hist[-4:]

        # (a) synthesize future trajectory and attitudes
        x1, x2, x3, x4_nom, a_cmd = self.f_x(x_ref, v_ref, a_ref)
        R1, R2_nom = self.f_R(a_cmd, yaw_ref)

        # Residual GP correction using past window including current state
        phi = self._phi()
        if phi is not None:
            mu_x, var_x = self.gp_fx.predict(phi.reshape(1, -1))
            w_x = self.trust_scale_fx * (1.0 / (1.0 + var_x[0] / self.var_clip_fx))
            x4 = x4_nom + w_x * mu_x[0]

            mu_r, var_r = self.gp_fr.predict(phi.reshape(1, -1))
            w_r = self.trust_scale_fr * (1.0 / (1.0 + var_r[0] / self.var_clip_fr))
            R2 = R2_nom @ exp_SO3(w_r * mu_r[0])
        else:
            x4 = x4_nom
            R2 = R2_nom

        # (b) compute desired thrusts via block inverse
        m, dt = self.m, self.dt
        g = self.g
        y0 = m / (dt**2) * (x2 - 2 * x1 + self.x) - g
        y1 = m / (dt**2) * (x3 - 2 * x2 + x1) - g
        y2 = m / (dt**2) * (x4 - 2 * x3 + x2) - g

        ez = np.array([0.0, 0.0, 1.0])
        A = np.column_stack((self.R @ ez, R1 @ ez, R2 @ ez))
        Y = np.column_stack((y0, y1, y2))

        lam = self.lam
        condA = float("inf")
        try:
            condA = np.linalg.cond(A)
            if not np.isfinite(condA) or condA > 1e3:
                raise np.linalg.LinAlgError
            Ainv_damped = np.linalg.inv(A.T @ A + (lam**2) * np.eye(3)) @ A.T
            D = Ainv_damped @ Y
            T0, T1, T2 = np.diag(D)
            T = float(np.clip(T0, 0.0, 4.0 * self.max_force))
            off_diag_norm = float(np.linalg.norm(D - np.diag(np.diag(D)), ord="fro"))

            alpha_hat_raw = R1.T @ R2 - self.R.T @ R1
            alpha_hat = 0.5 * (alpha_hat_raw - alpha_hat_raw.T)
            alpha = vee(alpha_hat) / (dt**2)
            M = self.I @ alpha - np.cross(self.I @ self.omega, self.omega)
        except np.linalg.LinAlgError:
            T = float(
                np.clip((self.m * a_cmd - g) @ (self.R @ ez), 0.0, 4.0 * self.max_force)
            )
            R_ref = orientation_from_accel(a_cmd, yaw_ref, self.m, self.g)
            xi = so3_log(self.R.T @ R_ref)
            self.e_R_int = self.leak_R * self.e_R_int + self.dt * xi
            self.e_R_int = np.clip(self.e_R_int, -1.0, 1.0)
            M = self.k_Rp * xi - self.k_Rd * self.omega + self.k_Ri * self.e_R_int
            alpha = np.zeros(3)
            off_diag_norm = float("nan")

        if self.use_gpr and self.gpr_model is not None and predict_forces is not None:
            feat = np.concatenate(([T], M))
            forces = predict_forces(self.gpr_model, feat.reshape(1, -1))[0]
            forces = np.clip(forces, 0.0, self.max_force)
            A_alloc = np.array(
                [
                    [1, 1, 1, 1],
                    [0, -self.l, 0, self.l],
                    [self.l, 0, -self.l, 0],
                    [-self.c_t, self.c_t, -self.c_t, self.c_t],
                ]
            )
            TM_act = A_alloc @ forces
            T_act = float(TM_act[0])
            M_act = TM_act[1:]
        else:
            forces, T_act, M_act = self.rotor_forces(T, M)

        # Update translational dynamics with actual thrust
        self.x += dt * self.v
        self.v += dt * (self.g + (self.R @ ez * T_act) / self.m)

        # Update rotational dynamics with actual moments
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
            off_diag_norm,
        )
