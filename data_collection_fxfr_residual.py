import numpy as np
from typing import Tuple
from simulation import (
    Quadrotor,
    simulate,
    so3_log,
    generate_structured_trajectory,
)


def _logR(R: np.ndarray) -> np.ndarray:
    return so3_log(R)


def collect_fxfr_residual_data(
    runs: int = 100,
    steps: int = 20,
    hold_steps: int = 0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect training data for residual corrections of f_x and f_R.

    Returns
    -------
    X_fx, Y_fx : ndarray
        Training inputs and targets for Δx4 prediction.
    X_fr, Y_fr : ndarray
        Training inputs and targets for Δξ2 prediction.
    """
    rng = np.random.default_rng(seed)
    X_fx, Y_fx, X_fr, Y_fr = [], [], [], []

    for _ in range(runs):
        quad = Quadrotor()
        target = rng.uniform(0.6, 1.2, size=3)
        pos, _, _, R_hist, _, _, _, _ = simulate(
            steps, target=target, hold_steps=hold_steps, quad=quad
        )

        T = len(pos)
        traj = list(
            generate_structured_trajectory(
                start=pos[0], goal=target, n_steps=T, dt=quad.dt
            )
        )

        q_nom = Quadrotor()
        q_nom.__dict__.update(
            {k: v.copy() if hasattr(v, "copy") else v for k, v in quad.__dict__.items()}
        )

        for t in range(3, T - 4):
            x_hist = pos[t - 3 : t + 1].reshape(-1)
            logR_tm1 = _logR(R_hist[t - 1])
            logR_t = _logR(R_hist[t])
            phi = np.concatenate([x_hist, logR_tm1, logR_t])

            q_nom.x = pos[t].copy()
            q_nom.v = (pos[t] - pos[t - 1]) / q_nom.dt
            q_nom.R = R_hist[t].copy()
            q_nom.omega = _logR(R_hist[t - 1].T @ R_hist[t]) / q_nom.dt

            x_ref_t, v_ref_t, a_ref_t = traj[t]
            x1, x2, x3, x4_nom, a_cmd = q_nom.f_x(x_ref_t, v_ref_t, a_ref_t)
            _, R2_nom = q_nom.f_R(a_cmd=a_cmd, yaw_ref=0.0)

            x_true_p4 = pos[t + 4]
            R_true_p2 = R_hist[t + 2]

            dx4 = x_true_p4 - x4_nom
            dxi2 = _logR(R2_nom.T @ R_true_p2)

            X_fx.append(phi)
            Y_fx.append(dx4)
            X_fr.append(phi)
            Y_fr.append(dxi2)

    return (
        np.array(X_fx),
        np.array(Y_fx),
        np.array(X_fr),
        np.array(Y_fr),
    )
