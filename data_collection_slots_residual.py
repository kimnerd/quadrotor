import numpy as np
from typing import Tuple
from simulation import simulate, so3_log, generate_structured_trajectory
from plants_ab import RealQuadrotor
from controller_C_nominal import NominalInverseController


def _logR(R: np.ndarray) -> np.ndarray:
    return so3_log(R)


def collect_slots_residual_data(
    runs: int = 20,
    steps: int = 250,
    hold_steps: int = 50,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect residual training data using plant B and nominal controller C.
    Returns X_fx, Y_fx, X_fr, Y_fr.
    """
    rng = np.random.default_rng(seed)
    X_fx, Y_fx, X_fr, Y_fr = [], [], [], []

    for _ in range(runs):
        plantB = RealQuadrotor()
        target = rng.uniform(0.6, 1.2, size=3)
        pos, _, _, R_hist, _, _, _, _ = simulate(
            steps, target=target, hold_steps=hold_steps, quad=plantB
        )
        T = len(pos)
        traj = list(
            generate_structured_trajectory(start=pos[0], goal=target, n_steps=steps, dt=plantB.dt)
        )
        if traj:
            last = traj[-1]
        else:
            last = (target, np.zeros(3), np.zeros(3))
        traj.extend([last] * hold_steps)

        ctrlC = NominalInverseController()
        for t in range(3, T - 4):
            x_hist = pos[t - 3 : t + 1].reshape(-1)
            logR_tm1 = _logR(R_hist[t - 1])
            logR_t = _logR(R_hist[t])
            phi = np.concatenate([x_hist, logR_tm1, logR_t])

            x_ref_t, v_ref_t, a_ref_t = traj[t]
            ctrlC.sync_state(
                x=pos[t],
                v=(pos[t] - pos[t - 1]) / plantB.dt,
                R=R_hist[t],
                omega=_logR(R_hist[t - 1].T @ R_hist[t]) / plantB.dt,
            )
            x1, x2, x3, x4_nom, R1, R2_nom, a_cmd = ctrlC.rollout_nominal_slots(
                x_ref_t, v_ref_t, a_ref_t
            )

            dx4 = pos[t + 4] - x4_nom
            dxi2 = _logR(R2_nom.T @ R_hist[t + 2])

            X_fx.append(phi)
            Y_fx.append(dx4)
            X_fr.append(phi)
            Y_fr.append(dxi2)

    return np.array(X_fx), np.array(Y_fx), np.array(X_fr), np.array(Y_fr)
