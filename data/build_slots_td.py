import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np

from controller_C_nominal import NominalInverseController
from controller_D_slot_corrected import SlotCorrectedController
from plants_ab import RealQuadrotor
from simulation import generate_structured_trajectory, so3_log, finite_diff_y, simulate


def _zeta1(
    x_hist: np.ndarray, logR_tm1: np.ndarray, logR_t: np.ndarray, u_hist: np.ndarray
) -> np.ndarray:
    assert x_hist.shape == (12,)
    assert logR_tm1.shape == (3,)
    assert logR_t.shape == (3,)
    assert u_hist.shape == (12,)
    return np.concatenate([x_hist, logR_tm1, logR_t, u_hist])


def build_slots_td(
    runs: int = 20,
    steps: int = 250,
    hold_steps: int = 50,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X_y: list[np.ndarray] = []
    Y_y: list[np.ndarray] = []
    X_r: list[np.ndarray] = []
    Y_r: list[np.ndarray] = []

    for _ in range(runs):
        quad = SlotCorrectedController(gp_fx=None, gp_fr=None)
        target = rng.uniform(0.6, 1.2, size=3)
        pos, _, _, R_hist, _, _, _, _ = simulate(
            steps, target=target, hold_steps=hold_steps, quad=quad
        )
        u_arr = np.array(quad.u_log)
        x_full = np.vstack([np.zeros(3), pos])
        R_full = np.concatenate([np.eye(3)[None, :, :], R_hist], axis=0)
        T_sim = len(pos)
        traj = list(
            generate_structured_trajectory(
                start=x_full[0], goal=target, n_steps=steps, dt=quad.dt
            )
        )
        if traj:
            last = traj[-1]
        else:
            last = (target, np.zeros(3), np.zeros(3))
        traj.extend([last] * hold_steps)

        ctrl = NominalInverseController()
        for t in range(3, T_sim - 4):
            x_hist = x_full[t - 3 : t + 1].reshape(-1)
            logR_tm1 = so3_log(R_full[t - 1])
            logR_t = so3_log(R_full[t])
            u_hist = u_arr[t - 3 : t].reshape(-1)
            z1 = _zeta1(x_hist, logR_tm1, logR_t, u_hist)

            v = (x_full[t] - x_full[t - 1]) / quad.dt
            omega = so3_log(R_full[t - 1].T @ R_full[t]) / quad.dt
            ctrl.sync_state(x_full[t], v, R_full[t], omega)
            x_ref_t, v_ref_t, a_ref_t = traj[t]
            x1, x2, x3, x4_nom, R1, R2_nom, _ = ctrl.rollout_nominal_slots(
                x_ref_t, v_ref_t, a_ref_t
            )
            y0_nom = finite_diff_y(quad.m, quad.dt, quad.g, x_full[t], x1, x2)
            y1_nom = finite_diff_y(quad.m, quad.dt, quad.g, x1, x2, x3)
            y2_nom = finite_diff_y(quad.m, quad.dt, quad.g, x2, x3, x4_nom)
            y_nom = np.concatenate([y0_nom, y1_nom, y2_nom])

            y0_true = finite_diff_y(quad.m, quad.dt, quad.g, x_full[t], x_full[t + 1], x_full[t + 2])
            y1_true = finite_diff_y(quad.m, quad.dt, quad.g, x_full[t + 1], x_full[t + 2], x_full[t + 3])
            y2_true = finite_diff_y(quad.m, quad.dt, quad.g, x_full[t + 2], x_full[t + 3], x_full[t + 4])
            y_true = np.concatenate([y0_true, y1_true, y2_true])
            dy = y_true - y_nom

            R_true_p2 = R_full[t + 2]
            xi2_true = so3_log(R1.T @ R_true_p2)
            dxi2 = so3_log(R2_nom.T @ R_true_p2)
            xi2_nom = so3_log(R1.T @ R2_nom)

            X_y.append(np.concatenate([z1, y_nom]))
            Y_y.append(dy)
            X_r.append(np.concatenate([z1, xi2_nom]))
            Y_r.append(dxi2)

    X_y_arr = np.array(X_y)
    Y_y_arr = np.array(Y_y)
    X_r_arr = np.array(X_r)
    Y_r_arr = np.array(Y_r)
    return X_y_arr, Y_y_arr, X_r_arr, Y_r_arr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--hold", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="artifacts/slots_td.npz")
    args = p.parse_args()

    X_y, Y_y, X_r, Y_r = build_slots_td(
        runs=args.runs, steps=args.steps, hold_steps=args.hold, seed=args.seed
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, X_y=X_y, Y_y=Y_y, X_r=X_r, Y_r=Y_r)

    stats_path = Path("artifacts/slots_td_stats.txt")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    mean_dy = float(np.mean(np.linalg.norm(Y_y, axis=1)))
    mean_dxi = float(np.mean(np.linalg.norm(Y_r, axis=1)))
    y_part = X_y[:, -9:]
    y_nom_all = y_part
    y_true_all = y_part + Y_y
    sig_nom = float(np.mean(np.linalg.norm(y_part - y_nom_all, axis=1)))
    sig_true = float(np.mean(np.linalg.norm(y_part - y_true_all, axis=1)))
    with stats_path.open("w") as fh:
        fh.write(
            json.dumps(
                {
                    "mean_dy": mean_dy,
                    "mean_dxi2": mean_dxi,
                    "sig_nom": sig_nom,
                    "sig_true": sig_true,
                }
            )
        )

    print(
        f"[TD] X_y:{X_y.shape} Y_y:{Y_y.shape} X_r:{X_r.shape} Y_r:{Y_r.shape}"
    )
    print(
        f"[TD] mean||Δy||={mean_dy:.3f}, mean||Δxi2||={mean_dxi:.3f}"
    )
    print(f"[TD] sig_nom={sig_nom:.3e}, sig_true={sig_true:.3e}")


if __name__ == "__main__":
    main()

