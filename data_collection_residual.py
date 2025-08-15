from __future__ import annotations

"""Data collection utilities for residual GPR training."""

from typing import Tuple

import numpy as np

from simulation import Quadrotor, simulate


class RecorderQuadrotor(Quadrotor):
    """Quadrotor wrapper to capture features and nominal forces for training."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_phi: np.ndarray | None = None
        self.last_forces_nom: np.ndarray | None = None

    def rotor_forces(self, T: float, M: np.ndarray):
        # Compute nominal forces without saturation and record
        A_alloc = np.array(
            [
                [1, 1, 1, 1],
                [0, -self.l, 0, self.l],
                [self.l, 0, -self.l, 0],
                [-self.c_t, self.c_t, -self.c_t, self.c_t],
            ]
        )
        TM = np.concatenate(([T], M))
        forces_nom = np.linalg.pinv(A_alloc) @ TM
        self.last_forces_nom = forces_nom.copy()

        ez = np.array([0.0, 0.0, 1.0])
        Re = self.R @ ez
        self.last_phi = np.concatenate(([T], M, Re, self.omega))

        # Continue with nominal saturation behaviour
        forces_unc = np.linalg.pinv(A_alloc) @ TM
        forces = np.clip(forces_unc, 0.0, self.max_force)
        TM_act = A_alloc @ forces
        return forces, float(TM_act[0]), TM_act[1:]


def collect_residual_data(
    steps: int = 2000,
    target: np.ndarray | None = None,
    hold_steps: int = 200,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect training data for residual GPR.

    Returns
    -------
    X : ndarray of shape (N, 10)
        Features ``[T, Mx, My, Mz, (R@ez), omega]``.
    Y : ndarray of shape (N, 4)
        Residuals ``f_true - f_nom``.
    """

    rng = np.random.default_rng(seed)
    if target is None:
        target = np.array([1.0, 1.0, 1.0])

    quad = RecorderQuadrotor()

    # Warm start to populate initial state
    simulate(200, target=target, hold_steps=hold_steps, quad=quad)

    Xs, Ys = [], []
    # Hidden actuator model parameters
    gains = rng.uniform(0.9, 1.1, size=4)
    biases = rng.uniform(-0.5, 0.5, size=4)

    for _ in range(steps):
        goal = rng.uniform(0.5, 1.2, size=3)
        simulate(60, target=goal, hold_steps=10, quad=quad)

        if quad.last_phi is None or quad.last_forces_nom is None:
            continue
        phi = quad.last_phi
        f_nom = quad.last_forces_nom

        # Synthesize "true" forces with bias/gain/noise and saturation
        f_true = gains * f_nom + biases + rng.normal(0.0, 0.05, size=4)
        f_true = np.clip(f_true, 0.0, quad.max_force)
        delta = f_true - f_nom

        Xs.append(phi)
        Ys.append(delta)

    return np.array(Xs), np.array(Ys)
