from __future__ import annotations

"""Quadrotor subclass that applies residual GPR corrections to rotor forces."""

from typing import Tuple

import numpy as np

from residual_gpr import ResidualMRGPR
from simulation import Quadrotor


class ResidualQuadrotor(Quadrotor):
    """Quadrotor with residual-based rotor allocation using a GPR model."""

    def __init__(
        self,
        residual_model: ResidualMRGPR,
        residual_trust_scale: float = 1.0,
        residual_var_clip: float = 5.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.residual_model = residual_model
        self.residual_trust_scale = residual_trust_scale
        self.residual_var_clip = residual_var_clip

    def _features_for_residual(self, T: float, M: np.ndarray) -> np.ndarray:
        ez = np.array([0.0, 0.0, 1.0])
        Re = self.R @ ez
        return np.concatenate(([T], M, Re, self.omega))

    def rotor_forces(self, T: float, M: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """Override rotor allocation with residual GPR correction."""

        # Nominal inverse allocation (no saturation)
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

        # Residual prediction
        phi = self._features_for_residual(T, M).reshape(1, -1)
        mu, var = self.residual_model.predict(phi)
        mu, var = mu[0], var[0]

        # Variance-based gating
        w = self.residual_trust_scale * (1.0 / (1.0 + (var / self.residual_var_clip)))
        forces = forces_nom + w * mu
        forces = np.clip(forces, 0.0, self.max_force)

        TM_act = A_alloc @ forces
        return forces, float(TM_act[0]), TM_act[1:]
