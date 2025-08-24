import numpy as np
from simulation import Quadrotor, hat


class NominalQuadrotor(Quadrotor):
    """Nominal plant with default parameters."""
    pass


class RealQuadrotor(Quadrotor):
    """Real plant with parameter mismatch and simple disturbances."""

    def __init__(self, wind: np.ndarray | None = None, kv: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        # Parameter mismatches
        self.m *= 1.10
        self.I = np.diag(np.diag(self.I) * np.array([1.05, 0.95, 1.10]))
        self.max_force *= 0.9
        # Asymmetric yaw coefficients per rotor
        self.c_t_asym = self.c_t * np.array([1.0, 0.95, 1.05, 1.0])
        self.k_v = kv
        self.wind = np.zeros(3) if wind is None else wind

    def rotor_forces(self, T: float, M: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        A_alloc = np.array(
            [
                [1, 1, 1, 1],
                [0, -self.l, 0, self.l],
                [self.l, 0, -self.l, 0],
                [-self.c_t_asym[0], self.c_t_asym[1], -self.c_t_asym[2], self.c_t_asym[3]],
            ]
        )
        forces_unc = np.linalg.pinv(A_alloc) @ np.concatenate(([T], M))
        forces = np.clip(forces_unc, 0.0, self.max_force)
        TM_act = A_alloc @ forces
        return forces, float(TM_act[0]), TM_act[1:]

    def step(
        self,
        x_ref: np.ndarray,
        v_ref: np.ndarray,
        a_ref: np.ndarray,
        yaw_ref: float = 0.0,
    ):
        forces, R, x_ref_out, R1, angle_error, condA, off_diag = super().step(
            x_ref, v_ref, a_ref, yaw_ref
        )
        # Apply simple drag and wind disturbance
        self.v += self.dt * (-self.k_v * self.v + self.wind)
        return forces, R, x_ref_out, R1, angle_error, condA, off_diag
