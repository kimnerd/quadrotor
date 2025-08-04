try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "numpy is required to run this simulation. Install it via 'pip install numpy'."
    ) from exc


def hat(omega: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix of omega."""
    return np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0],
    ])


def vee(mat: np.ndarray) -> np.ndarray:
    """Vector from skew-symmetric matrix."""
    return np.array([mat[2, 1], mat[0, 2], mat[1, 0]])


class Quadrotor:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.m = 1.0
        self.I = np.diag([1.0, 1.0, 1.0])
        self.l = 0.25
        self.c_t = 0.01
        self.g = np.array([0.0, 0.0, -9.81])

        # State variables
        self.x = np.zeros(3)
        self.v = np.zeros(3)
        self.R = np.eye(3)
        self.omega = np.zeros(3)

        # History for control law
        self.x_prev = np.copy(self.x)
        self.R_prev = np.copy(self.R)

    def f_x(self, x, x_ref):
        """Correction function for position. Placeholder PD term."""
        k_p = 0.5
        return x + k_p * (x_ref - x)

    def f_R(self, R, R_ref):
        """Correction function for rotation. Returns reference orientation."""
        return R_ref

    def thrust_and_torque(self, x_ref, R_ref):
        # Compute thrust
        term = (self.f_x(self.x, x_ref) - 2 * self.f_x(self.x_prev, x_ref) + self.x)
        vec = self.m / self.dt**2 * term - self.g
        ez = self.R @ np.array([0, 0, 1])
        T = np.linalg.pinv(ez.reshape(3, 1)) @ vec
        T = float(T)

        # Compute torque
        fR_prev = self.f_R(self.R_prev, R_ref)
        fR_curr = self.f_R(self.R, R_ref)
        mat = fR_prev.T @ fR_curr - self.R.T @ fR_prev
        M = self.I / self.dt**2 @ vee(mat) - np.cross(self.I @ self.omega, self.omega)
        return T, M

    def rotor_forces(self, T, M):
        l, c_t = self.l, self.c_t
        A = np.array([
            [1, 1, 1, 1],
            [0, -l, 0, l],
            [l, 0, -l, 0],
            [-c_t, c_t, -c_t, c_t],
        ])
        forces = np.linalg.pinv(A) @ np.concatenate(([T], M))
        return forces

    def step(self, x_ref, R_ref):
        T, M = self.thrust_and_torque(x_ref, R_ref)
        forces = self.rotor_forces(T, M)

        # Update translational dynamics
        self.x_prev = np.copy(self.x)
        self.x += self.dt * self.v
        self.v += self.dt * (self.g + self.R @ np.array([0, 0, 1]) * T) / self.m

        # Update rotational dynamics
        self.R_prev = np.copy(self.R)
        self.omega += self.dt * np.linalg.inv(self.I) @ (
            np.cross(self.I @ self.omega, self.omega) + M
        )
        self.R += self.dt * self.R @ hat(self.omega)
        # Re-orthonormalize R
        u, _, vh = np.linalg.svd(self.R)
        self.R = u @ vh

        return forces


def simulate(steps=100):
    quad = Quadrotor()
    x_ref = np.array([1.0, 1.0, 1.0])
    R_ref = np.eye(3)

    positions = []
    forces = []
    for _ in range(steps):
        f = quad.step(x_ref, R_ref)
        positions.append(quad.x.copy())
        forces.append(f)
    return np.array(positions), np.array(forces)


if __name__ == "__main__":
    try:  # pragma: no cover - runtime dependency guard
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required to plot the trajectory. Install it via 'pip install matplotlib'."
        ) from exc

    positions, forces = simulate(200)
    dt = 0.01
    t = np.arange(len(positions)) * dt

    plt.plot(t, positions[:, 0], label="x")
    plt.plot(t, positions[:, 1], label="y")
    plt.plot(t, positions[:, 2], label="z")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.legend()
    plt.savefig("trajectory.png")

    for i, (pos, f) in enumerate(zip(positions[:5], forces[:5])):
        print(f"Step {i}: pos={pos}, forces={f}")

