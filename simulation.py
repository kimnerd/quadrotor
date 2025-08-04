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


def make_R_ref_from_acc(a_cmd: np.ndarray, g: np.ndarray = np.array([0.0, 0.0, -9.81])) -> np.ndarray:
    """Create a reference rotation matrix from desired acceleration.

    The returned matrix aligns the body z-axis with the desired force
    direction ``a_cmd - g`` while keeping the yaw fixed so that the body
    x-axis is as close as possible to the world x-axis.

    Parameters
    ----------
    a_cmd:
        Desired acceleration command.
    g:
        Gravity vector used to compute the desired force direction. Defaults
        to standard Earth gravity ``[0, 0, -9.81]``.
    """

    b3 = a_cmd - g
    norm_b3 = np.linalg.norm(b3)
    if norm_b3 < 1e-6:
        b3 = np.array([0.0, 0.0, 1.0])
    else:
        b3 /= norm_b3

    b1_ref = np.array([1.0, 0.0, 0.0])
    b2 = np.cross(b3, b1_ref)
    norm_b2 = np.linalg.norm(b2)
    if norm_b2 < 1e-6:
        b2 = np.array([0.0, 1.0, 0.0])
    else:
        b2 /= norm_b2
    b1 = np.cross(b2, b3)
    return np.column_stack((b1, b2, b3))


class Quadrotor:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.m = 1.0
        self.I = np.diag([1.0, 1.0, 1.0])
        self.l = 0.25
        self.c_t = 0.01
        self.g = np.array([0.0, 0.0, -9.81])

        # Tunable PID gains
        self.kx_p, self.kx_d, self.kx_i = 2.0, 0.8, 0.2
        self.kR_p, self.kR_d, self.kR_i = 4.0, 0.2, 0.05

        # Rotor force limits (Newtons)
        self.max_force = 20.0

        # State variables
        self.x = np.zeros(3)
        self.v = np.zeros(3)
        self.R = np.eye(3)
        self.omega = np.zeros(3)

        # Integral errors for PID control
        self.x_int = np.zeros(3)
        self.R_int = np.zeros(3)

    def f_x(self, x, v, x_ref):
        """PID controller for translational dynamics returning acceleration."""
        k_p, k_d, k_i = self.kx_p, self.kx_d, self.kx_i
        error = x_ref - x
        self.x_int += error * self.dt
        # simple anti-windup to keep the integral term bounded
        self.x_int = np.clip(self.x_int, -0.5, 0.5)
        return k_p * error - k_d * v + k_i * self.x_int

    def f_R(self, R, omega, R_ref):
        """PID controller for rotational dynamics returning torque."""
        k_p, k_d, k_i = self.kR_p, self.kR_d, self.kR_i
        e_R_mat = 0.5 * (R_ref.T @ R - R.T @ R_ref)
        e_R = vee(e_R_mat)
        self.R_int += e_R * self.dt
        self.R_int = np.clip(self.R_int, -0.5, 0.5)
        return -k_p * e_R - k_d * omega - k_i * self.R_int

    def thrust_and_torque(self, x_ref, R_ref=None, auto_ref=True):
        """Return thrust and torque for given references.

        If ``auto_ref`` is True, ``R_ref`` is ignored and a reference
        attitude is computed from the desired acceleration. Otherwise the
        provided ``R_ref`` is used (or identity if ``None``).
        """

        # Compute thrust command
        a_cmd = self.f_x(self.x, self.v, x_ref)
        if auto_ref:
            R_ref = make_R_ref_from_acc(a_cmd, self.g)
        elif R_ref is None:
            R_ref = np.eye(3)
        vec = self.m * (a_cmd - self.g)
        if auto_ref:
            ez = R_ref @ np.array([0, 0, 1])
        else:
            ez = self.R @ np.array([0, 0, 1])
        norm_ez = np.linalg.norm(ez)
        if norm_ez < 1e-6:
            ez = np.array([0, 0, 1])
            norm_ez = 1.0
        T = float(ez @ vec / (norm_ez**2))

        # Compute torque
        M = self.f_R(self.R, self.omega, R_ref)
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
        # Enforce physical rotor limits
        forces = np.clip(forces, 0.0, self.max_force)
        return forces

    def step(self, x_ref, R_ref=None, auto_ref=True):
        T, M = self.thrust_and_torque(x_ref, R_ref, auto_ref=auto_ref)
        forces = self.rotor_forces(T, M)

        # Update translational dynamics
        self.x += self.dt * self.v
        self.v += self.dt * (self.g + (self.R @ np.array([0, 0, 1]) * T) / self.m)

        # Update rotational dynamics
        self.omega += self.dt * np.linalg.inv(self.I) @ (
            np.cross(self.I @ self.omega, self.omega) + M
        )
        self.R += self.dt * self.R @ hat(self.omega)
        # Re-orthonormalize R
        u, _, vh = np.linalg.svd(self.R)
        self.R = u @ vh

        return forces


def simulate(steps=100, auto_ref=True):
    """Run a simple position-hold simulation.

    Parameters
    ----------
    steps:
        Number of simulation steps to run.
    auto_ref:
        If ``True``, compute the reference attitude from the desired
        acceleration each step. Otherwise ``R_ref`` is used directly.
    """

    quad = Quadrotor()
    x_ref = np.array([1.0, 1.0, 1.0])
    R_ref = np.eye(3)

    positions = []
    forces = []
    for _ in range(steps):
        f = quad.step(x_ref, R_ref, auto_ref=auto_ref)
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

    positions, forces = simulate(200, auto_ref=True)
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

