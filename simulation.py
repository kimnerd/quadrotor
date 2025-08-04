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


def make_R_ref_from_acc(a_cmd: np.ndarray) -> np.ndarray:
    """Create a reference rotation matrix from a desired direction.

    The returned matrix aligns the body z-axis with ``a_cmd`` while keeping
    the yaw fixed so that the body x-axis is as close as possible to the
    world x-axis. Gravity is ignored; ``a_cmd`` is treated purely as a
    direction vector.

    Parameters
    ----------
    a_cmd:
        Desired direction for the body z-axis.
    """

    b3 = a_cmd
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


def compute_R_ref(current_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """Compute a reference orientation pointing toward the target position.

    A simple proportional term generates a direction vector toward
    ``target_pos`` which is converted to a rotation matrix via
    :func:`make_R_ref_from_acc`. Gravity is not considered; the resulting
    matrix merely points along the target direction.

    Parameters
    ----------
    current_pos:
        The current position of the quadrotor.
    target_pos:
        Desired position to move toward.
    """

    kp = 1.0
    a_cmd = kp * (target_pos - current_pos)
    return make_R_ref_from_acc(a_cmd)


def generate_reference_points(
    start: np.ndarray, goal: np.ndarray, n_segments: int = 10
) -> list[np.ndarray]:
    """Generate evenly spaced waypoints between ``start`` and ``goal``.

    Parameters
    ----------
    start:
        Starting position.
    goal:
        Target position.
    n_segments:
        Number of segments dividing the straight line path. Higher values
        create more intermediate reference points.
    """

    if n_segments < 1:
        return [goal]
    weights = np.linspace(0.0, 1.0, n_segments + 1)[1:]
    return [start + w * (goal - start) for w in weights]


class Quadrotor:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.m = 1.0
        self.I = np.diag([1.0, 1.0, 1.0])
        self.l = 0.25
        self.c_t = 0.01
        self.g = np.array([0.0, 0.0, -9.81])

        # Rotor force limits (Newtons)
        self.max_force = 20.0

        # State variables
        self.x = np.zeros(3)
        self.v = np.zeros(3)
        self.R = np.eye(3)
        self.omega = np.zeros(3)

        # Previous values required for the discrete-time control law
        self.fx_prev = self.x.copy()
        self.fR_prev = self.R.copy()

        # Path following state
        self.path: list[np.ndarray] = []
        self.current_wp = 0

    def set_path(self, path: list[np.ndarray]):
        """Assign a list of waypoints for the quadrotor to follow."""
        self.path = list(path)
        self.current_wp = 0

    def f_x(self, x, x_ref):
        """Correction function for position (placeholder)."""
        return x_ref

    def f_R(self, R, R_ref):
        """Correction function for rotation (placeholder)."""
        return R_ref

    def thrust_and_torque(self, x_ref, R_ref):
        """Return thrust and torque using the structured control law."""

        fx = self.f_x(self.x, x_ref)
        ez = self.R @ np.array([0, 0, 1])
        vec = self.m / (self.dt**2) * (fx - 2 * self.fx_prev + self.x) - self.g
        norm_ez = np.dot(ez, ez)
        if norm_ez < 1e-6:
            ez = np.array([0.0, 0.0, 1.0])
            norm_ez = 1.0
        T = float(ez @ vec / norm_ez)

        fR = self.f_R(self.R, R_ref)
        vee_term = vee(self.fR_prev.T @ fR - self.R.T @ self.fR_prev)
        M = self.I @ (vee_term / (self.dt**2)) - np.cross(self.I @ self.omega, self.omega)

        # Update stored previous values for next step
        self.fx_prev = fx.copy()
        self.fR_prev = fR.copy()

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

    def step(self) -> np.ndarray:
        """Advance the simulation by one time step following the path."""

        if self.current_wp < len(self.path):
            x_ref = self.path[self.current_wp]
            if np.linalg.norm(self.x - x_ref) < 1e-2:
                self.current_wp += 1
                if self.current_wp < len(self.path):
                    x_ref = self.path[self.current_wp]
        else:
            x_ref = self.x

        R_ref = compute_R_ref(self.x, x_ref)

        T, M = self.thrust_and_torque(x_ref, R_ref)
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


def simulate(
    steps: int = 100, target: np.ndarray | None = None, n_segments: int = 10
):
    """Run a simple path-following simulation."""

    quad = Quadrotor()
    if target is None:
        target = np.array([1.0, 1.0, 1.0])
    quad.set_path(generate_reference_points(quad.x, target, n_segments))

    positions = []
    forces = []
    for _ in range(steps):
        f = quad.step()
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

