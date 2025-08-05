try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "numpy is required to run this simulation. Install it via 'pip install numpy'."
    ) from exc

from typing import Iterable, Iterator, Optional


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


def exp_SO3(omega: np.ndarray) -> np.ndarray:
    """Matrix exponential for so(3) vectors."""
    theta = np.linalg.norm(omega)
    K = hat(omega)
    if theta < 1e-8:
        return np.eye(3) + K
    return (
        np.eye(3)
        + (np.sin(theta) / theta) * K
        + ((1 - np.cos(theta)) / (theta**2)) * (K @ K)
    )


def log_SO3(R: np.ndarray) -> np.ndarray:
    """Matrix logarithm for elements of SO(3)."""
    cos_theta = (np.trace(R) - 1) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return 0.5 * (R - R.T)
    return theta / (2 * np.sin(theta)) * (R - R.T)


def make_R_ref_from_acc(a_cmd: np.ndarray) -> np.ndarray:
    """Create a reference rotation matrix from a desired acceleration.

    The returned matrix aligns the body z-axis with ``a_cmd`` while keeping
    yaw as close as possible to the world x-axis.  Gravity is ignored; only
    the direction of ``a_cmd`` matters."""

    b3 = a_cmd.copy()
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


def generate_structured_trajectory(
    start: np.ndarray, goal: np.ndarray, n_steps: int, dt: float
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(x_ref, a_ref)`` pairs satisfying second-order consistency.

    The position sequence follows a quadratic profile with constant
    acceleration so that

    ``x[k+2] - 2*x[k+1] + x[k] = dt**2 * a``

    for all ``k``.  The constant acceleration ``a`` is chosen such that the
    final reference ``x_ref`` equals ``goal`` at ``k = n_steps - 1``.
    """

    if n_steps < 1:
        yield goal, np.zeros(3)
        return

    a = 2 * (goal - start) / (((n_steps - 1) ** 2) * (dt**2))
    for k in range(n_steps):
        x_ref = start + 0.5 * a * (k**2) * (dt**2)
        yield x_ref, a


def generate_orientation_refs(
    omega_refs: Iterable[np.ndarray], R0: np.ndarray, dt: float
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Integrate a sequence of angular rates into orientation references."""

    R = R0.copy()
    for omega in omega_refs:
        yield R.copy(), omega
        R = R @ exp_SO3(dt * omega)


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

        # Reference states
        self.x_ref = self.x.copy()
        self.a_ref = np.zeros(3)
        self.R_ref = self.R.copy()

        # Previous rotation error for torque computation
        self.fR_err_prev = np.zeros(3)

        # Trajectory iterators
        self.trans_iter: Optional[Iterator[tuple[np.ndarray, np.ndarray]]] = None
        self.orient_iter: Optional[Iterator[tuple[np.ndarray, np.ndarray]]] = None

    def set_path(
        self,
        translational: Iterable[tuple[np.ndarray, np.ndarray]],
        orientation: Iterable[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        """Load structured reference trajectories.

        Parameters
        ----------
        translational:
            Iterable yielding ``(x_ref, a_ref)`` pairs.
        orientation:
            Iterable yielding ``(R_ref, omega_ref)`` pairs.  ``omega_ref`` is
            integrated internally by :func:`generate_orientation_refs` and is
            provided for completeness only.
        """

        self.trans_iter = iter(translational)
        self.x_ref, self.a_ref = next(self.trans_iter, (self.x.copy(), np.zeros(3)))

        if orientation is not None:
            self.orient_iter = iter(orientation)
            self.R_ref, _ = next(
                self.orient_iter, (self.R.copy(), np.zeros(3))
            )
        else:
            self.orient_iter = None
            self.R_ref = self.R.copy()

        self.fR_err_prev.fill(0.0)

    def rotation_error(self, R, R_ref):
        """Geodesic rotation error vector."""
        return vee(log_SO3(R.T @ R_ref))

    def thrust_and_torque(self, a_ref, R_ref):
        """Return thrust and torque using the structured control law."""

        ez = self.R @ np.array([0, 0, 1])
        T = float(ez @ (self.m * a_ref - self.g))

        fR = self.rotation_error(self.R, R_ref)
        diff = fR - self.fR_err_prev
        M = self.I @ (diff / (self.dt**2)) - np.cross(self.I @ self.omega, self.omega)

        self.fR_err_prev = fR.copy()

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

    def step(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Advance the simulation one step using preloaded references."""

        x_ref, a_ref, R_ref = self.x_ref, self.a_ref, self.R_ref

        T, M = self.thrust_and_torque(a_ref, R_ref)
        forces = self.rotor_forces(T, M)

        # Update translational dynamics
        self.x += self.dt * self.v
        self.v += self.dt * (self.g + (self.R @ np.array([0, 0, 1]) * T) / self.m)

        # Update rotational dynamics
        self.omega += self.dt * np.linalg.inv(self.I) @ (
            np.cross(self.I @ self.omega, self.omega) + M
        )
        self.R += self.dt * self.R @ hat(self.omega)
        u, _, vh = np.linalg.svd(self.R)
        self.R = u @ vh

        # Compute attitude tracking error with respect to the reference
        R_err = R_ref.T @ self.R
        angle_error = np.arccos(
            np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        )

        # Advance references for next step
        if self.trans_iter is not None:
            self.x_ref, self.a_ref = next(
                self.trans_iter, (self.x_ref, self.a_ref)
            )
        if self.orient_iter is not None:
            self.R_ref, _ = next(
                self.orient_iter, (self.R_ref, np.zeros(3))
            )

        return forces, self.R.copy(), R_ref.copy(), float(angle_error)


def simulate(steps: int = 100, target: np.ndarray | None = None):
    """Run a simulation with structured translational and rotational references."""

    quad = Quadrotor()
    if target is None:
        target = np.array([1.0, 1.0, 1.0])

    # Generate translational and rotational trajectories
    trans_traj = generate_structured_trajectory(
        quad.x, target, steps, quad.dt
    )
    a = 2 * (target - quad.x) / (((steps - 1) ** 2) * (quad.dt**2))
    R0 = make_R_ref_from_acc(a)
    omega_refs = (np.zeros(3) for _ in range(steps))
    orient_traj = generate_orientation_refs(omega_refs, R0, quad.dt)

    quad.set_path(trans_traj, orient_traj)

    positions = []
    forces = []
    attitude_errors = []
    rotations = []
    rotation_refs = []
    for _ in range(steps):
        f, R, R_ref, err = quad.step()
        positions.append(quad.x.copy())
        forces.append(f)
        rotations.append(R)
        rotation_refs.append(R_ref)
        attitude_errors.append(err)
    return (
        np.array(positions),
        np.array(forces),
        np.array(attitude_errors),
        np.array(rotations),
        np.array(rotation_refs),
    )


if __name__ == "__main__":
    try:  # pragma: no cover - runtime dependency guard
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required to plot the trajectory. Install it via 'pip install matplotlib'."
        ) from exc

    positions, forces, attitude_errors, rotations, rotation_refs = simulate(200)
    dt = 0.01
    t = np.arange(len(positions)) * dt

    plt.plot(t, positions[:, 0], label="x")
    plt.plot(t, positions[:, 1], label="y")
    plt.plot(t, positions[:, 2], label="z")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.legend()
    plt.savefig("trajectory.png")

    for i, (pos, f, err, R, R_ref) in enumerate(
        zip(
            positions[:5],
            forces[:5],
            attitude_errors[:5],
            rotations[:5],
            rotation_refs[:5],
        )
    ):
        print(
            f"Step {i}: pos={pos}, forces={f}, R={R}, R_ref={R_ref}, angle_error={err}"
        )

