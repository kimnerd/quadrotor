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
) -> Iterator[np.ndarray]:
    """Yield evenly spaced waypoints between ``start`` and ``goal``.

    The function returns a generator so that only one waypoint is held in
    memory at a time.  This allows extremely large ``n_segments`` values
    without allocating massive arrays.

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
        yield goal
        return
    step = (goal - start) / n_segments
    for i in range(1, n_segments + 1):
        yield start + step * i


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

        # Reference orientation state
        self.R_ref = self.R.copy()

        # PID controller state for position
        self.pos_err_int = np.zeros(3)
        self.pos_err_prev = np.zeros(3)

        # Previous rotation error for torque computation
        self.fR_err_prev = np.zeros(3)

        # PID gains
        self.kp_pos = 1.0
        self.ki_pos = 0.0
        self.kd_pos = 0.3
        self.kp_att = 1.0
        self.ki_att = 0.0
        self.kd_att = 0.3  # Retained for compatibility

        # Path following state
        self.path_iter: Optional[Iterator[np.ndarray]] = None
        self.current_wp: Optional[np.ndarray] = None

    def set_path(self, path: Iterable[np.ndarray]):
        """Assign a waypoint generator for the quadrotor to follow."""
        self.path_iter = iter(path)
        self.current_wp = next(self.path_iter, None)

        # Reset stored terms for new path
        self.R_ref = self.R.copy()
        self.fR_err_prev.fill(0.0)
        self.pos_err_int.fill(0.0)
        self.pos_err_prev.fill(0.0)
        # No attitude integral state needed for structured control

    def a_ref(self, x, x_ref):
        """PID acceleration command for position."""
        err = x_ref - x
        self.pos_err_int += err * self.dt
        derr = (err - self.pos_err_prev) / self.dt
        self.pos_err_prev = err
        return (
            self.kp_pos * err
            + self.ki_pos * self.pos_err_int
            + self.kd_pos * derr
        )

    def rotation_error(self, R, R_ref):
        """Geodesic rotation error vector."""
        return vee(log_SO3(R.T @ R_ref))

    def thrust_and_torque(self, x_ref, R_ref):
        """Return thrust and torque using the structured control law."""

        a_cmd = self.a_ref(self.x, x_ref)
        max_a = 1e3
        norm_a = np.linalg.norm(a_cmd)
        if norm_a > max_a:
            a_cmd = a_cmd / norm_a * max_a
        ez = self.R @ np.array([0, 0, 1])
        T = float(ez @ (self.m * a_cmd - self.g))

        rot_err = self.rotation_error(self.R, R_ref)
        diff = rot_err - self.fR_err_prev
        max_diff = 1e3
        norm_diff = np.linalg.norm(diff)
        if norm_diff > max_diff:
            diff = diff / norm_diff * max_diff
        M = self.I @ (diff / (self.dt**2)) - np.cross(self.I @ self.omega, self.omega)

        self.fR_err_prev = rot_err.copy()

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
        """Advance the simulation and report attitude error.

        Returns
        -------
        forces:
            Array of rotor forces applied this step.
        R:
            Current rotation matrix after the update.
        R_ref:
            Reference rotation matrix computed for this step.
        angle_error:
            Magnitude of the rotation between the current orientation and the
            reference orientation in radians.
        """

        if self.current_wp is not None:
            x_ref = self.current_wp
            if np.linalg.norm(self.x - x_ref) < 1e-2 and self.path_iter is not None:
                self.current_wp = next(self.path_iter, None)
                x_ref = self.current_wp if self.current_wp is not None else self.x
        else:
            x_ref = self.x

        R_des = compute_R_ref(self.x, x_ref)
        omega_ref = vee(log_SO3(self.R_ref.T @ R_des)) / self.dt
        self.R_ref = self.R_ref @ exp_SO3(self.dt * omega_ref)
        u, _, vh = np.linalg.svd(self.R_ref)
        self.R_ref = u @ vh

        T, M = self.thrust_and_torque(x_ref, self.R_ref)
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

        # Compute attitude tracking error with respect to the reference
        R_err = self.R_ref.T @ self.R
        angle_error = np.arccos(
            np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        )

        return forces, self.R.copy(), self.R_ref.copy(), float(angle_error)


def simulate(
    steps: int = 100, target: np.ndarray | None = None, segments: int = 10
):
    """Run a simple path-following simulation.

    Parameters
    ----------
    steps:
        Number of simulation iterations to perform.
    target:
        Final position for the quadrotor.  If ``None`` a default target is
        used.
    segments:
        Number of path segments between the current position and ``target``.
        This affects only the density of waypoints; it is independent of
        ``steps``.

    Returns
    -------
    positions:
        Array of positions at each time step.
    forces:
        Array of rotor forces for each step.
    attitude_errors:
        Orientation tracking error (radians) for every step.
    rotations:
        Rotation matrix of the vehicle at each step.
    rotation_refs:
        Reference rotation matrix tracked at each step.
    """

    quad = Quadrotor()
    if target is None:
        target = np.array([1.0, 1.0, 1.0])

    quad.set_path(generate_reference_points(quad.x, target, segments))

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

