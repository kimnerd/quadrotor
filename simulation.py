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

def f_x(x: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    """Translational lift ``f_x`` returning the reference position."""
    return x_ref


def f_R(R: np.ndarray, R_ref: np.ndarray) -> np.ndarray:
    """Orientation lift ``f_R`` returning the reference orientation."""
    return R_ref


def generate_structured_trajectory(
    start: np.ndarray, goal: np.ndarray, n_steps: int, dt: float
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(x_ref, a_ref)`` pairs satisfying the discrete condition.

    The trajectory follows a cubic polynomial with zero boundary velocity.
    Acceleration references are computed via discrete second differences
    so that ``x_ref[k+2] - 2*x_ref[k+1] + x_ref[k] = dt**2 * a_ref[k]``.
    """

    if n_steps < 1:
        yield goal, np.zeros(3)
        return

    T = (n_steps - 1) * dt
    if T <= 0:
        for _ in range(n_steps):
            yield start, np.zeros(3)
        return

    delta = goal - start
    x_refs: list[np.ndarray] = []
    for k in range(n_steps + 2):
        if k >= n_steps:
            x_ref = goal.copy()
        else:
            t = k * dt
            tau = t / T
            s = 3 * tau**2 - 2 * tau**3
            x_ref = start + delta * s
        x_refs.append(x_ref)

    for k in range(n_steps + 2):
        if k <= n_steps - 1:
            a_ref = (x_refs[k + 2] - 2 * x_refs[k + 1] + x_refs[k]) / (dt**2)
        else:
            a_ref = np.zeros(3)
        yield x_refs[k], a_ref


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

        # Reference data
        self.trans_refs: list[tuple[np.ndarray, np.ndarray]] = []
        self.trans_idx = 0
        self.x_ref = self.x.copy()
        self.f_x_prev = f_x(self.x, self.x_ref)
        self._f_x_now = self.f_x_prev

        self.orient_refs: list[tuple[np.ndarray, np.ndarray]] = []
        self.orient_idx = 0
        self.R_ref = self.R.copy()
        self.f_R_prev = f_R(self.R, self.R_ref)
        self._f_R_now = self.f_R_prev

    def set_path(
        self,
        translational: Iterable[tuple[np.ndarray, np.ndarray]],
        orientation: Iterable[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        """Load structured reference trajectories.

        ``translational`` and ``orientation`` should provide at least two steps of
        lookahead so that the control law can form second differences.  The
        ``orientation`` iterable is expected to yield ``(R_ref, omega_ref)`` pairs
        such as those produced by :func:`generate_orientation_refs`.
        """

        self.trans_refs = list(translational)
        if len(self.trans_refs) < 3:
            raise ValueError("translational references require at least three entries")
        self.trans_idx = 0
        self.x_ref, _ = self.trans_refs[0]
        self.f_x_prev = f_x(self.x, self.trans_refs[0][0])
        self._f_x_now = self.f_x_prev

        if orientation is not None:
            self.orient_refs = list(orientation)
        else:
            self.orient_refs = [(self.R.copy(), np.zeros(3))] * len(self.trans_refs)
        if len(self.orient_refs) < 3:
            raise ValueError("orientation references require at least three entries")
        self.orient_idx = 0
        self.R_ref = self.orient_refs[0][0]
        self.f_R_prev = f_R(self.R, self.orient_refs[0][0])
        self._f_R_now = self.f_R_prev

    def thrust_and_torque(self) -> tuple[float, np.ndarray]:
        """Compute thrust ``T`` and torque ``M`` following the feedback law."""
        x0 = f_x(self.x, self.trans_refs[self.trans_idx][0])
        x1 = f_x(self.x, self.trans_refs[self.trans_idx + 1][0])
        x2 = f_x(self.x, self.trans_refs[self.trans_idx + 2][0])
        acc = (x2 - 2 * x1 + x0) / (self.dt**2)
        T = float((self.R @ np.array([0, 0, 1])) @ (self.m * acc - self.g))

        R0 = f_R(self.R, self.orient_refs[self.orient_idx][0])
        R1 = f_R(self.R, self.orient_refs[self.orient_idx + 1][0])
        R2 = f_R(self.R, self.orient_refs[self.orient_idx + 2][0])
        alpha = vee(R1.T @ R2 - R0.T @ R1) / (self.dt**2)
        M = self.I @ alpha - np.cross(self.I @ self.omega, self.omega)

        self._f_x_now = x1
        self._f_R_now = R1

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

        R_ref_now = self.R_ref
        T, M = self.thrust_and_torque()

        # Cache reference lifts for the next step
        self.f_x_prev = self._f_x_now
        self.f_R_prev = self._f_R_now

        # Advance reference indices while keeping a two-step lookahead
        if self.trans_idx + 3 < len(self.trans_refs):
            self.trans_idx += 1
            self.x_ref, _ = self.trans_refs[self.trans_idx]
        else:
            self.x_ref = self._f_x_now

        if self.orient_idx + 3 < len(self.orient_refs):
            self.orient_idx += 1
            self.R_ref = self.orient_refs[self.orient_idx][0]
        else:
            self.R_ref = self._f_R_now

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
        R_err = R_ref_now.T @ self.R
        angle_error = np.arccos(
            np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        )

        return forces, self.R.copy(), R_ref_now.copy(), float(angle_error)


def simulate(
    steps: int = 100,
    target: np.ndarray | None = None,
    omega_refs: Iterable[np.ndarray] | None = None,
):
    """Run a simulation with structured translational and rotational references.

    Orientation references ``R_ref`` are generated by integrating desired angular
    rates via :func:`generate_orientation_refs`.  The function returns a tuple
    ``(positions, forces, attitude_errors, R_hist, R_refs)`` where ``R_hist``
    contains the actual rotation matrices and ``R_refs`` the integrated reference
    orientations.  A custom sequence of angular rates may be supplied via
    ``omega_refs`` to define a non-trivial ``R_ref``.
    """

    quad = Quadrotor()
    if target is None:
        target = np.array([1.0, 1.0, 1.0])

    # Generate translational trajectory
    trans_traj = list(
        generate_structured_trajectory(quad.x, target, steps, quad.dt)
    )

    # Orientation references
    if omega_refs is None:
        # Align body z-axis with required total acceleration to reach the target.
        g = quad.g
        R_refs = []
        for _, a_ref in trans_traj:
            f_d = a_ref - g
            norm = np.linalg.norm(f_d)
            if norm < 1e-8:
                R_refs.append(np.eye(3))
                continue
            z_b = f_d / norm
            x_c = np.array([1.0, 0.0, 0.0])
            y_b = np.cross(z_b, x_c)
            if np.linalg.norm(y_b) < 1e-8:
                x_c = np.array([0.0, 1.0, 0.0])
                y_b = np.cross(z_b, x_c)
            y_b /= np.linalg.norm(y_b)
            x_b = np.cross(y_b, z_b)
            R_refs.append(np.column_stack((x_b, y_b, z_b)))
        omega_refs = []
        for k in range(len(R_refs) - 1):
            Rk, Rk1 = R_refs[k], R_refs[k + 1]
            omega = vee((Rk.T @ Rk1 - Rk1.T @ Rk) / (2 * quad.dt))
            omega_refs.append(omega)
        omega_refs.append(np.zeros(3))
        orient_traj = list(zip(R_refs, omega_refs))
    else:
        # Integrate supplied angular rates to produce reference orientations
        orient_traj = list(generate_orientation_refs(omega_refs, quad.R, quad.dt))

    quad.set_path(trans_traj, orient_traj)

    positions = []
    forces = []
    attitude_errors = []
    R_hist = []
    R_refs = []
    for _ in range(steps):
        f, R, R_ref, err = quad.step()
        positions.append(quad.x.copy())
        forces.append(f)
        R_hist.append(R)
        R_refs.append(R_ref)
        attitude_errors.append(err)
    return (
        np.array(positions),
        np.array(forces),
        np.array(attitude_errors),
        np.array(R_hist),
        np.array(R_refs),
    )


if __name__ == "__main__":
    try:  # pragma: no cover - runtime dependency guard
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required to plot the trajectory. Install it via 'pip install matplotlib'."
        ) from exc

    positions, forces, attitude_errors, R_hist, R_refs = simulate(200)
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
            R_hist[:5],
            R_refs[:5],
        )
    ):
        print(
            f"Step {i}: pos={pos}, forces={f}, R={R}, R_ref={R_ref}, angle_error={err}"
        )

