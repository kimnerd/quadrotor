try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "numpy is required to run this simulation. Install it via 'pip install numpy'."
    ) from exc

from typing import Iterator


def hat(omega: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix of omega."""
    return np.array([
        [0.0, -omega[2], omega[1]],
        [omega[2], 0.0, -omega[0]],
        [-omega[1], omega[0], 0.0],
    ])


def vee(mat: np.ndarray) -> np.ndarray:
    """Vector from a skew-symmetric matrix."""
    return np.array([mat[2, 1], mat[0, 2], mat[1, 0]])


def so3_log(R: np.ndarray) -> np.ndarray:
    """Logarithm map from SO(3) to its Lie algebra."""
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(tr)
    if theta < 1e-8:
        return np.zeros(3)
    return (theta / (2.0 * np.sin(theta))) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )


def exp_SO3(omega: np.ndarray) -> np.ndarray:
    """Exponential map from so(3) to SO(3)."""
    th = np.linalg.norm(omega)
    K = hat(omega)
    if th < 1e-8:
        return np.eye(3) + K
    return (
        np.eye(3)
        + (np.sin(th) / th) * K
        + ((1.0 - np.cos(th)) / th**2) * (K @ K)
    )


def orientation_from_accel(
    a_ref: np.ndarray,
    yaw_ref: float,
    m: float,
    g: np.ndarray,
) -> np.ndarray:
    """Construct an attitude reference matching desired acceleration and yaw."""
    F_des = m * a_ref - g
    norm_F = np.linalg.norm(F_des)
    if norm_F < 1e-6:
        b3_ref = np.array([0.0, 0.0, 1.0])
    else:
        b3_ref = F_des / norm_F

    b1_des = np.array([np.cos(yaw_ref), np.sin(yaw_ref), 0.0])
    cross = np.cross(b3_ref, b1_des)
    if np.linalg.norm(cross) < 1e-6:
        b1_des = np.array([1.0, 0.0, 0.0])
        cross = np.cross(b3_ref, b1_des)

    b2_ref = cross / np.linalg.norm(cross)
    b1_ref = np.cross(b2_ref, b3_ref)
    return np.column_stack((b1_ref, b2_ref, b3_ref))


def generate_structured_trajectory(
    start: np.ndarray, goal: np.ndarray, n_steps: int, dt: float
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (x_ref, v_ref, a_ref) along a cubic path with zero boundary velocity."""
    if n_steps < 1:
        yield start, np.zeros(3), np.zeros(3)
        return

    T = (n_steps - 1) * dt
    if T <= 0:
        for _ in range(n_steps):
            yield start, np.zeros(3), np.zeros(3)
        return

    delta = goal - start
    x_refs, v_refs = [], []
    for k in range(n_steps):
        t = k * dt
        tau = t / T
        s = 3 * tau**2 - 2 * tau**3
        v = delta / T * (6 * tau - 6 * tau**2)
        x_ref = start + delta * s
        x_refs.append(x_ref)
        v_refs.append(v)

    # extend with terminal position for discrete acceleration consistency
    x_ext = x_refs + [goal, goal]
    a_refs = []
    for k in range(n_steps):
        a = (x_ext[k + 2] - 2 * x_ext[k + 1] + x_ext[k]) / (dt**2)
        a_refs.append(a)

    for k in range(n_steps):
        yield x_refs[k], v_refs[k], a_refs[k]


class Quadrotor:
    def __init__(
        self,
        dt: float = 0.01,
        # PID gains tuned for closer convergence to positional goals
        k_p: float = 13.658566456788206,
        k_i: float = 0.541892725165471,
        k_d: float = -9.99554733444939,
        leak: float = 0.995,
        k_pz: float | None = 3.502806621239776,
        k_iz: float | None = 0.7536126100372813,
        k_dz: float | None = -5.091050690358446,
    ):
        self.dt = dt
        self.m = 1.0
        self.I = np.diag([1.0, 1.0, 1.0])
        self.l = 0.25
        self.c_t = 0.01
        self.g = np.array([0.0, 0.0, -9.81])
        self.max_force = 20.0

        # Translational PID gains
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.leak = leak
        self.k_pz = k_p if k_pz is None else k_pz
        self.k_iz = k_i if k_iz is None else k_iz
        self.k_dz = k_d if k_dz is None else k_dz

        # Attitude PID gains and integral state
        self.k_Rp = 1.0
        self.k_Rd = 0.8
        self.k_Ri = 0.2
        self.leak_R = 0.995
        self.e_R_int = np.zeros(3)

        # block inverse damping parameter
        self.lam = 1e-4

        self.x = np.zeros(3)
        self.v = np.zeros(3)
        self.R = np.eye(3)
        self.omega = np.zeros(3)
        self.e_int = np.zeros(3)

    def rotor_forces(self, T: float, M: np.ndarray) -> np.ndarray:
        """Allocate rotor forces using a pseudoinverse map."""
        A_alloc = np.array(
            [
                [1, 1, 1, 1],
                [0, -self.l, 0, self.l],
                [self.l, 0, -self.l, 0],
                [-self.c_t, self.c_t, -self.c_t, self.c_t],
            ]
        )
        forces = np.linalg.pinv(A_alloc) @ np.concatenate(([T], M))
        return np.clip(forces, 0.0, self.max_force)

    def f_x(
        self,
        x_ref: np.ndarray,
        v_ref: np.ndarray,
        a_ref: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """PID acceleration command and forward rollout of a double integrator."""
        e_x = x_ref - self.x
        e_v = v_ref - self.v
        self.e_int = self.leak * self.e_int + self.dt * e_x
        self.e_int = np.clip(self.e_int, -2.0, 2.0)
        k_p_vec = np.array([self.k_p, self.k_p, self.k_pz])
        k_d_vec = np.array([self.k_d, self.k_d, self.k_dz])
        k_i_vec = np.array([self.k_i, self.k_i, self.k_iz])
        a_cmd = a_ref + k_p_vec * e_x + k_d_vec * e_v + k_i_vec * self.e_int

        x1, v1 = self.x + self.dt * self.v, self.v + self.dt * a_cmd
        x2, v2 = x1 + self.dt * v1, v1 + self.dt * a_cmd
        x3, v3 = x2 + self.dt * v2, v2 + self.dt * a_cmd
        x4, v4 = x3 + self.dt * v3, v3 + self.dt * a_cmd
        return x1, x2, x3, x4, a_cmd

    def f_R(self, a_cmd: np.ndarray, yaw_ref: float) -> tuple[np.ndarray, np.ndarray]:
        """Two-step PID rollout on SO(3) for future attitude commands."""
        R_ref = orientation_from_accel(a_cmd, yaw_ref, self.m, self.g)

        xi = so3_log(self.R.T @ R_ref)
        self.e_R_int = self.leak_R * self.e_R_int + self.dt * xi
        self.e_R_int = np.clip(self.e_R_int, -1.0, 1.0)

        omega_cmd = (
            self.k_Rp * xi
            - self.k_Rd * self.omega
            + self.k_Ri * self.e_R_int
        )

        R1 = self.R @ exp_SO3(self.dt * omega_cmd)

        xi1 = so3_log(R1.T @ R_ref)
        omega_cmd1 = (
            self.k_Rp * xi1
            - self.k_Rd * self.omega
            + self.k_Ri * self.e_R_int
        )

        R2 = R1 @ exp_SO3(self.dt * omega_cmd1)
        return R1, R2

    def step(
        self,
        x_ref: np.ndarray,
        v_ref: np.ndarray,
        a_ref: np.ndarray,
        yaw_ref: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Advance the simulation one step using block inverse thrust solving."""
        # (a) synthesize future trajectory and attitudes
        x1, x2, x3, x4, a_cmd = self.f_x(x_ref, v_ref, a_ref)
        R1, R2 = self.f_R(a_cmd, yaw_ref)

        # (b) compute desired thrusts via block inverse
        m, dt = self.m, self.dt
        g = self.g
        y0 = m / (dt**2) * (x2 - 2 * x1 + self.x) - g
        y1 = m / (dt**2) * (x3 - 2 * x2 + x1) - g
        y2 = m / (dt**2) * (x4 - 2 * x3 + x2) - g

        ez = np.array([0.0, 0.0, 1.0])
        A = np.column_stack((self.R @ ez, R1 @ ez, R2 @ ez))
        Y = np.column_stack((y0, y1, y2))
        condA = np.linalg.cond(A)

        lam = self.lam
        if condA > 1e3:
            T0 = y0 @ (self.R @ ez)
            T1 = T2 = 0.0
        else:
            Ainv_damped = np.linalg.inv(A.T @ A + (lam**2) * np.eye(3)) @ A.T
            D = Ainv_damped @ Y
            T0, T1, T2 = np.diag(D)
        T = float(np.clip(T0, 0.0, 4.0 * self.max_force))

        # (c) torque from discrete second difference of attitude
        alpha_hat_raw = R1.T @ R2 - self.R.T @ R1
        alpha_hat = 0.5 * (alpha_hat_raw - alpha_hat_raw.T)
        alpha = vee(alpha_hat) / (dt**2)
        M = self.I @ alpha - np.cross(self.I @ self.omega, self.omega)

        forces = self.rotor_forces(T, M)

        # Update translational dynamics
        self.x += dt * self.v
        self.v += dt * (self.g + (self.R @ ez * T) / self.m)

        # Update rotational dynamics
        self.omega += dt * np.linalg.inv(self.I) @ (
            np.cross(self.I @ self.omega, self.omega) + M
        )
        self.R += dt * self.R @ hat(self.omega)
        u, _, vh = np.linalg.svd(self.R)
        self.R = u @ vh

        angle_error = np.arccos(
            np.clip((np.trace(R1.T @ self.R) - 1.0) / 2.0, -1.0, 1.0)
        )
        return forces, self.R.copy(), x_ref.copy(), R1, float(angle_error)


def simulate(
    steps: int = 100,
    target: np.ndarray | None = None,
    hold_steps: int = 0,
):
    """Run the quadrotor simulation and optionally hold at the goal."""
    quad = Quadrotor()
    if target is None:
        target = np.array([1.0, 1.0, 1.0])

    traj = list(generate_structured_trajectory(quad.x, target, steps, quad.dt))
    if traj:
        last = traj[-1]
    else:
        last = (target, np.zeros(3), np.zeros(3))
    traj.extend([last] * hold_steps)

    positions, forces, attitude_errors, R_hist, x_refs, R_refs = [], [], [], [], [], []
    for k in range(len(traj)):
        x_ref, v_ref, a_ref = traj[k]
        f, R, x_r, R_ref, err = quad.step(x_ref, v_ref, a_ref)
        positions.append(quad.x.copy())
        forces.append(f)
        R_hist.append(R)
        x_refs.append(x_r)
        R_refs.append(R_ref)
        attitude_errors.append(err)
    return (
        np.array(positions),
        np.array(forces),
        np.array(attitude_errors),
        np.array(R_hist),
        np.array(x_refs),
        np.array(R_refs),
    )


if __name__ == "__main__":
    try:  # pragma: no cover - runtime dependency guard
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required to plot the trajectory. Install it via 'pip install matplotlib'."
        ) from exc

    positions, forces, attitude_errors, R_hist, x_refs, R_refs = simulate(200, hold_steps=400)
    dt = 0.01
    t = np.arange(len(positions)) * dt

    # Plot time history for quick inspection
    plt.figure()
    plt.plot(t, positions[:, 0], label="x")
    plt.plot(t, positions[:, 1], label="y")
    plt.plot(t, positions[:, 2], label="z")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("trajectory.png")

    # 3D animation of the trajectory
    from matplotlib import animation

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x_refs[:, 0], x_refs[:, 1], x_refs[:, 2], "k--", label="reference")
    line, = ax.plot([], [], [], "b", label="actual")
    point, = ax.plot([], [], [], "ro")
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1.2)
    ax.set_zlim(0, 1.2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend()

    def update(i):
        line.set_data(positions[: i + 1, 0], positions[: i + 1, 1])
        line.set_3d_properties(positions[: i + 1, 2])
        point.set_data([positions[i, 0]], [positions[i, 1]])
        point.set_3d_properties([positions[i, 2]])
        return line, point

    ani = animation.FuncAnimation(
        fig, update, frames=len(positions), interval=20, blit=True
    )
    ani.save("trajectory.gif", writer="pillow", fps=30)

    for i, (pos, f, err, R, x_r, R_ref) in enumerate(
        zip(
            positions[:5],
            forces[:5],
            attitude_errors[:5],
            R_hist[:5],
            x_refs[:5],
            R_refs[:5],
        )
    ):
        print(
            f"Step {i}: pos={pos}, x_ref={x_r}, forces={f}, R={R}, R_ref={R_ref}, angle_error={err}"
        )
