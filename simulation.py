try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "numpy is required to run this simulation. Install it via 'pip install numpy'."
    ) from exc

from typing import Iterator

try:
    from gpr_inverse import predict_forces
except Exception:  # pragma: no cover - optional dependency
    predict_forces = None


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


def finite_diff_y(m: float, dt: float, g: np.ndarray, x_i: np.ndarray, x_ip1: np.ndarray, x_ip2: np.ndarray) -> np.ndarray:
    """Compute slot target ``y`` using a second finite difference."""
    assert x_i.shape == (3,) and x_ip1.shape == (3,) and x_ip2.shape == (3,)
    return (m / (dt**2)) * (x_ip2 - 2.0 * x_ip1 + x_i) - g

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
        # PID gains chosen to keep x/y overshoot within ~10%
        k_p: float = 0.1,
        k_i: float = 0.0,
        k_d: float = 0.3,
        leak: float = 1.0,
        k_pz: float | None = 2.5,
        k_iz: float | None = 0.1,
        k_dz: float | None = 1.0,
        # Optional GPR inverse model
        gpr_model: object | None = None,
        use_gpr: bool = False,
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
        self.k_Rp = 5.0
        self.k_Rd = 2.5
        self.k_Ri = 0.7
        self.leak_R = 0.995
        self.e_R_int = np.zeros(3)

        # block inverse damping parameter
        self.lam = 1e-4

        self.x = np.zeros(3)
        self.v = np.zeros(3)
        self.R = np.eye(3)
        self.omega = np.zeros(3)
        self.e_int = np.zeros(3)

        # GPR inverse model handle
        self.gpr_model = gpr_model
        self.use_gpr = use_gpr

    def rotor_forces(self, T: float, M: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """Allocate rotor forces and return saturated thrust/torque."""
        A_alloc = np.array(
            [
                [1, 1, 1, 1],
                [0, -self.l, 0, self.l],
                [self.l, 0, -self.l, 0],
                [-self.c_t, self.c_t, -self.c_t, self.c_t],
            ]
        )
        forces_unc = np.linalg.pinv(A_alloc) @ np.concatenate(([T], M))
        forces = np.clip(forces_unc, 0.0, self.max_force)
        TM_act = A_alloc @ forces
        return forces, float(TM_act[0]), TM_act[1:]

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
        # prevent excessive thrust demands from aggressive errors
        a_cmd = np.clip(a_cmd, -1.0, 1.0)

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

        lam = self.lam
        condA = float("inf")
        try:
            condA = np.linalg.cond(A)
            # During holds the three-step block inverse becomes singular;
            # fall back to a single-step solve when ill-conditioned or failed.
            if not np.isfinite(condA) or condA > 1e3:
                raise np.linalg.LinAlgError
            Ainv_damped = np.linalg.inv(A.T @ A + (lam**2) * np.eye(3)) @ A.T
            D = Ainv_damped @ Y
            T0, T1, T2 = np.diag(D)
            T = float(np.clip(T0, 0.0, 4.0 * self.max_force))
            off_diag_norm = float(
                np.linalg.norm(D - np.diag(np.diag(D)), ord="fro")
            )

            # (c) torque from discrete second difference of attitude
            alpha_hat_raw = R1.T @ R2 - self.R.T @ R1
            alpha_hat = 0.5 * (alpha_hat_raw - alpha_hat_raw.T)
            alpha = vee(alpha_hat) / (dt**2)
            M = self.I @ alpha - np.cross(self.I @ self.omega, self.omega)
        except np.linalg.LinAlgError:
            # Simple hover control when block inverse is ill-conditioned
            T = float(
                np.clip((self.m * a_cmd - g) @ (self.R @ ez), 0.0, 4.0 * self.max_force)
            )
            R_ref = orientation_from_accel(a_cmd, yaw_ref, self.m, self.g)
            xi = so3_log(self.R.T @ R_ref)
            self.e_R_int = self.leak_R * self.e_R_int + self.dt * xi
            self.e_R_int = np.clip(self.e_R_int, -1.0, 1.0)
            M = (
                self.k_Rp * xi - self.k_Rd * self.omega + self.k_Ri * self.e_R_int
            )
            alpha = np.zeros(3)
            off_diag_norm = float("nan")

        if self.use_gpr and self.gpr_model is not None and predict_forces is not None:
            feat = np.concatenate(([T], M))
            forces = predict_forces(self.gpr_model, feat.reshape(1, -1))[0]
            forces = np.clip(forces, 0.0, self.max_force)
            A_alloc = np.array(
                [
                    [1, 1, 1, 1],
                    [0, -self.l, 0, self.l],
                    [self.l, 0, -self.l, 0],
                    [-self.c_t, self.c_t, -self.c_t, self.c_t],
                ]
            )
            TM_act = A_alloc @ forces
            T_act = float(TM_act[0])
            M_act = TM_act[1:]
        else:
            forces, T_act, M_act = self.rotor_forces(T, M)

        # Update translational dynamics with actual thrust
        self.x += dt * self.v
        self.v += dt * (self.g + (self.R @ ez * T_act) / self.m)

        # Update rotational dynamics with actual moments
        self.omega += dt * np.linalg.inv(self.I) @ (
            np.cross(self.I @ self.omega, self.omega) + M_act
        )
        self.R += dt * self.R @ hat(self.omega)
        u, _, vh = np.linalg.svd(self.R)
        self.R = u @ vh

        angle_error = np.arccos(
            np.clip((np.trace(R1.T @ self.R) - 1.0) / 2.0, -1.0, 1.0)
        )
        return (
            forces,
            self.R.copy(),
            x_ref.copy(),
            R1,
            float(angle_error),
            float(condA),
            off_diag_norm,
        )


def simulate(
    steps: int = 100,
    target: np.ndarray | None = None,
    hold_steps: int = 0,
    quad: Quadrotor | None = None,
):
    """Run the quadrotor simulation and optionally hold at the goal."""
    if quad is None:
        quad = Quadrotor()
    if target is None:
        target = np.array([1.0, 1.0, 1.0])

    traj = list(generate_structured_trajectory(quad.x, target, steps, quad.dt))
    if traj:
        last = traj[-1]
    else:
        last = (target, np.zeros(3), np.zeros(3))
    traj.extend([last] * hold_steps)

    positions, forces, attitude_errors, R_hist, x_refs, R_refs, conds, off_diags = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for k in range(len(traj)):
        x_ref, v_ref, a_ref = traj[k]
        f, R, x_r, R_ref, err, condA, offD = quad.step(x_ref, v_ref, a_ref)
        positions.append(quad.x.copy())
        forces.append(f)
        R_hist.append(R)
        x_refs.append(x_r)
        R_refs.append(R_ref)
        attitude_errors.append(err)
        conds.append(condA)
        off_diags.append(offD)
    return (
        np.array(positions),
        np.array(forces),
        np.array(attitude_errors),
        np.array(R_hist),
        np.array(x_refs),
        np.array(R_refs),
        np.array(conds),
        np.array(off_diags),
    )


def tune_pid(
    kp_vals: np.ndarray,
    ki_vals: np.ndarray,
    kd_vals: np.ndarray,
    steps: int = 200,
    target: np.ndarray | None = None,
    hold_steps: int = 50,
    terminal_weight: float = 1000.0,
) -> tuple[tuple[float, float, float], float]:
    """Brute-force PID tuning over a grid minimizing tracking and terminal error.

    The cost accumulates squared position errors over the full trajectory while
    also heavily penalizing the final position error after holding at the goal.
    Additional penalties discourage large overshoots in the horizontal plane.
    """
    if target is None:
        target = np.array([1.0, 1.0, 1.0])

    best_gains = (0.0, 0.0, 0.0)
    best_cost = np.inf
    for kp in kp_vals:
        for ki in ki_vals:
            for kd in kd_vals:
                quad = Quadrotor(k_p=kp, k_i=ki, k_d=kd)
                positions, _, _, _, x_refs, _, _, _ = simulate(
                    steps, target=target, quad=quad, hold_steps=hold_steps
                )
                err = positions - x_refs
                final_err = positions[-1] - target
                over = np.maximum(positions - target, 0.0)
                overshoot_xy = np.max(over[:, :2], axis=0)
                cost = float(
                    np.sum(err**2)
                    + terminal_weight * np.sum(final_err**2)
                    + 100.0 * np.sum(overshoot_xy)
                )
                if cost < best_cost:
                    best_cost = cost
                    best_gains = (float(kp), float(ki), float(kd))
    return best_gains, best_cost


if __name__ == "__main__":
    try:  # pragma: no cover - runtime dependency guard
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required to plot the trajectory. Install it via 'pip install matplotlib'."
        ) from exc
    from data_collection import collect_random_rotor_data
    from gpr_inverse import train_inverse_gpr

    # Train MR-GPR inverse model on random samples
    X, y = collect_random_rotor_data(steps=200)
    gpr_model = train_inverse_gpr(X, y)

    # Simulate ideal inverse allocation and MR-GPR allocation
    quad_ideal = Quadrotor()
    (
        pos_ideal,
        forces_ideal,
        err_ideal,
        R_hist_ideal,
        x_refs,
        R_refs_ideal,
        conds_ideal,
        off_diags_ideal,
    ) = simulate(200, hold_steps=400, quad=quad_ideal)

    quad_gpr = Quadrotor(gpr_model=gpr_model, use_gpr=True)
    (
        pos_gpr,
        forces_gpr,
        err_gpr,
        R_hist_gpr,
        _,
        _,
        _,
        _,
    ) = simulate(200, hold_steps=400, quad=quad_gpr)

    dt = 0.01
    t = np.arange(len(pos_ideal)) * dt

    # Plot time history for quick inspection
    plt.figure()
    plt.plot(t, pos_ideal[:, 0], label="x ideal")
    plt.plot(t, pos_ideal[:, 1], label="y ideal")
    plt.plot(t, pos_ideal[:, 2], label="z ideal")
    plt.plot(t, pos_gpr[:, 0], "--", label="x mr-gpr")
    plt.plot(t, pos_gpr[:, 1], "--", label="y mr-gpr")
    plt.plot(t, pos_gpr[:, 2], "--", label="z mr-gpr")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("trajectory.png")

    # 3D animation comparing trajectories
    from matplotlib import animation

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x_refs[:, 0], x_refs[:, 1], x_refs[:, 2], "k--", label="reference")
    line_ideal, = ax.plot([], [], [], "b", label="ideal")
    point_ideal, = ax.plot([], [], [], "bo")
    line_gpr, = ax.plot([], [], [], "r", label="mr-gpr")
    point_gpr, = ax.plot([], [], [], "ro")
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1.2)
    ax.set_zlim(0, 1.2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend()

    def update(i):
        line_ideal.set_data(pos_ideal[: i + 1, 0], pos_ideal[: i + 1, 1])
        line_ideal.set_3d_properties(pos_ideal[: i + 1, 2])
        point_ideal.set_data([pos_ideal[i, 0]], [pos_ideal[i, 1]])
        point_ideal.set_3d_properties([pos_ideal[i, 2]])
        line_gpr.set_data(pos_gpr[: i + 1, 0], pos_gpr[: i + 1, 1])
        line_gpr.set_3d_properties(pos_gpr[: i + 1, 2])
        point_gpr.set_data([pos_gpr[i, 0]], [pos_gpr[i, 1]])
        point_gpr.set_3d_properties([pos_gpr[i, 2]])
        return line_ideal, point_ideal, line_gpr, point_gpr

    ani = animation.FuncAnimation(
        fig, update, frames=len(pos_ideal), interval=20, blit=True
    )
    ani.save("trajectory.gif", writer="pillow", fps=30)

    for i, (
        p_i,
        f_i,
        e_i,
        R_i,
        x_r_i,
        R_ref_i,
        c_i,
        off_i,
        p_g,
        f_g,
        e_g,
    ) in enumerate(
        zip(
            pos_ideal[:5],
            forces_ideal[:5],
            err_ideal[:5],
            R_hist_ideal[:5],
            x_refs[:5],
            R_refs_ideal[:5],
            conds_ideal[:5],
            off_diags_ideal[:5],
            pos_gpr[:5],
            forces_gpr[:5],
            err_gpr[:5],
        )
    ):
        print(
            f"Step {i}: ideal pos={p_i}, forces={f_i}, angle_error={e_i}; "
            f"mr-gpr pos={p_g}, forces={f_g}, angle_error={e_g}"
        )
