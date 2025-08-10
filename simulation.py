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
    """Yield ``(x_ref, v_ref, a_ref)`` along a cubic path with zero boundary velocity."""
    if n_steps < 1:
        yield start, np.zeros(3), np.zeros(3)
        return

    T = (n_steps - 1) * dt
    if T <= 0:
        for _ in range(n_steps):
            yield start, np.zeros(3), np.zeros(3)
        return

    delta = goal - start
    for k in range(n_steps):
        t = k * dt
        tau = t / T
        s = 3 * tau ** 2 - 2 * tau ** 3
        v = delta / T * (6 * tau - 6 * tau ** 2)
        if k == 0 or k == n_steps - 1:
            a = np.zeros(3)
        else:
            a = delta / (T ** 2) * (6 - 12 * tau)
        x_ref = start + delta * s
        yield x_ref, v, a


class Quadrotor:
    def __init__(self, dt: float = 0.01):
        self.dt = dt
        self.m = 1.0
        self.I = np.diag([1.0, 1.0, 1.0])
        self.l = 0.25
        self.c_t = 0.01
        self.g = np.array([0.0, 0.0, -9.81])
        self.max_force = 20.0

        self.x = np.zeros(3)
        self.v = np.zeros(3)
        self.R = np.eye(3)
        self.omega = np.zeros(3)
        self.e_int = np.zeros(3)

    def rotor_forces(self, T: float, M: np.ndarray) -> np.ndarray:
        l, c_t = self.l, self.c_t
        A = np.array([
            [1, 1, 1, 1],
            [0, -l, 0, l],
            [l, 0, -l, 0],
            [-c_t, c_t, -c_t, c_t],
        ])
        forces = np.linalg.pinv(A) @ np.concatenate(([T], M))
        return np.clip(forces, 0.0, self.max_force)

    def step(
        self,
        x_ref: np.ndarray,
        v_ref: np.ndarray,
        a_ref: np.ndarray,
        yaw_ref: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Advance the simulation one step toward the reference state."""
        # Tuned gains for accurate and well damped trajectory tracking
        k_p = 1.2
        k_i = 0.1
        k_d = 4.2
        k_R = 50.0
        k_omega = 12.0

        e_x = x_ref - self.x
        e_v = v_ref - self.v
        # Slight integral leak prevents windup when holding position
        self.e_int = 0.98 * self.e_int + self.dt * e_x
        self.e_int = np.clip(self.e_int, -2.0, 2.0)
        a_cmd = a_ref + k_p * e_x + k_d * e_v + k_i * self.e_int
        R_ref = orientation_from_accel(a_cmd, yaw_ref, self.m, self.g)

        F_des = self.m * a_cmd - self.g
        T = float(np.dot(F_des, self.R @ np.array([0.0, 0.0, 1.0])))

        e_R = 0.5 * vee(R_ref.T @ self.R - self.R.T @ R_ref)
        e_omega = self.omega
        M = -k_R * e_R - k_omega * e_omega
        M = np.clip(M, -4.0, 4.0)

        forces = self.rotor_forces(T, M)

        # Update translational dynamics
        self.x += self.dt * self.v
        self.v += self.dt * (self.g + (self.R @ np.array([0.0, 0.0, 1.0]) * T) / self.m)

        # Update rotational dynamics
        self.omega += self.dt * np.linalg.inv(self.I) @ (
            np.cross(self.I @ self.omega, self.omega) + M
        )
        self.R += self.dt * self.R @ hat(self.omega)
        u, _, vh = np.linalg.svd(self.R)
        self.R = u @ vh

        angle_error = np.arccos(
            np.clip((np.trace(R_ref.T @ self.R) - 1.0) / 2.0, -1.0, 1.0)
        )
        return forces, self.R.copy(), x_ref.copy(), R_ref, float(angle_error)


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

    positions, forces, attitude_errors, R_hist, x_refs, R_refs = simulate(200)
    dt = 0.01
    t = np.arange(len(positions)) * dt

    plt.plot(t, positions[:, 0], label="x")
    plt.plot(t, positions[:, 1], label="y")
    plt.plot(t, positions[:, 2], label="z")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.legend()
    plt.savefig("trajectory.png")

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
