"""Random rotor-force data collection for training inverse models."""

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit("numpy required. Install via 'pip install numpy'.") from exc

from simulation import Quadrotor


def collect_random_rotor_data(
    steps: int,
    force_range: tuple[float, float] = (0.0, 20.0),
    quad: Quadrotor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect random thrust/torque samples and corresponding rotor forces."""
    if quad is None:
        quad = Quadrotor()
    A_alloc = np.array(
        [
            [1, 1, 1, 1],
            [0, -quad.l, 0, quad.l],
            [quad.l, 0, -quad.l, 0],
            [-quad.c_t, quad.c_t, -quad.c_t, quad.c_t],
        ]
    )
    X, y = [], []
    for _ in range(steps):
        forces = np.random.uniform(force_range[0], force_range[1], size=4)
        TM = A_alloc @ forces
        X.append(TM)
        y.append(forces)
    return np.array(X), np.array(y)
