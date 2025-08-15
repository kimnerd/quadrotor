from __future__ import annotations

"""Train and compare baseline, absolute MR-GPR, and residual MR-GPR."""

import numpy as np

from data_collection import collect_random_rotor_data
from data_collection_residual import collect_residual_data
from gpr_inverse import train_inverse_gpr
from residual_allocator import ResidualQuadrotor
from residual_gpr import ResidualMRGPR
from simulation import Quadrotor, simulate


def main() -> None:
    # 1) Collect residual training data and train residual model
    X_res, Y_res = collect_residual_data(steps=3000)
    model_res = ResidualMRGPR(input_dim=X_res.shape[1])
    model_res.fit(X_res, Y_res)

    # 2) Absolute MR-GPR model (existing path)
    X_abs, y_abs = collect_random_rotor_data(steps=200)
    model_abs = train_inverse_gpr(X_abs, y_abs)

    # 3) Run simulations for the three approaches
    quad_nom = Quadrotor()
    pos_nom, forces_nom, err_nom, *_ = simulate(200, hold_steps=400, quad=quad_nom)

    quad_abs = Quadrotor(gpr_model=model_abs, use_gpr=True)
    pos_abs, forces_abs, err_abs, *_ = simulate(200, hold_steps=400, quad=quad_abs)

    quad_res = ResidualQuadrotor(residual_model=model_res)
    pos_res, forces_res, err_res, *_ = simulate(200, hold_steps=400, quad=quad_res)

    # 4) Metric comparison
    def metrics(pos: np.ndarray) -> tuple[float, np.ndarray]:
        target = np.array([1.0, 1.0, 1.0])
        final_err = np.linalg.norm(pos[-1] - target)
        overshoot_xy = np.maximum(pos - target, 0.0)[:, :2].max(axis=0)
        return final_err, overshoot_xy

    for name, pos in [
        ("baseline", pos_nom),
        ("abs-gpr", pos_abs),
        ("res-gpr", pos_res),
    ]:
        fe, ov = metrics(pos)
        print(f"{name}: final_err={fe:.4f}, overshoot_xy={ov}")


if __name__ == "__main__":
    main()
