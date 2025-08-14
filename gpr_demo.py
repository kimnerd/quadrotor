"""Demonstration of training a GPR inverse model and running quadrotor control."""

import numpy as np

from data_collection import collect_random_rotor_data
from gpr_inverse import train_inverse_gpr
from simulation import Quadrotor, simulate


def main() -> None:
    # 1. Collect random data
    X, y = collect_random_rotor_data(steps=2000)
    print(f"Collected dataset: X{X.shape}, y{y.shape}")

    # 2. Train GPR inverse model
    model = train_inverse_gpr(X, y)
    print("Trained GPR inverse model")

    # 3. Compare baseline control and GPR-based control
    target = np.array([1.0, 1.0, 1.0])

    steps = 400
    base_quad = Quadrotor()
    pos_base, *_ = simulate(steps=steps, target=target, quad=base_quad)
    err_base = np.linalg.norm(pos_base[-1] - target)
    print(f"Final position ideal allocation: {pos_base[-1]}, error {err_base:.3f}")

    gpr_quad = Quadrotor(gpr_model=model, use_gpr=True)
    pos_gpr, *_ = simulate(steps=steps, target=target, quad=gpr_quad)
    err_gpr = np.linalg.norm(pos_gpr[-1] - target)
    print(f"Final position with GPR control: {pos_gpr[-1]}, error {err_gpr:.3f}")


if __name__ == "__main__":
    main()
