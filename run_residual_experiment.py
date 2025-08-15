from __future__ import annotations

"""Train and compare baseline, absolute MR-GPR, and residual MR-GPR."""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

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

    # 5) Plot trajectories and generate a GIF comparing paths
    def save_plots_and_gif() -> None:
        # 3-D trajectory plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for pos, label in [
            (pos_nom, "baseline"),
            (pos_abs, "abs-gpr"),
            (pos_res, "res-gpr"),
        ]:
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label=label)
        ax.scatter([1], [1], [1], c="r", label="target")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend()
        fig.tight_layout()
        plt.savefig("trajectory_compare.png")
        plt.close(fig)

        # GIF: top-down XY view using Matplotlib animation
        fig2, ax2 = plt.subplots()
        all_pos = np.vstack([pos_nom, pos_abs, pos_res])
        ax2.set_xlim(all_pos[:, 0].min() - 0.1, all_pos[:, 0].max() + 0.1)
        ax2.set_ylim(all_pos[:, 1].min() - 0.1, all_pos[:, 1].max() + 0.1)
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.scatter([1], [1], c="r", label="target")
        line_nom, = ax2.plot([], [], label="baseline")
        line_abs, = ax2.plot([], [], label="abs-gpr")
        line_res, = ax2.plot([], [], label="res-gpr")
        ax2.legend()

        def update(frame: int):
            line_nom.set_data(pos_nom[:frame, 0], pos_nom[:frame, 1])
            line_abs.set_data(pos_abs[:frame, 0], pos_abs[:frame, 1])
            line_res.set_data(pos_res[:frame, 0], pos_res[:frame, 1])
            return line_nom, line_abs, line_res

        anim = FuncAnimation(fig2, update, frames=pos_nom.shape[0], interval=50, blit=True)
        anim.save("trajectory_compare.gif", writer=PillowWriter(fps=20))
        plt.close(fig2)

    save_plots_and_gif()


if __name__ == "__main__":
    main()
