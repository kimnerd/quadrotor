import argparse
import numpy as np
from simulation import Quadrotor, simulate
from data_collection import collect_random_rotor_data
from gpr_inverse import train_inverse_gpr
from data_collection_fxfr_residual import collect_fxfr_residual_data
from mrgpr_fxfr_residual import ResidualFxGP, ResidualFrGP
from controller_fxfr_residual import FxFrResidualQuad


def main(residual_data: str | None = None, abs_data: str | None = None) -> None:
    np.random.seed(0)

    if residual_data:
        data = np.load(residual_data)
        X_fx, Y_fx = data["X_fx"], data["Y_fx"]
        X_fr, Y_fr = data["X_fr"], data["Y_fr"]
    else:
        X_fx, Y_fx, X_fr, Y_fr = collect_fxfr_residual_data(
            runs=10, steps=150, hold_steps=30
        )
    gp_fx = ResidualFxGP(input_dim=X_fx.shape[1])
    gp_fx.fit(X_fx, Y_fx)
    gp_fr = ResidualFrGP(input_dim=X_fr.shape[1])
    gp_fr.fit(X_fr, Y_fr)

    if abs_data:
        data_abs = np.load(abs_data)
        X_abs, y_abs = data_abs["X_abs"], data_abs["y_abs"]
    else:
        X_abs, y_abs = collect_random_rotor_data(steps=200)
    model_abs = train_inverse_gpr(X_abs, y_abs)

    quad_nom = Quadrotor()
    pos_nom, *_ = simulate(200, hold_steps=400, quad=quad_nom)

    quad_abs = Quadrotor(gpr_model=model_abs, use_gpr=True)
    pos_abs, *_ = simulate(200, hold_steps=400, quad=quad_abs)

    quad_fxfr = FxFrResidualQuad(gp_fx=gp_fx, gp_fr=gp_fr)
    pos_fxfr, *_ = simulate(200, hold_steps=400, quad=quad_fxfr)

    def metrics(pos: np.ndarray) -> tuple[float, np.ndarray]:
        target = np.array([1.0, 1.0, 1.0])
        final_err = float(np.linalg.norm(pos[-1] - target))
        overshoot_xy = np.maximum(pos - target, 0.0)[:, :2].max(axis=0)
        return final_err, overshoot_xy

    for name, pos in [
        ("baseline", pos_nom),
        ("abs-gpr", pos_abs),
        ("fxfr-res", pos_fxfr),
    ]:
        fe, ov = metrics(pos)
        print(f"{name}: final_err={fe:.4f}, overshoot_xy={ov}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Fx/Fr residual GP experiment")
    parser.add_argument("--residual-data", type=str, default=None, help="npz file with X_fx,Y_fx,X_fr,Y_fr")
    parser.add_argument("--abs-data", type=str, default=None, help="npz file with X_abs,y_abs")
    args = parser.parse_args()
    main(residual_data=args.residual_data, abs_data=args.abs_data)
