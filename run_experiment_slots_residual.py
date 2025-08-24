import argparse
import argparse
import numpy as np
from simulation import simulate
from data_collection import collect_random_rotor_data
from gpr_inverse import train_inverse_gpr
from data_collection_slots_residual import collect_slots_residual_data
from residual_models import ResidualFxGP, ResidualFrGP
from controller_D_slot_corrected import SlotCorrectedController


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-data", type=str, default=None)
    parser.add_argument("--abs-data", type=str, default=None)
    parser.add_argument("--res-runs", type=int, default=30)
    parser.add_argument("--res-steps", type=int, default=250)
    parser.add_argument("--res-hold", type=int, default=50)
    parser.add_argument("--abs-steps", type=int, default=200)
    parser.add_argument("--sim-steps", type=int, default=200)
    parser.add_argument("--sim-hold", type=int, default=400)
    parser.add_argument(
        "--use-real-plant", type=lambda s: s.lower() == "true", default=True
    )
    args = parser.parse_args()

    if args.residual_data:
        try:
            data = np.load(args.residual_data)
            X_fx, Y_fx, X_fr, Y_fr = (
                data["X_fx"],
                data["Y_fx"],
                data["X_fr"],
                data["Y_fr"],
            )
        except FileNotFoundError:
            X_fx, Y_fx, X_fr, Y_fr = collect_slots_residual_data(
                runs=args.res_runs, steps=args.res_steps, hold_steps=args.res_hold
            )
            np.savez(
                args.residual_data,
                X_fx=X_fx,
                Y_fx=Y_fx,
                X_fr=X_fr,
                Y_fr=Y_fr,
            )
    else:
        X_fx, Y_fx, X_fr, Y_fr = collect_slots_residual_data(
            runs=args.res_runs, steps=args.res_steps, hold_steps=args.res_hold
        )

    gp_fx = ResidualFxGP(input_dim=X_fx.shape[1])
    gp_fx.fit(X_fx, Y_fx)
    gp_fr = ResidualFrGP(input_dim=X_fr.shape[1])
    gp_fr.fit(X_fr, Y_fr)

    if args.abs_data:
        try:
            data_abs = np.load(args.abs_data)
            X_abs, y_abs = data_abs["X_abs"], data_abs["y_abs"]
        except FileNotFoundError:
            X_abs, y_abs = collect_random_rotor_data(steps=args.abs_steps)
            np.savez(args.abs_data, X_abs=X_abs, y_abs=y_abs)
    else:
        X_abs, y_abs = collect_random_rotor_data(steps=args.abs_steps)
    model_abs = train_inverse_gpr(X_abs, y_abs)

    if args.use_real_plant:
        quad_base = SlotCorrectedController(gp_fx=None, gp_fr=None)
        pos_base, *_ = simulate(
            args.sim_steps, hold_steps=args.sim_hold, quad=quad_base
        )

        quad_abs = SlotCorrectedController(
            gp_fx=None,
            gp_fr=None,
            gpr_model=model_abs,
            use_gpr=True,
        )
        pos_abs, *_ = simulate(args.sim_steps, hold_steps=args.sim_hold, quad=quad_abs)

        quad_slot = SlotCorrectedController(gp_fx=gp_fx, gp_fr=gp_fr)
        pos_slot, *_ = simulate(args.sim_steps, hold_steps=args.sim_hold, quad=quad_slot)
    else:
        from plants_ab import NominalQuadrotor

        quad_base = NominalQuadrotor()
        pos_base, *_ = simulate(args.sim_steps, hold_steps=args.sim_hold, quad=quad_base)

        quad_abs = NominalQuadrotor(gpr_model=model_abs, use_gpr=True)
        pos_abs, *_ = simulate(args.sim_steps, hold_steps=args.sim_hold, quad=quad_abs)

        quad_slot = SlotCorrectedController(gp_fx=gp_fx, gp_fr=gp_fr)
        pos_slot, *_ = simulate(args.sim_steps, hold_steps=args.sim_hold, quad=quad_slot)

    target = np.array([1.0, 1.0, 1.0])

    def metrics(pos: np.ndarray):
        final_err = float(np.linalg.norm(pos[-1] - target))
        overshoot_xy = np.maximum(pos - target, 0.0)[:, :2].max(axis=0)
        return final_err, overshoot_xy

    for name, pos in [
        ("baseline", pos_base),
        ("abs-gpr", pos_abs),
        ("slot-corr", pos_slot),
    ]:
        fe, ov = metrics(pos)
        print(f"{name}: final_err={fe:.4f}, overshoot_xy={ov}")


if __name__ == "__main__":
    main()
