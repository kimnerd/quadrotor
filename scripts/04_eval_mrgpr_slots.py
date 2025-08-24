import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from controller_D_slot_corrected import SlotCorrectedController
from control.controller_mrgpr_slots import MRGPBlockController
from models.residual_slots_gp import ResidualYSlotsGP, ResidualR2GP
from simulation import simulate


def _metrics(pos, x_refs, forces, conds, target, max_force):
    final_err = float(np.linalg.norm(pos[-1] - target))
    rms_err = float(np.sqrt(np.mean((pos - x_refs) ** 2)))
    sat = float(np.mean((forces >= max_force - 1e-6).any(axis=1)))
    cond_mean = float(np.mean(conds))
    cond_violate = float(np.mean(conds > 1e3))
    return final_err, rms_err, sat, cond_mean, cond_violate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gp-y", type=str, required=True)
    p.add_argument("--gp-r", type=str, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--hold", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--improve", type=float, default=0.1)
    args = p.parse_args()

    gp_y = ResidualYSlotsGP.load(args.gp_y)
    gp_r = ResidualR2GP.load(args.gp_r)

    rng = np.random.default_rng(args.seed)

    metrics = {"baseline": [], "mrgp": []}
    for _ in range(args.episodes):
        target = rng.uniform(0.6, 1.2, size=3)

        quad_base = SlotCorrectedController(gp_fx=None, gp_fr=None)
        pos_b, forces_b, _, _, x_refs_b, _, conds_b, _ = simulate(
            args.steps, target=target, hold_steps=args.hold, quad=quad_base
        )
        metrics["baseline"].append(
            _metrics(pos_b, x_refs_b, forces_b, conds_b, target, quad_base.max_force)
        )

        quad_mrgp = MRGPBlockController(gp_y=gp_y, gp_r=gp_r)
        pos_m, forces_m, _, _, x_refs_m, _, conds_m, _ = simulate(
            args.steps, target=target, hold_steps=args.hold, quad=quad_mrgp
        )
        metrics["mrgp"].append(
            _metrics(pos_m, x_refs_m, forces_m, conds_m, target, quad_mrgp.max_force)
        )

    metrics_avg = {}
    for key, vals in metrics.items():
        arr = np.array(vals)
        metrics_avg[key] = {
            "final_err": float(np.mean(arr[:, 0])),
            "rms_err": float(np.mean(arr[:, 1])),
            "sat_pct": float(np.mean(arr[:, 2])),
            "cond_mean": float(np.mean(arr[:, 3])),
            "cond_violate": float(np.mean(arr[:, 4])),
        }

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/eval_metrics.json", "w") as fh:
        json.dump(metrics_avg, fh)

    b = metrics_avg["baseline"]
    m = metrics_avg["mrgp"]
    improve = (b["rms_err"] - m["rms_err"]) / max(b["rms_err"], 1e-9)
    sat_increase = m["sat_pct"] - b["sat_pct"]
    cond_increase = m["cond_violate"] - b["cond_violate"]

    print("Baseline:", b)
    print("MRGP:", m)

    pass_cond = (
        improve >= args.improve
        and sat_increase <= 0.05
        and cond_increase <= 0.05
    )
    if pass_cond:
        print("EVAL PASS")
    else:
        print("EVAL FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

