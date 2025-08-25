from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.residual_slots_gp import ResidualYSlotsGP, ResidualR2GP


def offline_apply_check(
    *,
    data: str = "artifacts/slots_td.npz",
    gp_y: str = "artifacts/gp_y.pkl",
    gp_r: str = "artifacts/gp_r.pkl",
    out_json: str = "artifacts/offline_apply_report.json",
    out_csv: str = "artifacts/offline_apply_dims.csv",
    skip_calib_check: bool = False,
    fail_below_y: float = 0.10,
    fail_below_r: float = 0.10,
    shuffle_runs: int = 3,
    seed: int = 0,
    gp_y_model: Optional[ResidualYSlotsGP] = None,
    gp_r_model: Optional[ResidualR2GP] = None,
) -> dict:
    """Apply GP corrections offline and report RMSE improvements."""

    data_npz = np.load(data)
    X_y, Y_y = data_npz["X_y"], data_npz["Y_y"]
    X_r, Y_r = data_npz["X_r"], data_npz["Y_r"]

    gp_y_model = gp_y_model or ResidualYSlotsGP.load(gp_y)
    gp_r_model = gp_r_model or ResidualR2GP.load(gp_r)

    dy_hat, vy = gp_y_model.predict(X_y)
    dr_hat, vr = gp_r_model.predict(X_r)

    err_y = Y_y - dy_hat
    err_r = Y_r - dr_hat

    rmse_y_base = float(np.sqrt(np.mean(Y_y**2)))
    rmse_y_after = float(np.sqrt(np.mean(err_y**2)))
    improve_y = (rmse_y_base - rmse_y_after) / max(rmse_y_base, 1e-12)

    rmse_r_base = float(np.sqrt(np.mean(Y_r**2)))
    rmse_r_after = float(np.sqrt(np.mean(err_r**2)))
    improve_r = (rmse_r_base - rmse_r_after) / max(rmse_r_base, 1e-12)

    calib_y = float(np.mean(np.abs(err_y) / (np.sqrt(vy) + 1e-9)))
    calib_r = float(np.mean(np.abs(err_r) / (np.sqrt(vr) + 1e-9)))

    rmse_y_dim_base = np.sqrt(np.mean(Y_y**2, axis=0))
    rmse_y_dim_after = np.sqrt(np.mean(err_y**2, axis=0))
    improve_y_dim = (rmse_y_dim_base - rmse_y_dim_after) / (rmse_y_dim_base + 1e-12)

    rmse_r_dim_base = np.sqrt(np.mean(Y_r**2, axis=0))
    rmse_r_dim_after = np.sqrt(np.mean(err_r**2, axis=0))
    improve_r_dim = (rmse_r_dim_base - rmse_r_dim_after) / (rmse_r_dim_base + 1e-12)

    rng = np.random.default_rng(seed)
    sh_y, sh_r = [], []
    for _ in range(shuffle_runs):
        perm = rng.permutation(X_y.shape[0])
        dy_s, _ = gp_y_model.predict(X_y[perm])
        dr_s, _ = gp_r_model.predict(X_r[perm])
        err_y_s = Y_y - dy_s
        err_r_s = Y_r - dr_s
        rmse_y_s = float(np.sqrt(np.mean(err_y_s**2)))
        rmse_r_s = float(np.sqrt(np.mean(err_r_s**2)))
        sh_y.append((rmse_y_base - rmse_y_s) / max(rmse_y_base, 1e-12))
        sh_r.append((rmse_r_base - rmse_r_s) / max(rmse_r_base, 1e-12))

    shuffle_stats = {
        "y_improve_mean": float(np.mean(sh_y)),
        "y_improve_std": float(np.std(sh_y)),
        "r_improve_mean": float(np.mean(sh_r)),
        "r_improve_std": float(np.std(sh_r)),
    }

    report = {
        "rmse": {
            "y": {"base": rmse_y_base, "after": rmse_y_after, "improve": improve_y},
            "r": {"base": rmse_r_base, "after": rmse_r_after, "improve": improve_r},
        },
        "calibration": {"y": calib_y, "r": calib_r},
        "improve_dim": {"y": improve_y_dim.tolist(), "r": improve_r_dim.tolist()},
        "shuffle": shuffle_stats,
        "config": {
            "data": data,
            "gp_y": gp_y,
            "gp_r": gp_r,
            "out_json": out_json,
            "out_csv": out_csv,
            "skip_calib_check": skip_calib_check,
            "fail_below_y": fail_below_y,
            "fail_below_r": fail_below_r,
            "shuffle_runs": shuffle_runs,
            "seed": seed,
        },
    }

    os.makedirs(Path(out_json).parent, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(report, fh)

    try:
        import csv
        with open(out_csv, "w", newline="") as cf:
            w = csv.writer(cf)
            w.writerow(["target", "improve"])
            for i, val in enumerate(improve_y_dim):
                w.writerow([f"dy[{i}]", float(val)])
            for i, val in enumerate(improve_r_dim):
                w.writerow([f"dr[{i}]", float(val)])
    except Exception as e:
        print(f"[WARN] failed to write per-dim CSV: {e}")

    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="artifacts/slots_td.npz")
    p.add_argument("--gp-y", type=str, default="artifacts/gp_y.pkl")
    p.add_argument("--gp-r", type=str, default="artifacts/gp_r.pkl")
    p.add_argument("--out-json", type=str, default="artifacts/offline_apply_report.json")
    p.add_argument("--out-csv", type=str, default="artifacts/offline_apply_dims.csv")
    p.add_argument("--skip-calib-check", action="store_true")
    p.add_argument("--fail-below-y", type=float, default=0.10)
    p.add_argument("--fail-below-r", type=float, default=0.10)
    p.add_argument("--shuffle-runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    report = offline_apply_check(**vars(args))

    rm_y = report["rmse"]["y"]
    rm_r = report["rmse"]["r"]
    sh = report["shuffle"]
    print(
        f"Δy: RMSE {rm_y['base']:.3f} -> {rm_y['after']:.3f} (improve {100*rm_y['improve']:.1f}%)"
    )
    print(
        f"Δξ2: RMSE {rm_r['base']:.3f} -> {rm_r['after']:.3f} (improve {100*rm_r['improve']:.1f}%)"
    )
    print(
        f"Shuffle Δy improve {100*sh['y_improve_mean']:.1f}±{100*sh['y_improve_std']:.1f}%", 
        f"Δξ2 improve {100*sh['r_improve_mean']:.1f}±{100*sh['r_improve_std']:.1f}%",
    )
    print(
        f"Calibration: y={report['calibration']['y']:.3f} r={report['calibration']['r']:.3f}"
    )

    calib_ok = 0.7 <= report["calibration"]["y"] <= 1.3 and 0.7 <= report["calibration"]["r"] <= 1.3
    pass_cond = (
        rm_y["improve"] >= args.fail_below_y
        and rm_r["improve"] >= args.fail_below_r
        and (args.skip_calib_check or calib_ok)
    )
    if pass_cond:
        print("APPLY PASS")
    else:
        print("APPLY FAIL")
        reasons = []
        if rm_y["improve"] < args.fail_below_y:
            reasons.append(
                f"Δy improve {100*rm_y['improve']:.1f}% < target {100*args.fail_below_y:.1f}%"
            )
        if rm_r["improve"] < args.fail_below_r:
            reasons.append(
                f"Δξ2 improve {100*rm_r['improve']:.1f}% < target {100*args.fail_below_r:.1f}%"
            )
        if not args.skip_calib_check and not calib_ok:
            if not (0.7 <= report['calibration']['y'] <= 1.3):
                reasons.append(
                    f"Calib_y {report['calibration']['y']:.2f} not in [0.7,1.3]"
                )
            if not (0.7 <= report['calibration']['r'] <= 1.3):
                reasons.append(
                    f"Calib_r {report['calibration']['r']:.2f} not in [0.7,1.3]"
                )
        if reasons:
            print("Reasons:", "; ".join(reasons))
        sys.exit(1)


if __name__ == "__main__":
    main()
