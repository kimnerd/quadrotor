import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import ConvergenceWarning

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.residual_slots_gp import ResidualYSlotsGP, ResidualR2GP


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="artifacts/slots_td.npz")
    p.add_argument("--out-y", type=str, default="artifacts/gp_y.pkl")
    p.add_argument("--out-r", type=str, default="artifacts/gp_r.pkl")
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    # speed/robustness knobs
    p.add_argument("--fast", action="store_true", help="iso RBF, restarts=0, optimizer=None, relax exit criteria")
    p.add_argument("--isotropic", action="store_true", help="use isotropic RBF instead of ARD")
    p.add_argument("--restarts", type=int, default=0)
    p.add_argument("--optimizer", type=str, default="fmin_l_bfgs_b", help="set to 'none' to skip hyperopt")
    p.add_argument("--alpha", type=float, default=1e-6)
    p.add_argument("--noise-cap", type=float, default=1e-2)
    p.add_argument("--subsample", type=int, default=0, help="randomly subsample N training points (0=all)")
    p.add_argument("--strict", action="store_true", help="fail run on poor RMSE")
    p.add_argument("--calibrate", action="store_true", help="temperature-scale predictive std on validation set")
    p.add_argument(
        "--calib-agg",
        type=str,
        default="median",
        choices=["median", "mean"],
        help="aggregation for |err|/std during calibration",
    )
    args = p.parse_args()

    data = np.load(args.data)
    X_y, Y_y, X_r, Y_r = data["X_y"], data["Y_y"], data["X_r"], data["Y_r"]

    if args.subsample and X_y.shape[0] > args.subsample:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(X_y.shape[0], args.subsample, replace=False)
        X_y, Y_y, X_r, Y_r = X_y[idx], Y_y[idx], X_r[idx], Y_r[idx]
        print(f"[GP] subsampled to {len(idx)} samples")

    Xy_tr, Xy_val, Yy_tr, Yy_val = train_test_split(
        X_y, Y_y, test_size=args.val_split, random_state=args.seed
    )
    Xr_tr, Xr_val, Yr_tr, Yr_val = train_test_split(
        X_r, Y_r, test_size=args.val_split, random_state=args.seed
    )

    # resolve mode
    isotropic = args.isotropic or args.fast
    restarts = 0 if args.fast else args.restarts
    optimizer = None if (args.fast or (args.optimizer.lower() == "none")) else args.optimizer
    alpha = args.alpha
    noise_cap = args.noise_cap

    gp_y = ResidualYSlotsGP(
        input_dim=X_y.shape[1],
        isotropic=isotropic,
        noise_cap=noise_cap,
        alpha=alpha,
        restarts=restarts,
        optimizer=optimizer,
    )
    gp_r = ResidualR2GP(
        input_dim=X_r.shape[1],
        isotropic=isotropic,
        noise_cap=noise_cap,
        alpha=alpha,
        restarts=restarts,
        optimizer=optimizer,
    )

    # fit with warning suppression; fallback on failure
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        try:
            gp_y.fit(Xy_tr, Yy_tr)
            gp_r.fit(Xr_tr, Yr_tr)
        except Exception as e:
            print(f"[GP] primary fit failed: {e}\n[GP] falling back to optimizer=None, isotropic")
            gp_y = ResidualYSlotsGP(
                input_dim=X_y.shape[1],
                isotropic=True,
                noise_cap=noise_cap,
                alpha=alpha,
                restarts=0,
                optimizer=None,
            )
            gp_r = ResidualR2GP(
                input_dim=X_r.shape[1],
                isotropic=True,
                noise_cap=noise_cap,
                alpha=alpha,
                restarts=0,
                optimizer=None,
            )
            gp_y.fit(Xy_tr, Yy_tr)
            gp_r.fit(Xr_tr, Yr_tr)

    if args.calibrate:
        gp_y.calibrate(Xy_val, Yy_val, agg=args.calib_agg)
        gp_r.calibrate(Xr_val, Yr_val, agg=args.calib_agg)

    dy_val, vy_val = gp_y.predict(Xy_val)
    dr_val, vr_val = gp_r.predict(Xr_val)

    if np.isnan(dy_val).any() or np.isnan(dr_val).any():
        raise SystemExit("NaN in predictions")

    err_y = Yy_val - dy_val
    err_r = Yr_val - dr_val
    rmse_y = float(np.sqrt(np.mean(err_y**2)))
    rmse_r = float(np.sqrt(np.mean(err_r**2)))
    rmse_y_dim = np.sqrt(np.mean(err_y**2, axis=0)).tolist()
    rmse_r_dim = np.sqrt(np.mean(err_r**2, axis=0)).tolist()
    calib_y = float(np.mean(np.abs(err_y / np.sqrt(vy_val))))
    calib_r = float(np.mean(np.abs(err_r / np.sqrt(vr_val))))

    os.makedirs(os.path.dirname(args.out_y), exist_ok=True)
    gp_y.save(args.out_y)
    gp_r.save(args.out_r)

    report = {
        "rmse_y": rmse_y,
        "rmse_r": rmse_r,
        "rmse_y_dim": rmse_y_dim,
        "rmse_r_dim": rmse_r_dim,
        "calib_y": calib_y,
        "calib_r": calib_r,
        "kernel_y": [str(m.kernel_) for m in gp_y.models],
        "kernel_r": [str(m.kernel_) for m in gp_r.models],
        "temps_y": gp_y.temp.tolist(),
        "temps_r": gp_r.temp.tolist(),
    }
    with open("artifacts/gp_report.json", "w") as fh:
        json.dump(report, fh)

    print(
        f"VAL Δy RMSE={rmse_y:.3f}  Δξ2 RMSE={rmse_r:.3f}  Calib_y={calib_y:.3f} Calib_r={calib_r:.3f}"
    )

    strict = args.strict and (not args.fast)
    if strict and (rmse_y > 0.6 or rmse_r > 0.25):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

