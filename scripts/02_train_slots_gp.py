import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.residual_slots_gp import ResidualYSlotsGP, ResidualR2GP


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="artifacts/slots_td.npz")
    p.add_argument("--out-y", type=str, default="artifacts/gp_y.pkl")
    p.add_argument("--out-r", type=str, default="artifacts/gp_r.pkl")
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    data = np.load(args.data)
    X_y, Y_y, X_r, Y_r = data["X_y"], data["Y_y"], data["X_r"], data["Y_r"]

    Xy_tr, Xy_val, Yy_tr, Yy_val = train_test_split(
        X_y, Y_y, test_size=args.val_split, random_state=args.seed
    )
    Xr_tr, Xr_val, Yr_tr, Yr_val = train_test_split(
        X_r, Y_r, test_size=args.val_split, random_state=args.seed
    )

    gp_y = ResidualYSlotsGP(input_dim=X_y.shape[1])
    gp_y.fit(Xy_tr, Yy_tr)
    gp_r = ResidualR2GP(input_dim=X_r.shape[1])
    gp_r.fit(Xr_tr, Yr_tr)

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
    }
    with open("artifacts/gp_report.json", "w") as fh:
        json.dump(report, fh)

    print(
        f"VAL Δy RMSE={rmse_y:.3f}  Δξ2 RMSE={rmse_r:.3f}  Calib_y={calib_y:.3f} Calib_r={calib_r:.3f}"
    )

    if rmse_y > 0.6 or rmse_r > 0.25:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

