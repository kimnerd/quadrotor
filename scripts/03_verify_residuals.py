import argparse
import json
import numpy as np
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.residual_slots_gp import ResidualYSlotsGP, ResidualR2GP


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="artifacts/slots_td.npz")
    p.add_argument("--gp-y", type=str, default=None)
    p.add_argument("--gp-r", type=str, default=None)
    args = p.parse_args()

    data = np.load(args.data)
    X_y, Y_y, X_r, Y_r = data["X_y"], data["Y_y"], data["X_r"], data["Y_r"]

    Xy_tr, Xy_te, Yy_tr, Yy_te = train_test_split(X_y, Y_y, test_size=0.2, random_state=0)
    Xr_tr, Xr_te, Yr_tr, Yr_te = train_test_split(X_r, Y_r, test_size=0.2, random_state=0)

    if args.gp_y and args.gp_r:
        gp_y = ResidualYSlotsGP.load(args.gp_y)
        gp_r = ResidualR2GP.load(args.gp_r)
    else:
        gp_y = ResidualYSlotsGP(input_dim=X_y.shape[1])
        gp_y.fit(Xy_tr, Yy_tr)
        gp_r = ResidualR2GP(input_dim=X_r.shape[1])
        gp_r.fit(Xr_tr, Yr_tr)

    dy, vy = gp_y.predict(Xy_te)
    dr, vr = gp_r.predict(Xr_te)

    err_y = Yy_te - dy
    err_r = Yr_te - dr
    rmse_y = float(np.sqrt(np.mean(err_y**2)))
    rmse_r = float(np.sqrt(np.mean(err_r**2)))
    r2_y = r2_score(Yy_te, dy, multioutput="raw_values")
    r2_r = r2_score(Yr_te, dr, multioutput="raw_values")
    calib_y = float(np.mean(np.abs(err_y / np.sqrt(vy))))
    calib_r = float(np.mean(np.abs(err_r / np.sqrt(vr))))
    corr_y = np.corrcoef(Yy_te.T, dy.T)[:9, 9:]
    corr_r = np.corrcoef(Yr_te.T, dr.T)[:3, 3:]

    report = {
        "rmse_y": rmse_y,
        "rmse_r": rmse_r,
        "r2_y": r2_y.tolist(),
        "r2_r": r2_r.tolist(),
        "calib_y": calib_y,
        "calib_r": calib_r,
        "corr_y": corr_y.tolist(),
        "corr_r": corr_r.tolist(),
    }
    with open("artifacts/residual_report.json", "w") as fh:
        json.dump(report, fh)

    pass_cond = (
        np.sum(r2_y > 0) >= 6 and r2_r.mean() > 0 and 0.7 <= calib_y <= 1.3 and 0.7 <= calib_r <= 1.3
    )
    if pass_cond:
        print("TEST PASS")
    else:
        print("TEST FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

