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
    p.add_argument("--report", action="store_true", help="pretty-print summary and failure reasons")
    p.add_argument("--out-json", type=str, default="artifacts/residual_report.json")
    p.add_argument("--out-csv", type=str, default="artifacts/residual_dims.csv", help="per-dimension RMSE/R2")
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
    calib_y = float(np.mean(np.abs(err_y / (np.sqrt(vy) + 1e-9))))
    calib_r = float(np.mean(np.abs(err_r / (np.sqrt(vr) + 1e-9))))

    def _safe_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        b = np.asarray(b)
        out = np.zeros((a.shape[0], b.shape[0]))
        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                ai = a[i]
                bj = b[j]
                sa = np.std(ai)
                sb = np.std(bj)
                if sa < 1e-12 or sb < 1e-12:
                    out[i, j] = 0.0
                else:
                    c = np.corrcoef(ai, bj)[0, 1]
                    if not np.isfinite(c):
                        c = 0.0
                    out[i, j] = c
        return out

    corr_y = _safe_corr(Yy_te.T, dy.T)
    corr_r = _safe_corr(Yr_te.T, dr.T)

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
    with open(args.out_json, "w") as fh:
        json.dump(report, fh)

    pass_cond = (
        np.sum(r2_y > 0) >= 6 and r2_r.mean() > 0 and 0.7 <= calib_y <= 1.3 and 0.7 <= calib_r <= 1.3
    )
    if args.report:
        good_y = int(np.sum(r2_y > 0))
        mean_r2r = float(np.mean(r2_r))
        print("\n=== Residual Verification (summary) ===")
        print(f"Δy RMSE={rmse_y:.3f}  Δξ2 RMSE={rmse_r:.3f}  Calib_y={calib_y:.3f}  Calib_r={calib_r:.3f}")
        print(f"R2: Δy >0 dims = {good_y}/9,  mean Δξ2 R2 = {mean_r2r:.3f}")
        if not pass_cond:
            reasons = []
            if good_y < 6:
                reasons.append(f"Δy R2>0 dims {good_y} < 6")
            if mean_r2r <= 0:
                reasons.append(f"mean Δξ2 R2 {mean_r2r:.3f} ≤ 0")
            if not (0.7 <= calib_y <= 1.3):
                reasons.append(f"Calib_y {calib_y:.2f} not in [0.7,1.3]")
            if not (0.7 <= calib_r <= 1.3):
                reasons.append(f"Calib_r {calib_r:.2f} not in [0.7,1.3]")
            if reasons:
                print("Reasons:", "; ".join(reasons))

    try:
        import csv
        rmse_y_dim = np.sqrt(np.mean((Yy_te - dy) ** 2, axis=0))
        rmse_r_dim = np.sqrt(np.mean((Yr_te - dr) ** 2, axis=0))
        with open(args.out_csv, "w", newline="") as cf:
            w = csv.writer(cf)
            w.writerow(["target", "rmse", "r2"])
            for i in range(9):
                w.writerow([f"dy[{i}]", float(rmse_y_dim[i]), float(r2_y[i])])
            for i in range(3):
                w.writerow([f"dr[{i}]", float(rmse_r_dim[i]), float(r2_r[i])])
    except Exception as e:
        print(f"[WARN] failed to write per-dim CSV: {e}")
    if pass_cond:
        print("TEST PASS")
    else:
        print("TEST FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

