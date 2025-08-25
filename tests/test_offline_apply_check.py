from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np


def load_module():
    spec = importlib.util.spec_from_file_location(
        "offline_check", Path(__file__).resolve().parents[1] / "scripts" / "03b_offline_apply_check.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod


def test_offline_apply_check(tmp_path):
    mod = load_module()
    rng = np.random.default_rng(0)
    X_y = rng.normal(size=(20, 39))
    Y_y = rng.normal(size=(20, 9))
    X_r = rng.normal(size=(20, 33))
    Y_r = rng.normal(size=(20, 3))
    data_path = tmp_path / "data.npz"
    np.savez(data_path, X_y=X_y, Y_y=Y_y, X_r=X_r, Y_r=Y_r)

    class Dummy:
        def __init__(self, Y):
            self.Y = Y
        def predict(self, X):  # noqa: D401 - simple dummy
            return self.Y, np.ones_like(self.Y)

    gp_y = Dummy(Y_y)
    gp_r = Dummy(Y_r)
    out_json = tmp_path / "report.json"
    out_csv = tmp_path / "dims.csv"
    report = mod.offline_apply_check(
        data=str(data_path),
        gp_y="",
        gp_r="",
        out_json=str(out_json),
        out_csv=str(out_csv),
        skip_calib_check=True,
        gp_y_model=gp_y,
        gp_r_model=gp_r,
    )
    assert "rmse" in report and "calibration" in report
    assert "improve_dim" in report and "shuffle" in report
    assert report["rmse"]["y"]["improve"] > 0.0
    assert out_json.exists() and out_csv.exists()
    loaded = json.loads(out_json.read_text())
    assert "rmse" in loaded
