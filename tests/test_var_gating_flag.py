from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from control.controller_mrgpr_slots import MRGPBlockController


class FakeGP:
    def __init__(self, out_dim: int, mean: float = 1.0, var: float = 100.0) -> None:
        self.out_dim = out_dim
        self.mean = mean
        self.var = var

    def predict(self, X: np.ndarray):
        n = X.shape[0]
        return (
            np.full((n, self.out_dim), self.mean),
            np.full((n, self.out_dim), self.var),
        )


def test_ignore_var_trust_overrides_gating() -> None:
    gp_y = FakeGP(9, mean=1.0, var=100.0)
    gp_r = FakeGP(3, mean=1.0, var=100.0)

    ctrl_on = MRGPBlockController(
        gp_y,
        gp_r,
        trust_scale_y=0.7,
        trust_scale_r=0.5,
        var_clip_y=0.1,
        var_clip_r=0.1,
        ignore_var_trust=True,
    )
    ctrl_off = MRGPBlockController(
        gp_y,
        gp_r,
        trust_scale_y=0.7,
        trust_scale_r=0.5,
        var_clip_y=0.1,
        var_clip_r=0.1,
        ignore_var_trust=False,
    )
    # Dummy predictions
    X_dummy = np.zeros((1, 1))
    dy, vy = gp_y.predict(X_dummy)
    dr, vr = gp_r.predict(X_dummy)

    wy_off = 0.7 * (1.0 / (1.0 + vy[0] / 0.1))
    wr_off = 0.5 * (1.0 / (1.0 + vr[0] / 0.1))
    wy_on = 0.7
    wr_on = 0.5

    assert np.allclose(wy_on, 0.7) and np.all(wy_off < 0.01)
    assert np.allclose(wr_on, 0.5) and np.all(wr_off < 0.01)

    # ensure controllers stored flag properly (sanity)
    assert ctrl_on.ignore_var_trust is True
    assert ctrl_off.ignore_var_trust is False
