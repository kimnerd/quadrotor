import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data.build_slots_td import build_slots_td


def test_td_shapes():
    X_y, Y_y, X_r, Y_r = build_slots_td(runs=2, steps=80, hold_steps=20, seed=0)
    assert X_y.shape[1] == 39 and Y_y.shape[1] == 9
    assert X_r.shape[1] == 33 and Y_r.shape[1] == 3
    assert X_y.shape[0] == Y_y.shape[0]
    assert X_r.shape[0] == Y_r.shape[0]
    assert np.isfinite(X_y).all() and np.isfinite(Y_y).all()
    assert np.isfinite(X_r).all() and np.isfinite(Y_r).all()
