import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.residual_slots_gp import ResidualYSlotsGP, ResidualR2GP


def test_gp_save_load(tmp_path):
    Xy = np.random.randn(30, 39)
    Yy = np.random.randn(30, 9)
    Xr = np.random.randn(30, 33)
    Yr = np.random.randn(30, 3)

    gp_y = ResidualYSlotsGP()
    gp_y.fit(Xy, Yy)
    m1, v1 = gp_y.predict(Xy[:5])
    path_y = tmp_path / "gp_y.pkl"
    gp_y.save(path_y)
    gp_y2 = ResidualYSlotsGP.load(path_y)
    m2, v2 = gp_y2.predict(Xy[:5])
    assert np.allclose(m1, m2)
    assert np.allclose(v1, v2)

    gp_r = ResidualR2GP()
    gp_r.fit(Xr, Yr)
    mr1, vr1 = gp_r.predict(Xr[:5])
    path_r = tmp_path / "gp_r.pkl"
    gp_r.save(path_r)
    gp_r2 = ResidualR2GP.load(path_r)
    mr2, vr2 = gp_r2.predict(Xr[:5])
    assert np.allclose(mr1, mr2)
    assert np.allclose(vr1, vr2)
