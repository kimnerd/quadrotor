import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from simulation import finite_diff_y

class TestSlotsUtils(unittest.TestCase):
    def test_finite_diff_y_constant(self):
        m = 1.0
        dt = 0.1
        g = np.array([0.0, 0.0, -9.81])
        x = np.array([1.0, 2.0, 3.0])
        y = finite_diff_y(m, dt, g, x, x, x)
        self.assertTrue(np.allclose(y, -g, atol=1e-9))

if __name__ == "__main__":
    unittest.main()
