from __future__ import annotations

"""Residual Gaussian Process Regression model for rotor-force deltas."""

from typing import Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


class ResidualMRGPR:
    """Predict rotor force residuals (delta forces) using independent GPRs."""

    def __init__(self, input_dim: int):
        kernel = RBF(
            length_scale=np.ones(input_dim),
            length_scale_bounds=(1e-3, 1e3),
        ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
        # Maintain four independent models for simplicity/performance.
        self.models = [
            GaussianProcessRegressor(kernel=kernel, normalize_y=True) for _ in range(4)
        ]
        self.input_dim = input_dim

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fit all rotor residual models.

        Parameters
        ----------
        X: np.ndarray
            Training features with shape ``(N, input_dim)``.
        Y: np.ndarray
            Residual forces with shape ``(N, 4)`` where ``Y = f_true - f_nom``.
        """

        for i in range(4):
            self.models[i].fit(X, Y[:, i])

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict residual means and variances for each rotor.

        Parameters
        ----------
        X: np.ndarray
            Query features with shape ``(N, input_dim)``.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple ``(mean, var)`` each of shape ``(N, 4)`` where values are
            stacked across the four independent GPR models.
        """

        means, vars_ = [], []
        for i in range(4):
            m, v = self.models[i].predict(X, return_std=True)
            means.append(m)
            vars_.append(v**2)
        return np.stack(means, axis=1), np.stack(vars_, axis=1)
