"""GP models for slot-target residuals."""

from __future__ import annotations

import pickle
from typing import Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler


def _make_gp(input_dim: int) -> GaussianProcessRegressor:
    return GaussianProcessRegressor(
        kernel=RBF(length_scale=np.ones(input_dim), length_scale_bounds=(1e-3, 1e3))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1)),
        normalize_y=True,
    )


class ResidualYSlotsGP:
    """Input ``X_y`` (39) -> Output Δy slots (9)."""

    def __init__(self, input_dim: int = 39):
        self.models = [_make_gp(input_dim) for _ in range(9)]
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        assert X.shape[0] == Y.shape[0]
        assert X.shape[1] == 39 and Y.shape[1] == 9
        Xs = self.scaler.fit_transform(X)
        for i in range(9):
            self.models[i].fit(Xs, Y[:, i])

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xs = self.scaler.transform(X)
        means, vars_ = [], []
        for i in range(9):
            m, s = self.models[i].predict(Xs, return_std=True)
            means.append(m)
            vars_.append(s**2)
        return np.stack(means, axis=1), np.stack(vars_, axis=1)

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(dict(models=self.models, scaler=self.scaler), fh)

    @staticmethod
    def load(path: str) -> "ResidualYSlotsGP":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        obj = ResidualYSlotsGP()
        obj.models = data["models"]
        obj.scaler = data["scaler"]
        return obj


class ResidualR2GP:
    """Input ``X_r`` (33) -> Output Δξ2 (3)."""

    def __init__(self, input_dim: int = 33):
        self.models = [_make_gp(input_dim) for _ in range(3)]
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        assert X.shape[0] == Y.shape[0]
        assert X.shape[1] == 33 and Y.shape[1] == 3
        Xs = self.scaler.fit_transform(X)
        for i in range(3):
            self.models[i].fit(Xs, Y[:, i])

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xs = self.scaler.transform(X)
        means, vars_ = [], []
        for i in range(3):
            m, s = self.models[i].predict(Xs, return_std=True)
            means.append(m)
            vars_.append(s**2)
        return np.stack(means, axis=1), np.stack(vars_, axis=1)

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(dict(models=self.models, scaler=self.scaler), fh)

    @staticmethod
    def load(path: str) -> "ResidualR2GP":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        obj = ResidualR2GP()
        obj.models = data["models"]
        obj.scaler = data["scaler"]
        return obj

