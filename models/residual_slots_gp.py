"""GP models for slot-target residuals."""

from __future__ import annotations

import pickle
from typing import Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from typing import Optional


def _make_gp(
    input_dim: int,
    *,
    isotropic: bool = False,
    noise_cap: float = 1e-2,
    alpha: float = 1e-6,
    restarts: int = 0,
    optimizer: Optional[str] = "fmin_l_bfgs_b",
) -> GaussianProcessRegressor:
    """Factory for a basic RBF+White GP regressor."""
    ls = 1.0 if isotropic else np.ones(input_dim)
    ls_bounds = (1e-2, 1e2)
    kernel = RBF(length_scale=ls, length_scale_bounds=ls_bounds) + \
             WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, noise_cap))
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=restarts,
        optimizer=optimizer,
        random_state=0,
    )


class ResidualYSlotsGP:
    """Input ``X_y`` (39) -> Output Δy slots (9)."""

    def __init__(
        self,
        input_dim: int = 39,
        *,
        isotropic: bool = False,
        noise_cap: float = 1e-2,
        alpha: float = 1e-6,
        restarts: int = 0,
        optimizer: Optional[str] = "fmin_l_bfgs_b",
    ):
        self.models = [
            _make_gp(
                input_dim,
                isotropic=isotropic,
                noise_cap=noise_cap,
                alpha=alpha,
                restarts=restarts,
                optimizer=optimizer,
            )
            for _ in range(9)
        ]
        self.scaler = StandardScaler()
        # per-output temperature scale for predictive std
        self.temp = np.ones(9, dtype=float)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        assert X.shape[0] == Y.shape[0]
        assert X.shape[1] == 39 and Y.shape[1] == 9
        Xs = self.scaler.fit_transform(X)
        for i in range(9):
            self.models[i].fit(Xs, Y[:, i])

    def _predict_raw(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xs = self.scaler.transform(X)
        means, vars_ = [], []
        for i in range(9):
            m, s = self.models[i].predict(Xs, return_std=True)
            means.append(m)
            vars_.append(s**2)
        return np.stack(means, axis=1), np.stack(vars_, axis=1)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m, v = self._predict_raw(X)
        v_scaled = v * (self.temp[None, :] ** 2)
        return m, v_scaled

    def calibrate(self, X_val: np.ndarray, Y_val: np.ndarray, agg: str = "median") -> None:
        m, v = self._predict_raw(X_val)
        std = np.sqrt(np.maximum(v, 1e-12))
        ratio = np.abs(Y_val - m) / std
        if agg == "mean":
            t = np.mean(ratio, axis=0)
        else:
            t = np.median(ratio, axis=0)
        self.temp = np.clip(t, 0.5, 2.0).astype(float)

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(dict(models=self.models, scaler=self.scaler, temp=self.temp), fh)

    @staticmethod
    def load(path: str) -> "ResidualYSlotsGP":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        obj = ResidualYSlotsGP()
        obj.models = data["models"]
        obj.scaler = data["scaler"]
        obj.temp = data.get("temp", np.ones(9, dtype=float))
        return obj


class ResidualR2GP:
    """Input ``X_r`` (33) -> Output Δξ2 (3)."""

    def __init__(
        self,
        input_dim: int = 33,
        *,
        isotropic: bool = False,
        noise_cap: float = 1e-2,
        alpha: float = 1e-6,
        restarts: int = 0,
        optimizer: Optional[str] = "fmin_l_bfgs_b",
    ):
        self.models = [
            _make_gp(
                input_dim,
                isotropic=isotropic,
                noise_cap=noise_cap,
                alpha=alpha,
                restarts=restarts,
                optimizer=optimizer,
            )
            for _ in range(3)
        ]
        self.scaler = StandardScaler()
        self.temp = np.ones(3, dtype=float)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        assert X.shape[0] == Y.shape[0]
        assert X.shape[1] == 33 and Y.shape[1] == 3
        Xs = self.scaler.fit_transform(X)
        for i in range(3):
            self.models[i].fit(Xs, Y[:, i])

    def _predict_raw(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xs = self.scaler.transform(X)
        means, vars_ = [], []
        for i in range(3):
            m, s = self.models[i].predict(Xs, return_std=True)
            means.append(m)
            vars_.append(s**2)
        return np.stack(means, axis=1), np.stack(vars_, axis=1)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m, v = self._predict_raw(X)
        v_scaled = v * (self.temp[None, :] ** 2)
        return m, v_scaled

    def calibrate(self, X_val: np.ndarray, Y_val: np.ndarray, agg: str = "median") -> None:
        m, v = self._predict_raw(X_val)
        std = np.sqrt(np.maximum(v, 1e-12))
        ratio = np.abs(Y_val - m) / std
        if agg == "mean":
            t = np.mean(ratio, axis=0)
        else:
            t = np.median(ratio, axis=0)
        self.temp = np.clip(t, 0.5, 2.0).astype(float)

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(dict(models=self.models, scaler=self.scaler, temp=self.temp), fh)

    @staticmethod
    def load(path: str) -> "ResidualR2GP":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        obj = ResidualR2GP()
        obj.models = data["models"]
        obj.scaler = data["scaler"]
        obj.temp = data.get("temp", np.ones(3, dtype=float))
        return obj

