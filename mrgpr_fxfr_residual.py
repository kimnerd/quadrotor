import numpy as np
from typing import Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

def _make_gp(input_dim: int) -> GaussianProcessRegressor:
    return GaussianProcessRegressor(
        kernel=RBF(length_scale=np.ones(input_dim), length_scale_bounds=(1e-3, 1e3))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1)),
        normalize_y=True,
    )


class ResidualFxGP:
    """phi -> Δx4 (R^3)"""

    def __init__(self, input_dim: int) -> None:
        self.models = [_make_gp(input_dim) for _ in range(3)]
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        Xs = self.scaler.fit_transform(X)
        for i in range(3):
            self.models[i].fit(Xs, Y[:, i])

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xs = self.scaler.transform(X)
        means, vars_ = [], []
        for i in range(3):
            m, s = self.models[i].predict(Xs, return_std=True)
            means.append(m)
            vars_.append(s ** 2)
        return np.stack(means, axis=1), np.stack(vars_, axis=1)


class ResidualFrGP:
    """phi -> Δξ2 (so(3) 벡터 R^3)"""

    def __init__(self, input_dim: int) -> None:
        self.models = [_make_gp(input_dim) for _ in range(3)]
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        Xs = self.scaler.fit_transform(X)
        for i in range(3):
            self.models[i].fit(Xs, Y[:, i])

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xs = self.scaler.transform(X)
        means, vars_ = [], []
        for i in range(3):
            m, s = self.models[i].predict(Xs, return_std=True)
            means.append(m)
            vars_.append(s ** 2)
        return np.stack(means, axis=1), np.stack(vars_, axis=1)
