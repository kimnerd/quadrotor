"""Gaussian Process Regression based inverse mapping from accel/orientation to rotor forces."""

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("numpy required. Install via 'pip install numpy'.") from exc

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover - optional dependency guard
    raise SystemExit(
        "scikit-learn required for GPR. Install via 'pip install scikit-learn'."
    ) from exc

import pickle
from dataclasses import dataclass


@dataclass
class InverseGPR:
    """Container holding a trained GPR model and feature/target scalers."""

    gpr: GaussianProcessRegressor
    x_scaler: StandardScaler
    y_scaler: StandardScaler


def train_inverse_gpr(X: np.ndarray, y: np.ndarray) -> InverseGPR:
    """Train a GaussianProcessRegressor with standardized inputs/outputs."""
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    Xs = x_scaler.fit_transform(X)
    ys = y_scaler.fit_transform(y)

    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, normalize_y=False)
    gpr.fit(Xs, ys)
    return InverseGPR(gpr, x_scaler, y_scaler)


def predict_forces(model: InverseGPR, X_query: np.ndarray) -> np.ndarray:
    """Predict rotor forces given query features."""
    Xs = model.x_scaler.transform(X_query)
    ys = model.gpr.predict(Xs)
    return model.y_scaler.inverse_transform(ys)


def save_model(model: InverseGPR, path: str) -> None:
    """Serialize GPR model to disk."""
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str) -> InverseGPR:
    """Load a serialized GPR model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
