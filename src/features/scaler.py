"""
Z-score normalization of HRV feature vectors. Fit on training data only; persist for val/test and inference.
"""

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import yaml

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    StandardScaler = None  # type: ignore

try:
    import joblib
except ImportError:
    joblib = None


def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.isabs(config_path) and not os.path.isfile(config_path):
        root = Path(__file__).resolve().parents[2]
        config_path = root / config_path
    with open(config_path) as f:
        return yaml.safe_load(f)


def fit_scaler(
    hrv_features_train: np.ndarray,
    path: Optional[str] = None,
    config_path: str = "config.yaml",
) -> "StandardScaler":
    """
    Fit StandardScaler on training HRV features (N, n_features) and save to path.
    Path can be omitted to use config paths.scaler.
    """
    if StandardScaler is None:
        raise ImportError("scikit-learn is required for fit_scaler")
    if path is None or path == "":
        config = load_config(config_path)
        path = config.get("paths", {}).get("scaler", "models/artifacts/scaler.pkl")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    X = np.asarray(hrv_features_train, dtype=np.float64)
    # Handle NaNs: fit on non-NaN rows only, or use nan-friendly strategy
    mask = ~np.any(np.isnan(X), axis=1)
    if np.sum(mask) == 0:
        raise ValueError("No valid (non-NaN) rows in hrv_features_train")
    scaler = StandardScaler()
    scaler.fit(X[mask])
    if joblib is not None:
        joblib.dump(scaler, path)
    else:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(scaler, f)
    return scaler


def load_scaler(path: Optional[str] = None, config_path: str = "config.yaml") -> "StandardScaler":
    """Load scaler from path; path from config paths.scaler if not given."""
    if path is None or path == "":
        config = load_config(config_path)
        path = config.get("paths", {}).get("scaler", "models/artifacts/scaler.pkl")
    if joblib is not None:
        return joblib.load(path)
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def transform(features: np.ndarray, scaler: "StandardScaler") -> np.ndarray:
    """Transform features (any shape with last dim n_features); NaNs passed through."""
    X = np.asarray(features, dtype=np.float64)
    orig_shape = X.shape
    X_flat = X.reshape(-1, orig_shape[-1])
    nan_mask = np.any(np.isnan(X_flat), axis=1)
    out = np.full_like(X_flat, np.nan)
    if np.any(~nan_mask):
        out[~nan_mask] = scaler.transform(X_flat[~nan_mask])
    return out.reshape(orig_shape).astype(np.float32)
