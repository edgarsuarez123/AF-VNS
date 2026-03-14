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
    # Per-column fit: compute mean/var from non-NaN entries per feature
    # so partial rows (e.g. time-domain only) still contribute
    n_features = X.shape[1]
    means = np.zeros(n_features)
    vars_ = np.ones(n_features)
    n_samples_seen = 0
    any_valid = False
    for col in range(n_features):
        col_vals = X[:, col]
        valid = col_vals[~np.isnan(col_vals)]
        if len(valid) > 0:
            means[col] = valid.mean()
            vars_[col] = valid.var() if len(valid) > 1 else 1.0
            any_valid = True
            n_samples_seen = max(n_samples_seen, len(valid))
        else:
            means[col] = 0.0
            vars_[col] = 1.0
    if not any_valid:
        raise ValueError("No valid (non-NaN) values in any feature column of hrv_features_train")
    scaler = StandardScaler()
    scaler.mean_ = means
    scaler.var_ = vars_
    scaler.scale_ = np.sqrt(vars_)
    scaler.n_samples_seen_ = n_samples_seen
    scaler.n_features_in_ = n_features
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
    """Transform features (any shape with last dim n_features); NaNs passed through per-column."""
    X = np.asarray(features, dtype=np.float64)
    orig_shape = X.shape
    X_flat = X.reshape(-1, orig_shape[-1])
    out = np.full_like(X_flat, np.nan)
    n_features = X_flat.shape[1]
    for col in range(n_features):
        col_vals = X_flat[:, col]
        valid = ~np.isnan(col_vals)
        if np.any(valid):
            out[valid, col] = (col_vals[valid] - scaler.mean_[col]) / scaler.scale_[col]
    return out.reshape(orig_shape).astype(np.float32)
