"""
Dimensionality reduction (TR-2.3 support).

Phase 1 datasets are typically single-channel, so this module behaves as:
- 1 channel: pass-through
- >1 channel: PCA to 1 component (or configurable)

This is intentionally modular so Galea multi-channel can be reduced to a Sparrow-like single-channel signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from sklearn.decomposition import PCA
except ImportError:  # pragma: no cover
    PCA = None  # type: ignore

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None


@dataclass
class PCAReducer:
    n_components: int = 1
    _is_passthrough: bool = False
    _pca: Optional["PCA"] = None  # noqa: F821

    def fit(self, X: np.ndarray) -> "PCAReducer":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"PCAReducer.fit expects 2D array (n_samples, n_channels); got {X.shape}")
        n_channels = X.shape[1]
        if n_channels == 1:
            self._is_passthrough = True
            self._pca = None
            return self
        if PCA is None:
            raise ImportError("scikit-learn is required for PCA reduction")
        self._is_passthrough = False
        self._pca = PCA(n_components=self.n_components)
        self._pca.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"PCAReducer.transform expects 2D array (n_samples, n_channels); got {X.shape}")
        if self._is_passthrough:
            return X
        if self._pca is None:
            raise ValueError("PCAReducer is not fitted (call fit first), or passthrough was not selected")
        return self._pca.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_components": self.n_components,
            "is_passthrough": self._is_passthrough,
            "pca": self._pca,
        }
        if joblib is not None:
            joblib.dump(payload, path)
        else:
            import pickle
            with open(path, "wb") as f:
                pickle.dump(payload, f)

    @staticmethod
    def load(path: str) -> "PCAReducer":
        if joblib is not None:
            payload = joblib.load(path)
        else:
            import pickle
            with open(path, "rb") as f:
                payload = pickle.load(f)
        obj = PCAReducer(n_components=int(payload.get("n_components", 1)))
        obj._is_passthrough = bool(payload.get("is_passthrough", False))
        obj._pca = payload.get("pca", None)
        return obj

