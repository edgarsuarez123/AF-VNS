"""
F13 — Supervised Arousal State Classifier for tinnitus aVNS.

Wraps sklearn GradientBoostingClassifier (or LogisticRegression) on 5 EDA
features per 60s window.  Trained on WESAD stress vs. baseline labels.

Label convention:
  0 = in-band  (baseline / receptive state) → predict() returns True  → safe to stimulate
  1 = out-of-band (stress / hyperaroused)   → predict() returns False → hold stimulation

Features (must match tinnitus_precompute_arousal.FEATURE_NAMES):
  tonic_scl_mean, tonic_scl_std, phasic_mean, max_scr_amplitude, scr_rate

Usage:
    clf = ArousalClassifier()
    clf.fit(X_train, y_train)
    clf.save("models/checkpoints/arousal_classifier.pkl")

    clf2 = ArousalClassifier.load("models/checkpoints/arousal_classifier.pkl")
    in_band = clf2.predict(feature_vector)   # bool
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "tonic_scl_mean",
    "tonic_scl_std",
    "phasic_mean",
    "max_scr_amplitude",
    "scr_rate",
]
N_FEATURES = len(FEATURE_NAMES)


class ArousalClassifier:
    """Supervised EDA arousal state classifier.

    Parameters
    ----------
    model_type : str
        "gradient_boosting" (default) or "logistic_regression"
    config_path : str
        Path to config YAML (reads ``arousal_classifier`` section for defaults)
    """

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        config_path: str = "config_tinnitus.yaml",
    ) -> None:
        self._model_type = model_type
        self._config_path = config_path
        self._model = None
        self._scaler = None
        self._train_medians: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "ArousalClassifier":
        """Fit classifier on training feature matrix.

        Parameters
        ----------
        X_train : (N, N_FEATURES) float array
        y_train : (N,) int labels — 0=in-band, 1=out-of-band

        Returns self for chaining.
        """
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.int32).ravel()

        # Store column medians for NaN imputation at inference time
        self._train_medians = np.nanmedian(X, axis=0)
        X_imp = self._impute(X)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_imp)

        self._model = self._build_model()
        self._model.fit(X_scaled, y)
        self._is_fitted = True

        logger.info(
            "ArousalClassifier (%s) fitted on %d samples (%d in-band / %d out-of-band)",
            self._model_type, len(y), int((y == 0).sum()), int((y == 1).sum()),
        )
        return self

    def _build_model(self):
        """Instantiate the sklearn estimator from config."""
        cfg = self._read_ac_config()

        if self._model_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(C=1.0, max_iter=1000, random_state=42)

        # gradient_boosting (default)
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=int(cfg.get("n_estimators", 100)),
            max_depth=int(cfg.get("max_depth", 3)),
            learning_rate=float(cfg.get("learning_rate", 0.1)),
            subsample=float(cfg.get("subsample", 0.8)),
            random_state=42,
        )

    def _read_ac_config(self) -> dict:
        """Load arousal_classifier section from config YAML (best-effort)."""
        try:
            path = self._config_path
            if not os.path.isabs(path) and not os.path.isfile(path):
                path = str(Path(__file__).resolve().parents[2] / path)
            with open(path) as f:
                return yaml.safe_load(f).get("arousal_classifier", {})
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # NaN imputation
    # ------------------------------------------------------------------

    def _impute(self, X: np.ndarray) -> np.ndarray:
        """Replace NaN with per-column training median (0.0 if no median)."""
        X_out = X.copy()
        medians = self._train_medians if self._train_medians is not None else np.zeros(X.shape[1])
        for col in range(X.shape[1]):
            nan_mask = ~np.isfinite(X_out[:, col])
            if np.any(nan_mask):
                fill = float(medians[col]) if np.isfinite(medians[col]) else 0.0
                X_out[nan_mask, col] = fill
        return X_out

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, feature_vector: np.ndarray) -> bool:
        """Predict whether current EDA state is in-band (safe to stimulate).

        Parameters
        ----------
        feature_vector : (N_FEATURES,) float array — EDA features for current window

        Returns
        -------
        True  = in-band  (label 0, baseline/receptive) — stimulation permitted
        False = out-of-band (label 1, stressed/drowsy) — hold stimulation

        Raises
        ------
        RuntimeError if classifier has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "ArousalClassifier has not been fitted. "
                "Call fit() or load a saved checkpoint first."
            )
        x = np.asarray(feature_vector, dtype=np.float64).ravel().reshape(1, -1)
        x_imp = self._impute(x)
        x_scaled = self._scaler.transform(x_imp)
        pred = int(self._model.predict(x_scaled)[0])
        return pred == 0  # label 0 = in-band = True

    def predict_proba(self, feature_vector: np.ndarray) -> float:
        """Return P(in-band) — probability that current state is label 0.

        Parameters
        ----------
        feature_vector : (N_FEATURES,) float array

        Returns
        -------
        float in [0, 1]; higher = more likely in-band / safe to stimulate
        """
        if not self._is_fitted:
            raise RuntimeError(
                "ArousalClassifier has not been fitted. "
                "Call fit() or load a saved checkpoint first."
            )
        x = np.asarray(feature_vector, dtype=np.float64).ravel().reshape(1, -1)
        x_imp = self._impute(x)
        x_scaled = self._scaler.transform(x_imp)
        proba = self._model.predict_proba(x_scaled)[0]  # [P(label=0), P(label=1)]
        return float(proba[0])

    def score(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate on a feature matrix.

        Returns
        -------
        dict with keys: accuracy, auroc, f1, confusion_matrix
        """
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int32).ravel()

        X_imp = self._impute(X_arr)
        X_scaled = self._scaler.transform(X_imp)
        y_pred = self._model.predict(X_scaled)
        # Use P(label=1) = out-of-band as the AUROC positive-class score
        y_proba_ood = self._model.predict_proba(X_scaled)[:, 1]

        results: dict = {
            "accuracy": float(accuracy_score(y_arr, y_pred)),
            "f1": float(f1_score(y_arr, y_pred, average="binary", pos_label=0, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_arr, y_pred).tolist(),
        }
        if len(np.unique(y_arr)) > 1:
            results["auroc"] = float(roc_auc_score(y_arr, y_proba_ood))
        else:
            results["auroc"] = float("nan")

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save classifier state to .pkl via joblib (pickle fallback)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": self._model_type,
            "model": self._model,
            "scaler": self._scaler,
            "train_medians": self._train_medians,
            "is_fitted": self._is_fitted,
            "feature_names": FEATURE_NAMES,
        }
        try:
            import joblib
            joblib.dump(payload, path)
        except ImportError:
            import pickle
            with open(path, "wb") as f:
                pickle.dump(payload, f)
        logger.info("ArousalClassifier saved to %s", path)

    @staticmethod
    def load(path: str) -> "ArousalClassifier":
        """Load classifier from .pkl file."""
        try:
            import joblib
            payload = joblib.load(path)
        except ImportError:
            import pickle
            with open(path, "rb") as f:
                payload = pickle.load(f)

        obj = ArousalClassifier(model_type=payload.get("model_type", "gradient_boosting"))
        obj._model = payload.get("model")
        obj._scaler = payload.get("scaler")
        obj._train_medians = payload.get("train_medians")
        obj._is_fitted = bool(payload.get("is_fitted", False))
        return obj


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_arousal_classifier(
    config_path: str = "config_tinnitus.yaml",
    checkpoint_path: Optional[str] = None,
) -> ArousalClassifier:
    """Build ArousalClassifier from config, loading checkpoint if it exists.

    Parameters
    ----------
    config_path : str
        Path to tinnitus config YAML.
    checkpoint_path : str or None
        Override checkpoint path; defaults to config's arousal_classifier.checkpoint_path.

    Returns
    -------
    Fitted ArousalClassifier if checkpoint exists, otherwise unfitted instance.
    """
    resolved = config_path
    if not os.path.isabs(resolved) and not os.path.isfile(resolved):
        resolved = str(Path(__file__).resolve().parents[2] / resolved)
    with open(resolved) as f:
        cfg = yaml.safe_load(f)

    ac_cfg = cfg.get("arousal_classifier", {})
    model_type = ac_cfg.get("model_type", "gradient_boosting")

    if checkpoint_path is None:
        checkpoint_path = ac_cfg.get("checkpoint_path", "models/checkpoints/arousal_classifier.pkl")

    if not Path(checkpoint_path).is_absolute():
        root = Path(resolved).parent
        checkpoint_path = str(root / checkpoint_path)

    if Path(checkpoint_path).exists():
        logger.info("Loading ArousalClassifier from %s", checkpoint_path)
        return ArousalClassifier.load(checkpoint_path)

    logger.warning("No checkpoint at %s — returning unfitted classifier", checkpoint_path)
    return ArousalClassifier(model_type=model_type, config_path=config_path)
