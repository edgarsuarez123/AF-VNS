"""
Tests for F13 — ArousalClassifier (supervised EDA arousal state classifier).

Covers: fit/predict, save/load roundtrip, predict_proba, score metrics,
NaN handling, config loading, and classifier integration with ArousalGate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.models.arousal_classifier import ArousalClassifier, N_FEATURES, FEATURE_NAMES

CONFIG_PATH = "config_tinnitus.yaml"
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_features(n: int, label: int, rng=RNG) -> tuple:
    """Generate synthetic feature matrix + labels.

    label=0 (in-band): low tonic mean + low SCR rate
    label=1 (out-of-band): high tonic mean + high SCR rate
    """
    X = rng.normal(0, 1, size=(n, N_FEATURES)).astype(np.float32)
    if label == 0:
        X[:, 0] += 2.0   # tonic_scl_mean: low = ~2 µS
        X[:, 4] += 0.5   # scr_rate: low
    else:
        X[:, 0] += 8.0   # tonic_scl_mean: high stress response
        X[:, 4] += 5.0   # scr_rate: high
    return X, np.full(n, label, dtype=np.int32)


def _make_dataset(n_per_class: int = 50):
    """Create balanced synthetic training set."""
    X0, y0 = _synthetic_features(n_per_class, 0)
    X1, y1 = _synthetic_features(n_per_class, 1)
    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])
    return X, y


def _fitted_clf() -> ArousalClassifier:
    clf = ArousalClassifier(model_type="gradient_boosting", config_path=CONFIG_PATH)
    X, y = _make_dataset()
    clf.fit(X, y)
    return clf


# ---------------------------------------------------------------------------
# Test 1: fit then predict returns bool
# ---------------------------------------------------------------------------

def test_fit_predict():
    """fit() succeeds; predict() returns bool."""
    clf = _fitted_clf()
    fv = np.array([2.0, 0.1, 0.05, 0.1, 0.5], dtype=np.float32)  # low-stress
    result = clf.predict(fv)
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Test 2: save/load roundtrip preserves predictions
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    """Save then load produces identical predictions."""
    clf = _fitted_clf()
    path = str(tmp_path / "test_clf.pkl")
    clf.save(path)

    clf2 = ArousalClassifier.load(path)
    assert clf2._is_fitted

    fv = np.array([2.0, 0.1, 0.05, 0.1, 0.5], dtype=np.float32)
    assert clf.predict(fv) == clf2.predict(fv)
    assert abs(clf.predict_proba(fv) - clf2.predict_proba(fv)) < 1e-6


# ---------------------------------------------------------------------------
# Test 3: predict_proba in [0, 1]
# ---------------------------------------------------------------------------

def test_predict_proba_range():
    """predict_proba() returns float in [0, 1]."""
    clf = _fitted_clf()
    for _ in range(10):
        fv = RNG.random(N_FEATURES).astype(np.float32)
        p = clf.predict_proba(fv)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# Test 4: score returns expected metric keys
# ---------------------------------------------------------------------------

def test_score_returns_metrics():
    """score() returns dict with accuracy, auroc, f1, confusion_matrix."""
    clf = _fitted_clf()
    X, y = _make_dataset(n_per_class=20)
    metrics = clf.score(X, y)
    assert "accuracy" in metrics
    assert "auroc" in metrics
    assert "f1" in metrics
    assert "confusion_matrix" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Test 5: NaN features don't crash
# ---------------------------------------------------------------------------

def test_nan_handling():
    """Features with NaN values are imputed and don't raise exceptions."""
    clf = _fitted_clf()
    fv_nan = np.array([np.nan, 0.1, np.nan, 0.1, np.nan], dtype=np.float32)
    result = clf.predict(fv_nan)
    assert isinstance(result, bool)

    p = clf.predict_proba(fv_nan)
    assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# Test 6: config loads arousal_classifier section
# ---------------------------------------------------------------------------

def test_config_loads():
    """ArousalClassifier reads model_type from config without crashing."""
    clf = ArousalClassifier(model_type="gradient_boosting", config_path=CONFIG_PATH)
    cfg = clf._read_ac_config()
    assert isinstance(cfg, dict)
    assert cfg.get("model_type") == "gradient_boosting"


# ---------------------------------------------------------------------------
# Test 7: unfitted classifier raises RuntimeError on predict
# ---------------------------------------------------------------------------

def test_unfitted_raises():
    """predict() on unfitted classifier raises RuntimeError."""
    clf = ArousalClassifier()
    with pytest.raises(RuntimeError, match="fitted"):
        clf.predict(np.zeros(N_FEATURES, dtype=np.float32))


# ---------------------------------------------------------------------------
# Test 8: logistic regression variant also works
# ---------------------------------------------------------------------------

def test_logistic_regression_variant():
    """model_type='logistic_regression' fits and predicts correctly."""
    clf = ArousalClassifier(model_type="logistic_regression", config_path=CONFIG_PATH)
    X, y = _make_dataset()
    clf.fit(X, y)
    fv = np.array([2.0, 0.1, 0.05, 0.1, 0.5], dtype=np.float32)
    result = clf.predict(fv)
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Test 9: feature names match N_FEATURES
# ---------------------------------------------------------------------------

def test_feature_names_count():
    """FEATURE_NAMES has N_FEATURES entries."""
    assert len(FEATURE_NAMES) == N_FEATURES
    assert N_FEATURES == 5


# ---------------------------------------------------------------------------
# Test 10: ArousalGate with classifier overrides threshold
# ---------------------------------------------------------------------------

def test_gate_with_classifier_overrides_threshold():
    """ArousalGate with fitted classifier uses classifier, not sigma threshold."""
    from src.models.arousal_gate import ArousalGate

    clf = _fitted_clf()

    # Patch classifier to always return True (in-band)
    clf.predict = lambda fv: True

    gate = ArousalGate(config_path=CONFIG_PATH, classifier=clf)
    # No calibrate() needed with classifier

    # Feed some EDA samples so update() runs
    eda = np.full(240, 2.0, dtype=np.float64)  # 60s at 4 Hz
    gate.update(eda, fs=4.0)

    assert gate.is_in_band() is True
    assert gate.get_state()["using_classifier"] is True


# ---------------------------------------------------------------------------
# Test 11: ArousalGate without classifier still uses rule-based path
# ---------------------------------------------------------------------------

def test_gate_without_classifier_uses_threshold():
    """ArousalGate without classifier still raises if not calibrated."""
    from src.models.arousal_gate import ArousalGate

    gate = ArousalGate(config_path=CONFIG_PATH, classifier=None)
    # No calibrate(), no classifier
    with pytest.raises(RuntimeError):
        gate.is_in_band()
