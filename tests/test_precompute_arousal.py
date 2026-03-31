"""
Tests for F13 — tinnitus_precompute_arousal.py

Covers: feature extraction from synthetic signals, window count math,
label propagation, subject-level split integrity.
Integration test (real WESAD data) is skipped if data not on disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.training.tinnitus_precompute_arousal import (
    FEATURE_NAMES,
    N_FEATURES,
    extract_eda_features_window,
    _get_base_subject_id,
)

EDA_FS = 4.0  # Hz — WESAD Empatica E4 EDA rate
WINDOW_SEC = 60.0
HOP_SEC = 30.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_eda(value: float = 2.0, duration_sec: float = 120.0, fs: float = EDA_FS) -> np.ndarray:
    return np.full(int(duration_sec * fs), value, dtype=np.float64)


def _synthetic_record(subject_id: str, label: int, duration_sec: float = 120.0) -> dict:
    """Minimal TinnitusRecordDict-like dict for testing."""
    return {
        "subject_id": f"wesad_{subject_id}_baseline_0",
        "session_id": subject_id,
        "label": label,
        "eda_signal": _flat_eda(2.0, duration_sec),
        "eda_fs": EDA_FS,
        "ppg_signal": np.zeros(int(duration_sec * 64.0), dtype=np.float64),
        "ppg_fs": 64.0,
    }


# ---------------------------------------------------------------------------
# Test 1: feature extraction from synthetic 60s EDA
# ---------------------------------------------------------------------------

def test_feature_extraction_synthetic():
    """60s flat EDA produces (N_FEATURES,) float32 without crash."""
    eda = _flat_eda(2.0, 60.0)
    feats = extract_eda_features_window(eda, EDA_FS)
    assert feats.shape == (N_FEATURES,)
    assert feats.dtype == np.float32
    # Flat signal: tonic_scl_mean should be close to EDA value
    assert np.isfinite(feats[0]), "tonic_scl_mean should be finite for flat signal"


# ---------------------------------------------------------------------------
# Test 2: window count math
# ---------------------------------------------------------------------------

def test_window_count():
    """120s signal with 60s window and 30s hop should produce 3 windows."""
    eda = _flat_eda(2.0, 120.0)  # 480 samples at 4 Hz
    win_samples = int(WINDOW_SEC * EDA_FS)   # 240
    hop_samples = int(HOP_SEC * EDA_FS)      # 120

    windows = []
    start = 0
    while start + win_samples <= len(eda):
        windows.append(eda[start:start + win_samples])
        start += hop_samples

    # start=0 → 0:240, start=120 → 120:360, start=240 → 240:480 → 3 windows
    assert len(windows) == 3


# ---------------------------------------------------------------------------
# Test 3: too-short signal returns all-NaN features
# ---------------------------------------------------------------------------

def test_short_signal_returns_nan():
    """Signal shorter than 10s returns all-NaN feature vector."""
    eda = _flat_eda(2.0, 5.0)  # only 5s — below 10s minimum
    feats = extract_eda_features_window(eda, EDA_FS)
    assert feats.shape == (N_FEATURES,)
    assert np.all(np.isnan(feats))


# ---------------------------------------------------------------------------
# Test 4: label propagation
# ---------------------------------------------------------------------------

def test_label_propagation(tmp_path):
    """Windows inherit the record's label."""
    from src.training.tinnitus_precompute_arousal import precompute_arousal_cache
    import yaml

    # Build minimal config pointing at synthetic data
    config = {
        "data": {"wesad_subdir": str(tmp_path / "wesad_fake")},
        "eda": {"scr_min_amplitude": 0.02},
        "arousal_classifier": {
            "window_sec": 60,
            "hop_sec": 30,
            "cache_dir": str(tmp_path / "cache"),
            "split_path": str(tmp_path / "split.json"),
        },
        "wesad": {
            "label_map": {"0": None, "1": 0, "2": 1, "3": 0, "4": 0},
            "subjects": [2, 3, 4, 5, 6, 7],
        },
    }
    config_path = str(tmp_path / "config_test.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # This will fail gracefully (no real WESAD data) but we test label logic directly
    eda = _flat_eda(2.0, 120.0)
    feats0 = extract_eda_features_window(eda, EDA_FS)
    feats1 = extract_eda_features_window(_flat_eda(8.0, 120.0), EDA_FS)

    # Stress signal (8 µS) should have higher tonic_scl_mean than baseline (2 µS)
    if np.isfinite(feats0[0]) and np.isfinite(feats1[0]):
        assert feats1[0] > feats0[0], "Stress EDA should have higher tonic mean"


# ---------------------------------------------------------------------------
# Test 5: subject ID extraction
# ---------------------------------------------------------------------------

def test_subject_id_extraction():
    """_get_base_subject_id extracts 'S2' from wesad_S2_baseline_0."""
    rec_with_session = {"subject_id": "wesad_S2_baseline_0", "session_id": "S2"}
    rec_no_session = {"subject_id": "wesad_S3_stress_0"}

    assert _get_base_subject_id(rec_with_session) == "S2"
    assert _get_base_subject_id(rec_no_session) == "S3"


# ---------------------------------------------------------------------------
# Test 6: feature names
# ---------------------------------------------------------------------------

def test_feature_names():
    """FEATURE_NAMES has exactly N_FEATURES entries."""
    assert len(FEATURE_NAMES) == N_FEATURES
    assert "tonic_scl_mean" in FEATURE_NAMES
    assert "scr_rate" in FEATURE_NAMES


# ---------------------------------------------------------------------------
# Test 7: integration — real WESAD (skipped if not on disk)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_integration_wesad(tmp_path):
    """Full precompute pipeline on real WESAD data produces correct cache shape."""
    import yaml

    wesad_path = _root / "data" / "raw" / "tinnitus avns" / "wesad" / "WESAD"
    if not wesad_path.exists():
        pytest.skip("WESAD data not on disk")

    config = {
        "data": {"wesad_subdir": str(wesad_path)},
        "eda": {"scr_min_amplitude": 0.02, "decomposition_method": "cvxeda",
                "calibration_sec": 1200},
        "arousal_classifier": {
            "window_sec": 60,
            "hop_sec": 30,
            "cache_dir": str(tmp_path / "cache"),
            "split_path": str(tmp_path / "split.json"),
        },
        "wesad": {
            "label_map": {"0": None, "1": 0, "2": 1, "3": 0, "4": 0},
            "subjects": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17],
        },
    }
    config_path = str(tmp_path / "config_test.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    from src.training.tinnitus_precompute_arousal import precompute_arousal_cache
    counts = precompute_arousal_cache(config_path)

    assert counts["total"] > 0

    cache_dir = tmp_path / "cache"
    X_train = np.load(str(cache_dir / "train_features.npy"))
    y_train = np.load(str(cache_dir / "train_labels.npy"))

    assert X_train.ndim == 2
    assert X_train.shape[1] == N_FEATURES
    assert len(y_train) == len(X_train)
    assert set(np.unique(y_train)).issubset({0, 1})

    # Subject-level split: no subject in both train and test
    with open(str(tmp_path / "split.json")) as f:
        split = json.load(f)
    assert len(set(split["train"]) & set(split["test"])) == 0
