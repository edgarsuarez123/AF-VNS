"""Tests for phase_evaluate.py — phase detector evaluation (S-27)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_cache(tmp_dir: Path, split: str = "test", n_windows: int = 100):
    """Create fake phase cache files for testing.

    Returns cache directory.
    """
    cache_dir = tmp_dir / "cache_test"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ECG: (n_windows, 1, 500)
    ecg = np.random.randn(n_windows, 1, 500).astype(np.float32)
    np.save(cache_dir / f"{split}_ecg.npy", ecg)

    # Labels: (n_windows, 10) — frame-level diastole/exhalation
    # Insert some NaN for realistic scenario
    diastole = np.random.choice([0, 1], size=(n_windows, 10)).astype(np.float32)
    diastole[0:5, :] = np.nan  # Some windows have no labels
    np.save(cache_dir / f"{split}_diastole.npy", diastole)

    exhalation = np.random.choice([0, 1], size=(n_windows, 10)).astype(np.float32)
    exhalation[5:10, :] = np.nan
    np.save(cache_dir / f"{split}_exhalation.npy", exhalation)

    # Quality mask
    quality = np.ones((n_windows, 10), dtype=bool)
    quality[0:5, :] = False
    np.save(cache_dir / f"{split}_quality.npy", quality)

    return cache_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPhaseEvaluation:

    def test_compute_frame_metrics_balanced(self):
        """Compute metrics on balanced binary labels."""
        from src.training.phase_evaluate import _compute_frame_metrics

        labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        probs = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.1, 0.85, 0.75])
        metrics = _compute_frame_metrics(labels, probs)

        assert metrics["n_frames"] == 8
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert "confusion_matrix" in metrics

    def test_compute_frame_metrics_with_nan(self):
        """Metrics should exclude NaN labels."""
        from src.training.phase_evaluate import _compute_frame_metrics

        labels = np.array([0.0, 1.0, np.nan, 1.0, 0.0, np.nan, 1.0])
        probs = np.array([0.2, 0.8, 0.5, 0.9, 0.1, 0.5, 0.85])
        metrics = _compute_frame_metrics(labels, probs)

        assert metrics["n_frames"] == 5  # 7 - 2 NaN
        assert not np.isnan(metrics["accuracy"])

    def test_compute_frame_metrics_all_nan(self):
        """All NaN labels → NaN metrics."""
        from src.training.phase_evaluate import _compute_frame_metrics

        labels = np.full(10, np.nan)
        probs = np.random.rand(10)
        metrics = _compute_frame_metrics(labels, probs)

        assert metrics["n_frames"] == 0
        assert np.isnan(metrics["accuracy"])
        assert np.isnan(metrics["precision_0"])

    def test_load_phase_cache(self, tmp_path):
        """Load fake phase cache."""
        from src.training.phase_evaluate import _load_phase_cache

        cache_dir = _make_fake_cache(tmp_path, split="test", n_windows=50)
        ecg_np, diastole_np, exhalation_np, quality_np = _load_phase_cache(cache_dir, "test")

        assert ecg_np.shape[0] == 50
        assert diastole_np.shape == (50, 10)
        assert exhalation_np.shape == (50, 10)
        assert quality_np.shape == (50, 10)

    def test_load_phase_cache_missing_file(self, tmp_path):
        """Missing cache file → FileNotFoundError."""
        from src.training.phase_evaluate import _load_phase_cache

        cache_dir = tmp_path / "missing"
        cache_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            _load_phase_cache(cache_dir, "test")

    def test_evaluate_phase_detector_basic(self, tmp_path):
        """Run evaluate_phase_detector on fake data (no model required for structure test)."""
        # This is a structural test — full evaluation requires a trained model
        # We verify the function exists and has correct signature
        from src.training.phase_evaluate import evaluate_phase_detector

        # Function should exist and be callable
        assert callable(evaluate_phase_detector)
        # Just verify no import errors

    def test_metrics_precision_recall(self):
        """Per-class precision and recall are correct."""
        from src.training.phase_evaluate import _compute_frame_metrics

        # Create data where we know the metrics
        # True: [1, 1, 0, 0, 1], Pred: [1, 0, 0, 1, 1]
        # TP=2 (indices 0,4), FP=1 (index 3), FN=1 (index 1), TN=1 (index 2)
        labels = np.array([1, 1, 0, 0, 1])
        probs = np.array([0.9, 0.4, 0.3, 0.6, 0.85])
        metrics = _compute_frame_metrics(labels, probs)

        # Class 0: precision=TN/(TN+FP)=1/2=0.5, recall=TN/(TN+FN)=1/2=0.5
        # Class 1: precision=TP/(TP+FP)=2/3≈0.667, recall=TP/(TP+FN)=2/3≈0.667
        assert metrics["n_frames"] == 5
        assert abs(metrics["precision_0"] - 0.5) < 0.01
        assert abs(metrics["recall_0"] - 0.5) < 0.01

    def test_confusion_matrix_shape(self):
        """Confusion matrix is 2x2 for binary classification."""
        from src.training.phase_evaluate import _compute_frame_metrics

        labels = np.array([0, 0, 1, 1])
        probs = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = _compute_frame_metrics(labels, probs)

        cm = metrics["confusion_matrix"]
        assert cm.shape == (2, 2)
        assert cm.sum() == 4
