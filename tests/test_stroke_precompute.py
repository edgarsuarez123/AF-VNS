"""
Tests for stroke_precompute_cache.build_stroke_cache() — phase detection version (S-16).

All fixtures use in-memory StrokeRecordDicts with synthetic ECG via neurokit2.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.stroke_precompute_cache import (
    build_stroke_cache,
    process_record,
    SHAREE_EVENT_PATIENTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _synth_ecg(duration: int = 30, fs: int = 250, heart_rate: int = 72) -> np.ndarray:
    """Generate synthetic ECG via neurokit2."""
    import neurokit2 as nk
    ecg = nk.ecg_simulate(duration=duration, sampling_rate=fs, heart_rate=heart_rate)
    return np.asarray(ecg, dtype=np.float64)


def _phase_records(
    n: int = 10,
    fs: float = 250.0,
    duration_sec: int = 30,
    heart_rate: int = 72,
) -> list:
    """Synthetic StrokeRecordDicts with realistic ECG for phase detection."""
    records = []
    for i in range(n):
        records.append({
            "subject_id": f"phase_subj_{i:03d}",
            "signal": _synth_ecg(duration=duration_sec, fs=int(fs), heart_rate=heart_rate),
            "fs": fs,
            "label": i % 2,
            "session_id": None,
            "epoch_type": None,
            "condition": None,
        })
    return records


def _make_config(tmp_dir: str) -> str:
    """Write minimal config YAML for phase precompute tests."""
    cfg = {
        "data": {"target_fs": 250, "waveform_sec": 10},
        "phase_detection": {
            "frame_rate_hz": 5.0,
            "fallback_fraction": 0.40,
            "min_beats": 2,
            "min_hr_bpm": 40.0,
            "max_hr_bpm": 200.0,
            "delineate_method": "dwt",
            "denoise_before_delineate": False,
        },
        "edr": {
            "method": "vangent2019",
            "min_resp_rate_bpm": 6.0,
            "max_resp_rate_bpm": 30.0,
            "min_resp_cycles": 2,
            "denoise_before_edr": False,
        },
        "wavelet": {"family": "cmor", "scale_range": [1, 64]},
    }
    path = str(Path(tmp_dir) / "config_stroke.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def _run_cache(records, tmp_dir, window_sec=2.0, stride_sec=0.2, min_valid=8):
    """Helper: write config and run build_stroke_cache()."""
    cfg_path = _make_config(tmp_dir)
    split_path = str(Path(tmp_dir) / "split.json")
    cache_dir = Path(tmp_dir) / "cache"

    meta = build_stroke_cache(
        records=records,
        config_path=cfg_path,
        split_path=split_path,
        cache_dir=cache_dir,
        workers=1,
        window_sec=window_sec,
        stride_sec=stride_sec,
        min_valid_frames=min_valid,
        frame_rate_hz=5.0,
    )
    return meta, cache_dir, split_path


# ---------------------------------------------------------------------------
# Tests: build_stroke_cache
# ---------------------------------------------------------------------------

class TestBuildStrokeCache:

    def test_creates_expected_files(self):
        records = _phase_records(n=6, duration_sec=30)
        with tempfile.TemporaryDirectory() as tmp:
            meta, cache_dir, _ = _run_cache(records, tmp)

            expected = [
                "train_ecg.npy", "train_diastole.npy",
                "train_exhalation.npy", "train_quality.npy",
                "val_ecg.npy", "val_diastole.npy",
                "val_exhalation.npy", "val_quality.npy",
                "test_ecg.npy", "test_diastole.npy",
                "test_exhalation.npy", "test_quality.npy",
                "phase_cache_meta.json",
            ]
            for fname in expected:
                assert (cache_dir / fname).exists(), f"Missing: {fname}"

    def test_array_shapes(self):
        records = _phase_records(n=6, duration_sec=30)
        with tempfile.TemporaryDirectory() as tmp:
            meta, cache_dir, _ = _run_cache(records, tmp)

            for split in ("train", "val", "test"):
                n = meta[f"n_{split}"]
                if n == 0:
                    continue
                ecg = np.load(cache_dir / f"{split}_ecg.npy")
                dia = np.load(cache_dir / f"{split}_diastole.npy")
                exh = np.load(cache_dir / f"{split}_exhalation.npy")
                qual = np.load(cache_dir / f"{split}_quality.npy")

                assert ecg.shape == (n, 500), f"{split} ecg {ecg.shape} != ({n}, 500)"
                assert dia.shape == (n, 10), f"{split} diastole {dia.shape} != ({n}, 10)"
                assert exh.shape == (n, 10), f"{split} exhalation {exh.shape} != ({n}, 10)"
                assert qual.shape == (n, 10), f"{split} quality {qual.shape} != ({n}, 10)"

    def test_meta_json(self):
        records = _phase_records(n=6, duration_sec=30)
        with tempfile.TemporaryDirectory() as tmp:
            meta, cache_dir, _ = _run_cache(records, tmp)

            with open(cache_dir / "phase_cache_meta.json") as f:
                saved = json.load(f)

            for key in ("window_samples", "frames_per_window", "n_train", "n_val",
                        "n_test", "total_windows"):
                assert key in saved, f"Missing key '{key}' in phase_cache_meta.json"
            assert saved["window_samples"] == 500
            assert saved["frames_per_window"] == 10
            total = saved["n_train"] + saved["n_val"] + saved["n_test"]
            assert total > 0, "Zero total windows"

    def test_no_leakage(self):
        records = _phase_records(n=10, duration_sec=30)
        with tempfile.TemporaryDirectory() as tmp:
            _, _, split_path = _run_cache(records, tmp)

            with open(split_path) as f:
                split = json.load(f)

            train_set = set(split["train"])
            val_set = set(split["val"])
            test_set = set(split["test"])

            assert train_set.isdisjoint(val_set), "Train/val overlap"
            assert train_set.isdisjoint(test_set), "Train/test overlap"
            assert val_set.isdisjoint(test_set), "Val/test overlap"

    def test_labels_binary_or_nan(self):
        records = _phase_records(n=6, duration_sec=30)
        with tempfile.TemporaryDirectory() as tmp:
            meta, cache_dir, _ = _run_cache(records, tmp)

            for split in ("train", "val", "test"):
                n = meta[f"n_{split}"]
                if n == 0:
                    continue
                for name in ("diastole", "exhalation"):
                    arr = np.load(cache_dir / f"{split}_{name}.npy")
                    valid = arr[~np.isnan(arr)]
                    assert set(valid.tolist()).issubset({0.0, 1.0}), \
                        f"{split} {name} has non-binary values"

    def test_quality_in_range(self):
        records = _phase_records(n=6, duration_sec=30)
        with tempfile.TemporaryDirectory() as tmp:
            meta, cache_dir, _ = _run_cache(records, tmp)

            for split in ("train", "val", "test"):
                n = meta[f"n_{split}"]
                if n == 0:
                    continue
                qual = np.load(cache_dir / f"{split}_quality.npy")
                assert np.all(qual >= 0.0) and np.all(qual <= 1.0), \
                    f"{split} quality out of [0, 1]"


# ---------------------------------------------------------------------------
# Tests: process_record
# ---------------------------------------------------------------------------

class TestProcessRecord:

    def test_valid_record(self):
        """30s synthetic ECG → returns windows."""
        rec = _phase_records(n=1, duration_sec=30)[0]
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_config(tmp)
            result = process_record(rec, cfg, window_sec=2.0, stride_sec=0.2)

        assert result is not None
        assert result["n_windows"] > 0
        assert len(result["ecg"]) == result["n_windows"]
        assert result["ecg"][0].shape == (500,)
        assert result["diastole"][0].shape == (10,)

    def test_too_short_signal(self):
        """1s signal → returns None (shorter than 2s window)."""
        rec = {
            "subject_id": "short_subj",
            "signal": np.zeros(250, dtype=np.float64),
            "fs": 250.0,
            "label": 0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_config(tmp)
            result = process_record(rec, cfg)
        assert result is None

    def test_dc_signal(self):
        """Flat DC signal → returns None (no R-peaks, no resp cycles)."""
        rec = {
            "subject_id": "dc_subj",
            "signal": np.zeros(250 * 30, dtype=np.float64),
            "fs": 250.0,
            "label": 0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_config(tmp)
            result = process_record(rec, cfg)
        assert result is None

    def test_window_count(self):
        """Verify expected window count for known duration."""
        duration = 30
        fs = 250
        window_sec = 2.0
        stride_sec = 0.2
        rec = _phase_records(n=1, duration_sec=duration, fs=fs)[0]

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_config(tmp)
            result = process_record(
                rec, cfg, window_sec=window_sec, stride_sec=stride_sec,
                min_valid_frames=0,  # don't filter
            )

        assert result is not None
        # max_start = 7500 - 500 + 1 = 7001; steps = range(0, 7001, 50)
        # expected = len(range(0, 7001, 50)) = 141
        # But frame bounds may reduce this — just check approximate
        expected_approx = int((duration - window_sec) / stride_sec) + 1  # 141
        assert abs(result["n_windows"] - expected_approx) <= 2

    def test_nan_preservation(self):
        """NaN labels from failed exhalation detection should be preserved."""
        # Use a short signal where EDR fails (< 10s) but diastole succeeds
        duration = 8  # < 10s for EDR, > 2s for diastole
        rec = _phase_records(n=1, duration_sec=duration)[0]
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_config(tmp)
            result = process_record(
                rec, cfg, window_sec=2.0, stride_sec=0.2, min_valid_frames=0,
            )

        if result is not None:
            # Exhalation should be all NaN (signal too short for EDR)
            exh_all = np.concatenate(result["exhalation"])
            assert np.all(np.isnan(exh_all)), \
                "Expected all-NaN exhalation labels for <10s signal"


# ---------------------------------------------------------------------------
# Backward compat
# ---------------------------------------------------------------------------

def test_sharee_event_patients_constant():
    """SHAREE_EVENT_PATIENTS constant still exists for backward compat."""
    assert len(SHAREE_EVENT_PATIENTS) == 17
    assert "02119" in SHAREE_EVENT_PATIENTS
