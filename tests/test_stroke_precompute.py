"""
Tests for stroke_precompute_cache.build_stroke_cache() (Step 9).

All fixtures use in-memory StrokeRecordDicts — no WFDB files written to disk.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.stroke_precompute_cache import build_stroke_cache, N_FEATURES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _stroke_records(
    n: int = 10,
    fs: float = 250.0,
    duration_sec: float = 400.0,
    seed: int = 42,
) -> list:
    """Synthetic StrokeRecordDicts long enough for 300s windows."""
    rng = np.random.default_rng(seed)
    samples = int(fs * duration_sec)
    records = []
    for i in range(n):
        records.append({
            "subject_id": f"stroke_subj_{i:03d}",
            "signal": rng.standard_normal(samples).astype(np.float64) * 0.5,
            "fs": fs,
            "label": i % 2,
            "session_id": None,
            "epoch_type": None,
            "condition": None,
        })
    return records


def _run_cache(records, tmp_dir, extra_cfg=None):
    """Helper: write a minimal config_stroke.yaml and run build_stroke_cache()."""
    import yaml

    cfg = {
        "data": {"waveform_sec": 10, "hrv_window_sec": 300, "stride_sec": 300, "target_fs": 250},
        "training": {"batch_size": 4},
        "hrv": {"subwindow_sec": 60},
        "artifact": {"amplitude_mad_multiple": 30, "rr_deviation_percent": 60,
                     "rr_fraction_threshold": 0.30},
        "wavelet": {"family": "cmor", "scale_range": [1, 64]},
    }
    if extra_cfg:
        cfg.update(extra_cfg)

    cfg_path = str(Path(tmp_dir) / "config_stroke.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    split_path = str(Path(tmp_dir) / "split.json")
    cache_dir = Path(tmp_dir) / "cache"
    scaler_path = str(Path(tmp_dir) / "scaler.pkl")

    meta = build_stroke_cache(
        records=records,
        config_path=cfg_path,
        split_path=split_path,
        cache_dir=cache_dir,
        scaler_path=scaler_path,
        workers=1,
    )
    return meta, cache_dir, split_path, scaler_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_stroke_cache_creates_expected_files():
    """build_stroke_cache writes all expected .npy files + cache_meta.json."""
    records = _stroke_records(n=10, duration_sec=400.0)
    with tempfile.TemporaryDirectory() as tmp:
        meta, cache_dir, _, _ = _run_cache(records, tmp)

        expected = [
            "train_short.npy", "train_hrv.npy", "train_hrv_scaled.npy",
            "train_labels.npy", "train_hrv_lengths.npy",
            "val_short.npy", "val_hrv.npy", "val_hrv_scaled.npy",
            "val_labels.npy", "val_hrv_lengths.npy",
            "test_short.npy", "test_hrv.npy", "test_hrv_scaled.npy",
            "test_labels.npy", "test_hrv_lengths.npy",
            "cache_meta.json",
        ]
        for fname in expected:
            assert (cache_dir / fname).exists(), f"Missing: {fname}"


def test_stroke_cache_array_shapes():
    """short → (n, max_short_len), hrv → (n, 5, 7), labels → (n,)."""
    records = _stroke_records(n=10, fs=250.0, duration_sec=400.0)
    with tempfile.TemporaryDirectory() as tmp:
        meta, cache_dir, _, _ = _run_cache(records, tmp)

        max_short_len = meta["max_short_len"]
        for split in ("train", "val", "test"):
            n = meta[f"n_{split}"]
            if n == 0:
                continue
            short = np.load(cache_dir / f"{split}_short.npy")
            hrv = np.load(cache_dir / f"{split}_hrv.npy")
            labels = np.load(cache_dir / f"{split}_labels.npy")

            assert short.shape == (n, max_short_len), \
                f"{split} short shape {short.shape} != ({n}, {max_short_len})"
            assert hrv.shape == (n, 5, N_FEATURES), \
                f"{split} hrv shape {hrv.shape} != ({n}, 5, {N_FEATURES})"
            assert labels.shape == (n,), \
                f"{split} labels shape {labels.shape} != ({n},)"


def test_stroke_cache_labels_binary():
    """All cached labels are 0.0 or 1.0."""
    records = _stroke_records(n=10, duration_sec=400.0)
    with tempfile.TemporaryDirectory() as tmp:
        _, cache_dir, _, _ = _run_cache(records, tmp)

        for split in ("train", "val", "test"):
            labels = np.load(cache_dir / f"{split}_labels.npy")
            for v in labels:
                assert v in (0.0, 1.0), f"Non-binary label {v} in {split}"


def test_stroke_cache_no_leakage():
    """Subject IDs in train/val/test splits are disjoint."""
    records = _stroke_records(n=10, duration_sec=400.0)
    with tempfile.TemporaryDirectory() as tmp:
        _, _, split_path, _ = _run_cache(records, tmp)

        with open(split_path) as f:
            split = json.load(f)

        train_set = set(split["train"])
        val_set = set(split["val"])
        test_set = set(split["test"])

        assert train_set.isdisjoint(val_set), "Train/val overlap"
        assert train_set.isdisjoint(test_set), "Train/test overlap"
        assert val_set.isdisjoint(test_set), "Val/test overlap"
        assert len(train_set) + len(val_set) + len(test_set) == 10


def test_stroke_cache_scaler_saved():
    """Scaler pkl exists and is loadable after cache build."""
    import joblib

    records = _stroke_records(n=10, duration_sec=400.0)
    with tempfile.TemporaryDirectory() as tmp:
        _, _, _, scaler_path = _run_cache(records, tmp)

        assert Path(scaler_path).exists(), "Scaler file missing"
        scaler = joblib.load(scaler_path)
        assert hasattr(scaler, "transform"), "Loaded object is not a scaler"


def test_stroke_cache_meta_json():
    """cache_meta.json has required keys with sensible values."""
    records = _stroke_records(n=10, duration_sec=400.0)
    with tempfile.TemporaryDirectory() as tmp:
        meta, cache_dir, _, _ = _run_cache(records, tmp)

        with open(cache_dir / "cache_meta.json") as f:
            saved_meta = json.load(f)

        for key in ("max_short_len", "n_train", "n_val", "n_test"):
            assert key in saved_meta, f"Missing key '{key}' in cache_meta.json"

        assert saved_meta["max_short_len"] > 0
        total = saved_meta["n_train"] + saved_meta["n_val"] + saved_meta["n_test"]
        assert total > 0, "Zero total windows in cache"
        assert saved_meta == meta, "Returned meta doesn't match saved cache_meta.json"
