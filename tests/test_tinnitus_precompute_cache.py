"""Tests for tinnitus_precompute_cache.py (F9)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.tinnitus_precompute_cache import process_record, build_tinnitus_cache

ROOT = Path(__file__).resolve().parents[1]
BIDMC_DIR = ROOT / "data/raw/stroke avns/bidmc"
WESAD_DIR = ROOT / "data/raw/tinnitus avns/wesad/WESAD"
CONFIG_PATH = str(ROOT / "config_tinnitus.yaml")

BIDMC_AVAILABLE = BIDMC_DIR.is_dir() and any(BIDMC_DIR.glob("*.hea"))
WESAD_AVAILABLE = WESAD_DIR.is_dir() and any(WESAD_DIR.iterdir())

WINDOW_SEC = 2.0
FRAME_RATE_HZ = 5.0
TARGET_FS = 125.0
WINDOW_SAMPLES = int(TARGET_FS * WINDOW_SEC)   # 250
FRAMES_PER_WIN = int(WINDOW_SEC * FRAME_RATE_HZ)  # 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(fs=125.0, duration_sec=30.0, with_resp=True):
    """Synthetic TinnitusRecordDict for unit tests."""
    rng = np.random.default_rng(42)
    n = int(fs * duration_sec)

    # Simple synthetic PPG: sine at 1 Hz + noise (enough for peak detection)
    t = np.arange(n) / fs
    ppg = np.sin(2 * np.pi * 1.1 * t) + 0.1 * rng.standard_normal(n)

    resp = None
    resp_fs = None
    resp_channel = None
    if with_resp:
        resp = np.sin(2 * np.pi * 0.25 * t[:n]) + 0.05 * rng.standard_normal(n)
        resp_fs = fs
        resp_channel = "impedance"

    return {
        "subject_id": "test_subj",
        "ppg_signal": ppg.astype(np.float64),
        "ppg_fs": fs,
        "resp_signal": resp,
        "resp_fs": resp_fs,
        "resp_channel": resp_channel,
        "eda_signal": None,
        "eda_fs": None,
        "label": 0,
        "session_id": None,
        "condition": None,
    }


# ---------------------------------------------------------------------------
# TestProcessRecord
# ---------------------------------------------------------------------------

class TestProcessRecord:
    """Unit tests for process_record()."""

    def test_returns_none_for_short_signal(self):
        rec = _make_record(fs=125.0, duration_sec=1.0)  # < window_samples
        result = process_record(rec, CONFIG_PATH)
        assert result is None

    def test_output_keys(self):
        rec = _make_record(fs=125.0, duration_sec=60.0)
        result = process_record(rec, CONFIG_PATH)
        assert result is not None
        for key in ("ppg", "diastole", "exhalation", "quality", "n_windows", "n_skipped"):
            assert key in result, f"Missing key: {key}"

    def test_window_shape_125hz(self):
        """125 Hz records produce 250-sample PPG windows."""
        rec = _make_record(fs=125.0, duration_sec=60.0)
        result = process_record(rec, CONFIG_PATH)
        assert result is not None and result["n_windows"] > 0
        ppg_arr = np.array(result["ppg"])
        assert ppg_arr.shape[1] == WINDOW_SAMPLES, (
            f"Expected {WINDOW_SAMPLES} samples, got {ppg_arr.shape[1]}"
        )

    def test_window_shape_64hz_resampled(self):
        """64 Hz records (WESAD BVP) are resampled to 125 Hz → 250 samples."""
        rec = _make_record(fs=64.0, duration_sec=60.0)
        result = process_record(rec, CONFIG_PATH)
        assert result is not None and result["n_windows"] > 0
        ppg_arr = np.array(result["ppg"])
        assert ppg_arr.shape[1] == WINDOW_SAMPLES, (
            f"Expected {WINDOW_SAMPLES} samples after resample, got {ppg_arr.shape[1]}"
        )

    def test_label_frames_shape(self):
        """Diastole and exhalation label arrays have frames_per_window columns."""
        rec = _make_record(fs=125.0, duration_sec=60.0)
        result = process_record(rec, CONFIG_PATH)
        assert result is not None and result["n_windows"] > 0
        dia = np.array(result["diastole"])
        exh = np.array(result["exhalation"])
        assert dia.shape[1] == FRAMES_PER_WIN
        assert exh.shape[1] == FRAMES_PER_WIN

    def test_no_resp_falls_back_to_ppg_derived(self):
        """Records without resp_signal fall back to PPG-derived exhalation."""
        rec = _make_record(fs=125.0, duration_sec=60.0, with_resp=False)
        result = process_record(rec, CONFIG_PATH)
        # Should not be None — PPG-derived exhalation should succeed or gracefully fail
        # (may return windows with NaN exhalation if PPG resp fails, but diastole should work)
        if result is not None:
            assert result["n_windows"] > 0
            assert result["exh_method"] in (
                "ppg_derived", "ppg_derived_failed", "none",
            )

    def test_n_windows_positive(self):
        rec = _make_record(fs=125.0, duration_sec=120.0)
        result = process_record(rec, CONFIG_PATH)
        assert result is not None
        assert result["n_windows"] > 0

    def test_ppg_dtype_float32(self):
        rec = _make_record(fs=125.0, duration_sec=60.0)
        result = process_record(rec, CONFIG_PATH)
        assert result is not None
        assert np.array(result["ppg"]).dtype == np.float32


# ---------------------------------------------------------------------------
# TestBuildTinnitusCache
# ---------------------------------------------------------------------------

class TestBuildTinnitusCache:
    """Unit tests for build_tinnitus_cache() using synthetic records."""

    def _make_records(self, n=6, fs=125.0, duration_sec=60.0):
        """n synthetic records with unique subject IDs."""
        records = []
        for i in range(n):
            rec = _make_record(fs=fs, duration_sec=duration_sec)
            rec["subject_id"] = f"synth_{i:02d}"
            records.append(rec)
        return records

    def test_cache_files_created(self, tmp_path):
        records = self._make_records(n=6)
        split_path = str(tmp_path / "split.json")
        meta = build_tinnitus_cache(
            records=records,
            config_path=CONFIG_PATH,
            split_path=split_path,
            cache_dir=tmp_path / "cache",
            workers=1,
        )
        cache_dir = tmp_path / "cache"
        for split in ("train", "val", "test"):
            for arr in ("ppg", "diastole", "exhalation", "quality"):
                f = cache_dir / f"{split}_{arr}.npy"
                assert f.exists(), f"Missing cache file: {f.name}"

    def test_meta_json_written(self, tmp_path):
        records = self._make_records(n=6)
        build_tinnitus_cache(
            records=records,
            config_path=CONFIG_PATH,
            split_path=str(tmp_path / "split.json"),
            cache_dir=tmp_path / "cache",
            workers=1,
        )
        meta_path = tmp_path / "cache" / "tinnitus_cache_meta.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        for key in ("window_samples", "frames_per_window", "n_train", "n_val", "n_test"):
            assert key in meta

    def test_npy_shapes(self, tmp_path):
        """Saved .npy files have correct shapes (N, 250) and (N, 10)."""
        records = self._make_records(n=6)
        build_tinnitus_cache(
            records=records,
            config_path=CONFIG_PATH,
            split_path=str(tmp_path / "split.json"),
            cache_dir=tmp_path / "cache",
            workers=1,
        )
        cache_dir = tmp_path / "cache"
        for split in ("train", "val", "test"):
            ppg = np.load(cache_dir / f"{split}_ppg.npy")
            dia = np.load(cache_dir / f"{split}_diastole.npy")
            exh = np.load(cache_dir / f"{split}_exhalation.npy")
            if ppg.shape[0] > 0:
                assert ppg.shape[1] == WINDOW_SAMPLES, (
                    f"{split}_ppg: expected shape[1]={WINDOW_SAMPLES}, got {ppg.shape[1]}"
                )
                assert dia.shape[1] == FRAMES_PER_WIN
                assert exh.shape[1] == FRAMES_PER_WIN
                assert dia.shape[0] == ppg.shape[0]

    def test_split_json_created(self, tmp_path):
        records = self._make_records(n=6)
        split_path = str(tmp_path / "split.json")
        build_tinnitus_cache(
            records=records,
            config_path=CONFIG_PATH,
            split_path=split_path,
            cache_dir=tmp_path / "cache",
            workers=1,
        )
        assert Path(split_path).exists()
        with open(split_path) as f:
            split = json.load(f)
        assert "train" in split
        assert "val" in split
        assert "test" in split

    def test_wesad_64hz_records_in_cache(self, tmp_path):
        """64 Hz WESAD records are correctly resampled and cached as 250 samples."""
        records = self._make_records(n=6, fs=64.0)
        build_tinnitus_cache(
            records=records,
            config_path=CONFIG_PATH,
            split_path=str(tmp_path / "split.json"),
            cache_dir=tmp_path / "cache",
            workers=1,
        )
        cache_dir = tmp_path / "cache"
        for split in ("train", "val", "test"):
            ppg = np.load(cache_dir / f"{split}_ppg.npy")
            if ppg.shape[0] > 0:
                assert ppg.shape[1] == WINDOW_SAMPLES


# ---------------------------------------------------------------------------
# Real data integration tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not BIDMC_AVAILABLE, reason="BIDMC dataset not available")
class TestBidmcIntegration:
    """Smoke test: process first 3 BIDMC records end-to-end."""

    def test_bidmc_records_produce_windows(self, tmp_path):
        from src.data.tinnitus_parsers import parse_bidmc_ppg_dir

        records = parse_bidmc_ppg_dir(str(BIDMC_DIR), CONFIG_PATH)[:3]
        assert len(records) > 0

        results = [process_record(r, CONFIG_PATH) for r in records]
        good = [r for r in results if r is not None]
        assert len(good) > 0, "All BIDMC records failed processing"

        for r in good:
            ppg_arr = np.array(r["ppg"])
            assert ppg_arr.shape[1] == WINDOW_SAMPLES
            assert r["n_windows"] > 0


@pytest.mark.skipif(not WESAD_AVAILABLE, reason="WESAD dataset not available")
class TestWesadIntegration:
    """Smoke test: process first 2 WESAD subjects end-to-end."""

    def test_wesad_records_produce_windows(self, tmp_path):
        from src.data.tinnitus_parsers import parse_wesad_dir

        records = parse_wesad_dir(str(WESAD_DIR), CONFIG_PATH)[:4]
        assert len(records) > 0

        results = [process_record(r, CONFIG_PATH) for r in records]
        good = [r for r in results if r is not None]
        assert len(good) > 0, "All WESAD records failed processing"

        for r in good:
            ppg_arr = np.array(r["ppg"])
            # WESAD at 64 Hz is resampled to 125 Hz → 250 samples
            assert ppg_arr.shape[1] == WINDOW_SAMPLES, (
                f"Expected 250 samples (64→125 Hz resample), got {ppg_arr.shape[1]}"
            )
