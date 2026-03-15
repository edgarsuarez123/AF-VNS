"""
Tests for streaming precompute pipeline: collect_all_subject_ids, iter_all_records,
create_split_from_ids, window extraction, and end-to-end cache build.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset_parsers import collect_all_subject_ids, iter_all_records
from src.data.splitter import create_split_from_ids, load_split


# ---------------------------------------------------------------------------
# Helpers — create minimal synthetic WFDB databases on disk
# ---------------------------------------------------------------------------

def _write_synthetic_wfdb(record_dir: Path, record_name: str, fs: float,
                          duration_sec: float, label: int = None):
    """Write a minimal WFDB .hea + .dat file, and optional .label file."""
    import wfdb
    n_samples = int(fs * duration_sec)
    signal = np.random.randn(n_samples, 1).astype(np.float64) * 0.3
    wfdb.wrsamp(
        record_name=record_name,
        fs=fs,
        units=["mV"],
        sig_name=["ECG"],
        p_signal=signal,
        write_dir=str(record_dir),
    )
    if label is not None:
        (record_dir / f"{record_name}.label").write_text(str(label))


def _make_synthetic_db(tmp: Path, config_path: Path,
                       n_afdb: int = 2, n_nsrdb: int = 2,
                       n_ltaf: int = 3, n_c17: int = 4,
                       max_ltaf_segments: int = None,
                       target_fs: float = 0,
                       stride_sec: float = 300):
    """Create synthetic databases and config in tmp dir. Returns config path."""
    raw_dir = tmp / "data" / "raw"

    # afdb: 250 Hz, 600s (10 min), AF label inferred from annotations
    afdb_dir = raw_dir / "afdb"
    afdb_dir.mkdir(parents=True)
    for i in range(n_afdb):
        _write_synthetic_wfdb(afdb_dir, f"af{i:03d}", fs=250.0, duration_sec=600.0)

    # nsrdb: 128 Hz, 600s, label=0
    nsrdb_dir = raw_dir / "nsrdb"
    nsrdb_dir.mkdir(parents=True)
    for i in range(n_nsrdb):
        _write_synthetic_wfdb(nsrdb_dir, f"nsr{i:03d}", fs=128.0, duration_sec=600.0)

    # ltafdb: 128 Hz, varying durations, label=1
    ltafdb_dir = raw_dir / "ltafdb"
    ltafdb_dir.mkdir(parents=True)
    for i in range(n_ltaf):
        _write_synthetic_wfdb(ltafdb_dir, f"seg{i:03d}", fs=128.0,
                              duration_sec=400.0 + i * 100, label=1)

    # challenge2017: 300 Hz, 45s, labels 0 or 1
    c17_dir = raw_dir / "challenge2017"
    c17_dir.mkdir(parents=True)
    for i in range(n_c17):
        _write_synthetic_wfdb(c17_dir, f"A{i:05d}", fs=300.0, duration_sec=45.0,
                              label=i % 2)

    cfg = {
        "data": {
            "raw_dir": str(raw_dir),
            "mimic3_subdir": str(raw_dir / "mimic3"),
            "split_path": str(tmp / "split.json"),
            "waveform_sec": 10,
            "hrv_window_sec": 300,
            "stride_sec": stride_sec,
            "target_fs": target_fs,
        },
        "paths": {
            "cache_dir": str(tmp / "cache"),
            "scaler": str(tmp / "scaler.pkl"),
            "split": str(tmp / "split.json"),
        },
        "hrv": {"subwindow_sec": 60},
        "wavelet": {"family": "cmor", "scale_range": [1, 64]},
        "artifact": {
            "amplitude_mad_multiple": 30,
            "rr_deviation_percent": 60,
            "rr_fraction_threshold": 0.30,
        },
        "training": {"batch_size": 32, "learning_rate": 0.001, "max_epochs": 1},
        "model": {
            "in_channels": 1, "cnn_embed_dim": 128, "rnn_hidden_size": 64,
            "rnn_num_layers": 1, "transformer_d_model": 64, "transformer_nhead": 4,
            "transformer_num_encoder_layers": 1, "transformer_dim_feedforward": 128,
            "hrv_seq_len": 5, "hrv_n_features": 7,
        },
    }
    if max_ltaf_segments is not None:
        cfg["data"]["max_ltaf_segments"] = max_ltaf_segments

    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f)

    return config_path


# ---------------------------------------------------------------------------
# Tests: collect_all_subject_ids
# ---------------------------------------------------------------------------

class TestCollectSubjectIds:

    def test_counts_match_databases(self):
        """Total IDs = n_afdb + n_nsrdb + n_ltaf + n_c17."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg_path = tmp / "config.yaml"
            _make_synthetic_db(tmp, cfg_path, n_afdb=2, n_nsrdb=2, n_ltaf=3, n_c17=4)
            ids = collect_all_subject_ids(str(cfg_path))
            assert len(ids) == 2 + 2 + 3 + 4

    def test_id_prefixes_correct(self):
        """ltafdb IDs prefixed with 'ltaf_', c17 with 'c17_'."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg_path = tmp / "config.yaml"
            _make_synthetic_db(tmp, cfg_path, n_afdb=1, n_nsrdb=1, n_ltaf=2, n_c17=2)
            ids = collect_all_subject_ids(str(cfg_path))
            ltaf_ids = [x for x in ids if x.startswith("ltaf_")]
            c17_ids = [x for x in ids if x.startswith("c17_")]
            assert len(ltaf_ids) == 2
            assert len(c17_ids) == 2

    def test_respects_max_ltaf_cap(self):
        """With max_ltaf_segments=2, only 2 ltaf IDs returned despite 5 on disk."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg_path = tmp / "config.yaml"
            _make_synthetic_db(tmp, cfg_path, n_afdb=1, n_nsrdb=1, n_ltaf=5, n_c17=1,
                               max_ltaf_segments=2)
            ids = collect_all_subject_ids(str(cfg_path))
            ltaf_ids = [x for x in ids if x.startswith("ltaf_")]
            assert len(ltaf_ids) == 2

    def test_no_signal_data_loaded(self):
        """collect_all_subject_ids should not load any wfdb records (fast)."""
        # This is a smoke test — if it runs in < 1s for a small DB, no signals loaded
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg_path = tmp / "config.yaml"
            _make_synthetic_db(tmp, cfg_path, n_afdb=2, n_nsrdb=2, n_ltaf=3, n_c17=4)
            import time
            t0 = time.time()
            ids = collect_all_subject_ids(str(cfg_path))
            elapsed = time.time() - t0
            assert len(ids) > 0
            assert elapsed < 5.0, f"Took {elapsed:.1f}s — likely loading signals"


# ---------------------------------------------------------------------------
# Tests: create_split_from_ids
# ---------------------------------------------------------------------------

class TestCreateSplitFromIds:

    def test_proportions_and_disjoint(self):
        """70/15/15 split with no overlap between sets."""
        with tempfile.TemporaryDirectory() as tmp:
            split_path = str(Path(tmp) / "split.json")
            ids = [f"subj_{i:03d}" for i in range(100)]
            train, val, test = create_split_from_ids(ids, split_path)

            assert len(train) == 70
            assert len(val) == 15
            assert len(test) == 15
            assert not set(train) & set(val)
            assert not set(train) & set(test)
            assert not set(val) & set(test)

    def test_roundtrip_load(self):
        """Split saved to JSON can be loaded back identically."""
        with tempfile.TemporaryDirectory() as tmp:
            split_path = str(Path(tmp) / "split.json")
            ids = [f"subj_{i:03d}" for i in range(20)]
            train, val, test = create_split_from_ids(ids, split_path)
            t2, v2, te2 = load_split(split_path)
            assert set(train) == set(t2)
            assert set(val) == set(v2)
            assert set(test) == set(te2)

    def test_deduplicates_ids(self):
        """Duplicate IDs in input produce correct unique split."""
        with tempfile.TemporaryDirectory() as tmp:
            split_path = str(Path(tmp) / "split.json")
            ids = [f"subj_{i:03d}" for i in range(10)] * 3  # 30 entries, 10 unique
            train, val, test = create_split_from_ids(ids, split_path)
            assert len(train) + len(val) + len(test) == 10


# ---------------------------------------------------------------------------
# Tests: iter_all_records
# ---------------------------------------------------------------------------

class TestIterAllRecords:

    def test_yields_valid_schema(self):
        """Each yielded record has subject_id (str), signal (1D ndarray), fs (float), label."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg_path = tmp / "config.yaml"
            _make_synthetic_db(tmp, cfg_path, n_afdb=1, n_nsrdb=1, n_ltaf=1, n_c17=1)
            for rec in iter_all_records(str(cfg_path)):
                assert isinstance(rec["subject_id"], str)
                assert isinstance(rec["signal"], np.ndarray)
                assert rec["signal"].ndim == 1
                assert isinstance(rec["fs"], float)
                assert rec.get("label") is not None

    def test_count_matches_ids(self):
        """Total yielded records == collect_all_subject_ids count."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg_path = tmp / "config.yaml"
            _make_synthetic_db(tmp, cfg_path, n_afdb=2, n_nsrdb=1, n_ltaf=2, n_c17=3)
            ids = collect_all_subject_ids(str(cfg_path))
            records = list(iter_all_records(str(cfg_path)))
            assert len(records) == len(ids)

    def test_resamples_to_target_fs(self):
        """With target_fs=250, all records (128 Hz nsrdb, 300 Hz c17) are resampled."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg_path = tmp / "config.yaml"
            _make_synthetic_db(tmp, cfg_path, n_afdb=0, n_nsrdb=1, n_ltaf=0, n_c17=1,
                               target_fs=250)
            for rec in iter_all_records(str(cfg_path)):
                assert abs(rec["fs"] - 250.0) < 0.1, f"{rec['subject_id']} fs={rec['fs']}"

    def test_respects_ltaf_cap(self):
        """With max_ltaf_segments=2, only 2 ltaf records yielded."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg_path = tmp / "config.yaml"
            _make_synthetic_db(tmp, cfg_path, n_afdb=0, n_nsrdb=0, n_ltaf=5, n_c17=0,
                               max_ltaf_segments=2)
            records = list(iter_all_records(str(cfg_path)))
            assert len(records) == 2


# ---------------------------------------------------------------------------
# Tests: window extraction logic
# ---------------------------------------------------------------------------

class TestWindowExtraction:

    def test_non_overlapping_stride_300(self):
        """With stride=300s and a 600s signal, expect exactly 2 windows 300s apart."""
        fs = 250.0
        sig_len = int(fs * 620)  # 620s -> 2 full 300s windows
        n_long = int(fs * 300)
        stride = int(fs * 300)
        starts = list(range(0, sig_len - n_long + 1, stride))
        assert len(starts) == 2
        assert starts[0] == 0
        assert starts[1] == int(fs * 300)

    def test_short_record_one_window(self):
        """Record < 300s but >= 10s should produce 1 window (short-record path)."""
        fs = 250.0
        sig_len = int(fs * 45)  # 45s — typical challenge2017 record
        n_short = int(fs * 10)
        n_long = int(fs * 300)

        if sig_len >= n_long:
            starts = list(range(0, sig_len - n_long + 1, int(fs * 300)))
        elif sig_len >= n_short:
            starts = [0]  # short record path
        else:
            starts = []

        assert len(starts) == 1

    def test_too_short_record_zero_windows(self):
        """Record < 10s should produce 0 windows."""
        fs = 250.0
        sig_len = int(fs * 5)  # 5s — too short for CNN
        n_short = int(fs * 10)

        if sig_len >= n_short:
            starts = [0]
        else:
            starts = []

        assert len(starts) == 0

    def test_long_record_window_count(self):
        """10-hour record at 250 Hz with stride=300s => 120 windows."""
        fs = 250.0
        sig_len = int(fs * 36000)  # 10 hours
        n_long = int(fs * 300)
        stride = int(fs * 300)
        starts = list(range(0, sig_len - n_long + 1, stride))
        assert len(starts) == 120  # (36000-300)/300 + 1 = 120


# ---------------------------------------------------------------------------
# Tests: end-to-end streaming precompute (small synthetic data)
# ---------------------------------------------------------------------------

class TestStreamingPrecompute:

    def test_produces_valid_cache_files(self):
        """Full precompute on tiny synthetic DB produces expected .npy files."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg_path = tmp / "config.yaml"
            # Need records >= 300s for valid HRV (non-NaN scaler input)
            _make_synthetic_db(tmp, cfg_path, n_afdb=3, n_nsrdb=3,
                               n_ltaf=0, n_c17=0, stride_sec=300)

            from src.training.precompute_cache import main as precompute_main
            precompute_main(config_path=str(cfg_path), workers=1)

            cache_dir = tmp / "cache"
            assert (cache_dir / "cache_meta.json").exists()

            with open(cache_dir / "cache_meta.json") as f:
                meta = json.load(f)

            total = meta["n_train"] + meta["n_val"] + meta["n_test"]
            assert total > 0, "Expected at least some cached windows"

            # Verify file shapes
            for split in ("train", "val", "test"):
                n = meta[f"n_{split}"]
                if n == 0:
                    continue
                short = np.load(cache_dir / f"{split}_short.npy")
                hrv = np.load(cache_dir / f"{split}_hrv.npy")
                hrv_s = np.load(cache_dir / f"{split}_hrv_scaled.npy")
                labels = np.load(cache_dir / f"{split}_labels.npy")

                assert short.shape[0] == n
                assert short.dtype == np.float32
                assert hrv.shape == (n, 5, 7)
                assert hrv.dtype == np.float32
                assert hrv_s.shape == hrv.shape
                assert labels.shape == (n,)
                assert labels.dtype == np.float32

    def test_split_regenerated_when_missing(self):
        """Precompute creates split.json if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg_path = tmp / "config.yaml"
            _make_synthetic_db(tmp, cfg_path, n_afdb=3, n_nsrdb=3,
                               n_ltaf=0, n_c17=0)

            split_path = tmp / "split.json"
            assert not split_path.exists()

            from src.training.precompute_cache import main as precompute_main
            precompute_main(config_path=str(cfg_path), workers=1)

            assert split_path.exists()
            train, val, test = load_split(str(split_path))
            assert len(train) + len(val) + len(test) == 6  # 3 afdb + 3 nsrdb
