"""
Tests for stroke_parsers.py — schema validation, label correctness, signal shape,
resampling, short-record skipping, and stub behaviour.

Unit tests use synthetic fixtures (random numpy arrays).
Integration tests (marked with @pytest.mark.integration) hit real data and are
excluded from CI by default.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import wfdb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.stroke_parsers import (
    parse_cerevasc_dir,
    parse_mimic3_stroke_dir,
    parse_sharee_dir,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_wfdb_record(
    tmp_dir: Path,
    name: str,
    n_samples: int = 3000,
    fs: int = 125,
    n_channels: int = 2,
    label: int = 1,
) -> Path:
    """Write a minimal WFDB record (.hea + .dat) and a .label file."""
    # Keep amplitude in ±1.0 mV so wfdb int32 digital conversion doesn't overflow
    rng = np.random.default_rng(42)
    sig = (rng.standard_normal((n_samples, n_channels)) * 0.5).astype(np.float64)
    wfdb.wrsamp(
        name,
        fs=fs,
        units=["mV"] * n_channels,
        sig_name=["II", "RESP"][:n_channels],
        p_signal=sig,
        write_dir=str(tmp_dir),
    )
    (tmp_dir / f"{name}.label").write_text(str(label))
    return tmp_dir


def _make_stroke_config(tmp_dir: Path, target_fs: float = 250.0, waveform_sec: float = 10.0) -> str:
    """Write a minimal config YAML and return its path."""
    import yaml
    cfg = {
        "data": {
            "target_fs": target_fs,
            "waveform_sec": waveform_sec,
        }
    }
    cfg_path = tmp_dir / "config_test.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return str(cfg_path)


# ---------------------------------------------------------------------------
# Unit tests — parse_mimic3_stroke_dir
# ---------------------------------------------------------------------------

def test_parse_mimic3_stroke_returns_correct_schema():
    """All required StrokeRecordDict keys are present in every returned record."""
    required_keys = {"subject_id", "signal", "fs", "label", "session_id", "epoch_type", "condition"}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _write_wfdb_record(tmp_dir, "rec001", n_samples=5000, fs=125, label=1)
        cfg = _make_stroke_config(tmp_dir)
        records = parse_mimic3_stroke_dir(str(tmp_dir), cfg)
        assert len(records) == 1
        assert required_keys.issubset(records[0].keys()), (
            f"Missing keys: {required_keys - set(records[0].keys())}"
        )


def test_parse_mimic3_stroke_labels_binary():
    """All returned labels are exactly 0 or 1."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _write_wfdb_record(tmp_dir, "stroke01", n_samples=5000, fs=125, label=1)
        _write_wfdb_record(tmp_dir, "ctrl01", n_samples=5000, fs=125, label=0)
        cfg = _make_stroke_config(tmp_dir)
        records = parse_mimic3_stroke_dir(str(tmp_dir), cfg)
        assert len(records) == 2
        for r in records:
            assert r["label"] in (0, 1), f"Label {r['label']} is not binary"


def test_parse_mimic3_stroke_signal_shape():
    """Returned signal is a 1D float64 array."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _write_wfdb_record(tmp_dir, "rec001", n_samples=5000, fs=125, n_channels=2, label=0)
        cfg = _make_stroke_config(tmp_dir)
        records = parse_mimic3_stroke_dir(str(tmp_dir), cfg)
        assert len(records) == 1
        sig = records[0]["signal"]
        assert sig.ndim == 1, f"Expected 1D signal, got shape {sig.shape}"
        assert sig.dtype == np.float64, f"Expected float64, got {sig.dtype}"


def test_parse_mimic3_stroke_resampled_to_target_fs():
    """Signal fs matches config target_fs after parsing (resampling applied)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        # Record at 125 Hz; config target is 250 Hz
        _write_wfdb_record(tmp_dir, "rec001", n_samples=5000, fs=125, label=1)
        cfg = _make_stroke_config(tmp_dir, target_fs=250.0)
        records = parse_mimic3_stroke_dir(str(tmp_dir), cfg)
        assert len(records) == 1
        assert abs(records[0]["fs"] - 250.0) < 0.1, (
            f"Expected fs=250.0 after resampling, got {records[0]['fs']}"
        )


def test_parse_mimic3_stroke_skips_short_records():
    """Records shorter than waveform_sec are excluded from results."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        # 50 samples at 125 Hz = 0.4 s < 10 s waveform_sec
        _write_wfdb_record(tmp_dir, "short_rec", n_samples=50, fs=125, label=1)
        # 5000 samples at 125 Hz = 40 s — should pass
        _write_wfdb_record(tmp_dir, "long_rec", n_samples=5000, fs=125, label=0)
        cfg = _make_stroke_config(tmp_dir, waveform_sec=10.0)
        records = parse_mimic3_stroke_dir(str(tmp_dir), cfg)
        ids = [r["subject_id"] for r in records]
        assert "short_rec" not in ids, "Short record should have been skipped"
        assert "long_rec" in ids, "Long record should be included"


# ---------------------------------------------------------------------------
# Unit tests — parse_sharee_dir
# ---------------------------------------------------------------------------

def test_parse_sharee_dir_default_label_zero():
    """Records not present in label_map receive label=0."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        # Write a record but don't put it in label_map
        rng = np.random.default_rng(42)
        sig = (rng.standard_normal((5000, 1)) * 0.5).astype(np.float64)
        wfdb.wrsamp(
            "sharee_001",
            fs=128,
            units=["mV"],
            sig_name=["III"],
            p_signal=sig,
            write_dir=str(tmp_dir),
        )
        # No .label file for SHAREE — labels come from label_map
        cfg = _make_stroke_config(tmp_dir, target_fs=0.0, waveform_sec=10.0)
        records = parse_sharee_dir(str(tmp_dir), label_map={}, config_path=cfg)
        assert len(records) == 1
        assert records[0]["label"] == 0, "Record not in label_map should default to 0"


# ---------------------------------------------------------------------------
# Unit tests — parse_cerevasc_dir (stub)
# ---------------------------------------------------------------------------

def test_parse_cerevasc_raises_not_implemented():
    """parse_cerevasc_dir() raises NotImplementedError (credential blocker)."""
    with pytest.raises(NotImplementedError):
        parse_cerevasc_dir("data/raw/stroke avns/cves", "config_stroke.yaml")


# ---------------------------------------------------------------------------
# Integration tests — real data (excluded from CI)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_parse_mimic3_stroke_integration():
    """Integration: parse real mimic3_stroke dir, expect ~300 records."""
    data_dir = "data/raw/stroke avns/mimic3_stroke"
    config_path = "config_stroke.yaml"
    if not Path(data_dir).is_dir():
        pytest.skip("mimic3_stroke data not found — skipping integration test")
    records = parse_mimic3_stroke_dir(data_dir, config_path)
    assert len(records) > 0, "Expected at least one record from mimic3_stroke"
    labels = [r["label"] for r in records]
    assert set(labels).issubset({0, 1}), "All labels must be binary"
    print(f"\nmimic3_stroke: {len(records)} records — "
          f"stroke={sum(labels)}, control={len(labels)-sum(labels)}")
