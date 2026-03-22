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
    _select_resp_channel,
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
# Unit tests — parse_cerevasc_dir
# ---------------------------------------------------------------------------

def _write_cves_record(
    proto_dir: Path,
    stem: str,
    n_samples: int = 5000,
    fs: int = 500,
    sig_name: str = "ecg",
    unit: str = "mV",
) -> None:
    """Write a minimal WFDB record into a CVES protocol subdirectory."""
    rng = np.random.default_rng(99)
    sig = (rng.standard_normal((n_samples, 1)) * 0.5).astype(np.float64)
    wfdb.wrsamp(
        stem,
        fs=fs,
        units=[unit],
        sig_name=[sig_name],
        p_signal=sig,
        write_dir=str(proto_dir),
    )


def _make_cves_dir(tmp_dir: Path) -> Path:
    """Build a minimal synthetic CVES directory structure with 3 subjects:
    - s0044: stroke (ID 44 in CVES_STROKE_IDS)
    - s0165: control (ID 165 in CVES_CONTROL_IDS)
    - s0999: unmatched (no label — should be skipped)
    """
    ss_dir = tmp_dir / "data" / "sit-stand"
    ss_dir.mkdir(parents=True)
    _write_cves_record(ss_dir, "s0044-sit-stand")
    _write_cves_record(ss_dir, "s0165-sit-stand")
    _write_cves_record(ss_dir, "s0999-sit-stand")
    return tmp_dir


def test_parse_cerevasc_returns_list():
    """parse_cerevasc_dir returns a list (not raises)."""
    with tempfile.TemporaryDirectory() as tmp:
        cves_dir = _make_cves_dir(Path(tmp))
        cfg = _make_stroke_config(Path(tmp), target_fs=0.0)
        result = parse_cerevasc_dir(str(cves_dir), cfg)
        assert isinstance(result, list)


def test_parse_cerevasc_skips_unmatched():
    """Subjects with no label entry (s0999) are skipped; only labeled subjects returned."""
    with tempfile.TemporaryDirectory() as tmp:
        cves_dir = _make_cves_dir(Path(tmp))
        cfg = _make_stroke_config(Path(tmp), target_fs=0.0)
        records = parse_cerevasc_dir(str(cves_dir), cfg)
        assert len(records) == 2
        ids = {r["subject_id"] for r in records}
        assert "cves_0999" not in ids, "Unmatched subject should be skipped"


def test_parse_cerevasc_labels_binary():
    """All returned labels are exactly 0 or 1."""
    with tempfile.TemporaryDirectory() as tmp:
        cves_dir = _make_cves_dir(Path(tmp))
        cfg = _make_stroke_config(Path(tmp), target_fs=0.0)
        records = parse_cerevasc_dir(str(cves_dir), cfg)
        for r in records:
            assert r["label"] in (0, 1), f"Non-binary label {r['label']}"


def test_parse_cerevasc_schema():
    """All required StrokeRecordDict keys present in every record."""
    required = {"subject_id", "signal", "fs", "label", "session_id", "epoch_type", "condition"}
    with tempfile.TemporaryDirectory() as tmp:
        cves_dir = _make_cves_dir(Path(tmp))
        cfg = _make_stroke_config(Path(tmp), target_fs=0.0)
        records = parse_cerevasc_dir(str(cves_dir), cfg)
        assert len(records) > 0
        for r in records:
            assert required.issubset(r.keys()), f"Missing keys: {required - set(r.keys())}"


def test_parse_cerevasc_subject_id_format():
    """subject_id follows cves_{numeric} pattern for all records."""
    import re
    with tempfile.TemporaryDirectory() as tmp:
        cves_dir = _make_cves_dir(Path(tmp))
        cfg = _make_stroke_config(Path(tmp), target_fs=0.0)
        records = parse_cerevasc_dir(str(cves_dir), cfg)
        for r in records:
            assert re.match(r"^cves_\d+$", r["subject_id"]), \
                f"subject_id '{r['subject_id']}' doesn't match cves_\\d+"


def test_parse_cerevasc_stroke_label_correct():
    """s0044 (stroke) gets label=1, s0165 (control) gets label=0."""
    with tempfile.TemporaryDirectory() as tmp:
        cves_dir = _make_cves_dir(Path(tmp))
        cfg = _make_stroke_config(Path(tmp), target_fs=0.0)
        records = parse_cerevasc_dir(str(cves_dir), cfg)
        by_id = {r["subject_id"]: r["label"] for r in records}
        assert by_id.get("cves_0044") == 1, "s0044 should be stroke (label=1)"
        assert by_id.get("cves_0165") == 0, "s0165 should be control (label=0)"


@pytest.mark.integration
def test_parse_cerevasc_integration():
    """Integration: parse real CVES dir, check count and label distribution."""
    data_dir = "data/raw/stroke avns/cves"
    config_path = "config_stroke.yaml"
    if not Path(data_dir).is_dir():
        pytest.skip("CVES data not found — skipping integration test")
    records = parse_cerevasc_dir(data_dir, config_path)
    assert len(records) >= 50, f"Expected ≥50 records, got {len(records)}"
    labels = [r["label"] for r in records]
    assert set(labels).issubset({0, 1})
    stroke = sum(labels)
    ctrl = len(labels) - stroke
    subj_ids = set(r["subject_id"] for r in records)
    print(f"\nCVES: {len(records)} records, {stroke} stroke, {ctrl} control, "
          f"{len(subj_ids)} unique subjects, fs={records[0]['fs']}")


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


# ---------------------------------------------------------------------------
# Unit tests — _select_resp_channel
# ---------------------------------------------------------------------------

def test_select_resp_channel_flow_rate_priority():
    """flow_rate is preferred over thermst."""
    idx, name = _select_resp_channel(["marker", "ecg", "thermst", "flow_rate"])
    assert idx == 3
    assert name == "flow_rate"


def test_select_resp_channel_thermst_fallback():
    """thermst is used when flow_rate is absent."""
    idx, name = _select_resp_channel(["marker", "ecg", "thermst"])
    assert idx == 2
    assert name == "thermst"


def test_select_resp_channel_resp_fallback():
    """resp is used when flow_rate and thermst are absent."""
    idx, name = _select_resp_channel(["ecg", "resp"])
    assert idx == 1
    assert name == "resp"


def test_select_resp_channel_none():
    """Returns (None, None) when no respiratory channel exists."""
    idx, name = _select_resp_channel(["ecg", "abp", "marker"])
    assert idx is None
    assert name is None


def test_select_resp_channel_case_insensitive():
    """Channel name matching is case-insensitive."""
    idx, name = _select_resp_channel(["ECG", "Flow_Rate", "Thermst"])
    assert idx == 1
    assert name == "flow_rate"


# ---------------------------------------------------------------------------
# Unit tests — CVES respiratory channel extraction
# ---------------------------------------------------------------------------

def _write_cves_multichannel_record(
    proto_dir: Path,
    stem: str,
    n_samples: int = 5000,
    fs: int = 500,
    sig_names=None,
    units=None,
) -> None:
    """Write a multi-channel WFDB record mimicking CVES hardware output."""
    if sig_names is None:
        sig_names = ["marker", "ecg", "abp", "thermst", "flow_rate"]
    if units is None:
        units = ["NU", "mV", "mmHg", "NU", "NU"]
    n_ch = len(sig_names)
    rng = np.random.default_rng(99)
    sig = (rng.standard_normal((n_samples, n_ch)) * 0.5).astype(np.float64)
    wfdb.wrsamp(
        stem,
        fs=fs,
        units=units,
        sig_name=sig_names,
        p_signal=sig,
        write_dir=str(proto_dir),
    )


def _make_cves_dir_multichannel(tmp_dir: Path) -> Path:
    """CVES directory with multi-channel records (including flow_rate + thermst)."""
    ss_dir = tmp_dir / "data" / "sit-stand"
    ss_dir.mkdir(parents=True)
    _write_cves_multichannel_record(ss_dir, "s0044-sit-stand")
    _write_cves_multichannel_record(ss_dir, "s0165-sit-stand")
    return tmp_dir


def test_parse_cerevasc_resp_signal_present():
    """Multi-channel CVES records return non-None resp_signal."""
    with tempfile.TemporaryDirectory() as tmp:
        cves_dir = _make_cves_dir_multichannel(Path(tmp))
        cfg = _make_stroke_config(Path(tmp), target_fs=0.0)
        records = parse_cerevasc_dir(str(cves_dir), cfg)
        assert len(records) == 2
        for r in records:
            assert r["resp_signal"] is not None, f"{r['subject_id']} missing resp_signal"
            assert r["resp_channel"] == "flow_rate", "Should prefer flow_rate"


def test_parse_cerevasc_resp_signal_shape():
    """resp_signal is 1D and same length as ECG signal."""
    with tempfile.TemporaryDirectory() as tmp:
        cves_dir = _make_cves_dir_multichannel(Path(tmp))
        cfg = _make_stroke_config(Path(tmp), target_fs=0.0)
        records = parse_cerevasc_dir(str(cves_dir), cfg)
        for r in records:
            assert r["resp_signal"].ndim == 1
            assert len(r["resp_signal"]) == len(r["signal"]), \
                "resp_signal and signal must have same length"


def test_parse_cerevasc_resp_signal_resampled():
    """resp_signal is resampled alongside ECG when target_fs differs."""
    with tempfile.TemporaryDirectory() as tmp:
        cves_dir = _make_cves_dir_multichannel(Path(tmp))
        cfg = _make_stroke_config(Path(tmp), target_fs=250.0)
        records = parse_cerevasc_dir(str(cves_dir), cfg)
        for r in records:
            assert r["resp_signal"] is not None
            assert len(r["resp_signal"]) == len(r["signal"]), \
                "After resampling, resp_signal and signal must have same length"


def test_parse_cerevasc_resp_missing_single_channel():
    """Single-channel ECG-only CVES records return resp_signal=None."""
    with tempfile.TemporaryDirectory() as tmp:
        cves_dir = _make_cves_dir(Path(tmp))  # uses original single-channel fixture
        cfg = _make_stroke_config(Path(tmp), target_fs=0.0)
        records = parse_cerevasc_dir(str(cves_dir), cfg)
        for r in records:
            assert r["resp_signal"] is None
            assert r["resp_channel"] is None


def test_parse_cerevasc_thermst_only():
    """When only thermst is available (no flow_rate), it is selected."""
    with tempfile.TemporaryDirectory() as tmp:
        ss_dir = Path(tmp) / "data" / "sit-stand"
        ss_dir.mkdir(parents=True)
        _write_cves_multichannel_record(
            ss_dir, "s0044-sit-stand",
            sig_names=["ecg", "thermst"],
            units=["mV", "NU"],
        )
        cfg = _make_stroke_config(Path(tmp), target_fs=0.0)
        records = parse_cerevasc_dir(str(Path(tmp)), cfg)
        assert len(records) == 1
        assert records[0]["resp_channel"] == "thermst"
        assert records[0]["resp_signal"] is not None


def test_parse_mimic3_resp_fields_none():
    """MIMIC-3 records always have resp_signal=None, resp_channel=None."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _write_wfdb_record(tmp_dir, "rec001", n_samples=5000, fs=125, label=1)
        cfg = _make_stroke_config(tmp_dir)
        records = parse_mimic3_stroke_dir(str(tmp_dir), cfg)
        assert len(records) == 1
        assert records[0]["resp_signal"] is None
        assert records[0]["resp_channel"] is None


def test_parse_sharee_resp_fields_none():
    """SHaRe records always have resp_signal=None, resp_channel=None."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
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
        cfg = _make_stroke_config(tmp_dir, target_fs=0.0, waveform_sec=10.0)
        records = parse_sharee_dir(str(tmp_dir), label_map={}, config_path=cfg)
        assert len(records) == 1
        assert records[0]["resp_signal"] is None
        assert records[0]["resp_channel"] is None


@pytest.mark.integration
def test_parse_cerevasc_integration_resp_channels():
    """Integration: real CVES records should have resp_signal for most protocols."""
    data_dir = "data/raw/stroke avns/cves"
    config_path = "config_stroke.yaml"
    if not Path(data_dir).is_dir():
        pytest.skip("CVES data not found")
    records = parse_cerevasc_dir(data_dir, config_path)
    n_with_resp = sum(1 for r in records if r["resp_signal"] is not None)
    n_total = len(records)
    print(f"\nCVES resp channels: {n_with_resp}/{n_total} records have respiratory reference")
    assert n_with_resp > 0, "Expected at least some records with respiratory reference"
