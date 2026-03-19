"""
Tests for StrokeDataset and get_stroke_dataloaders (Step 8).

All fixtures use in-memory StrokeRecordDicts — no WFDB files written.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.stroke_dataloaders import StrokeDataset, get_stroke_dataloaders
from src.data.splitter import create_split


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _stroke_records(
    fs: float = 250.0,
    num_subjects: int = 10,
    duration_sec: float = 400.0,
    seed: int = 42,
) -> list:
    """Synthetic StrokeRecordDicts — alternating stroke/control labels."""
    rng = np.random.default_rng(seed)
    n = int(fs * duration_sec)
    records = []
    for i in range(num_subjects):
        records.append({
            "subject_id": f"stroke_subj_{i:03d}",
            "signal": rng.standard_normal(n).astype(np.float64) * 0.5,
            "fs": fs,
            "label": i % 2,
            "session_id": None,
            "epoch_type": None,
            "condition": None,
        })
    return records


def _short_stroke_record(fs: float = 250.0, duration_sec: float = 60.0) -> dict:
    """One record shorter than 300s but longer than 10s."""
    rng = np.random.default_rng(7)
    n = int(fs * duration_sec)
    return {
        "subject_id": "short_subj_000",
        "signal": rng.standard_normal(n).astype(np.float64) * 0.5,
        "fs": fs,
        "label": 1,
        "session_id": None,
        "epoch_type": None,
        "condition": None,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_stroke_dataset_window_shapes():
    """short_t is (1, 10*fs), long_t is (1, 300*fs) for full-length records."""
    fs = 250.0
    records = _stroke_records(fs=fs, num_subjects=5, duration_sec=400.0)
    subject_ids = [r["subject_id"] for r in records]
    ds = StrokeDataset(records, subject_ids, waveform_sec=10.0, long_window_sec=300.0)

    assert len(ds) > 0
    short_t, long_t, label_t, fs_out = ds[0]

    assert short_t.shape == (1, int(10 * fs)), f"short_t shape {short_t.shape}"
    assert long_t.shape == (1, int(300 * fs)), f"long_t shape {long_t.shape}"
    assert label_t.shape == ()
    assert short_t.dtype == torch.float32
    assert long_t.dtype == torch.float32
    assert isinstance(fs_out, (int, float)) and fs_out == fs


def test_stroke_dataset_no_patient_leakage():
    """train/val/test subject sets are disjoint and cover all subjects."""
    records = _stroke_records(num_subjects=10, duration_sec=400.0)
    with tempfile.TemporaryDirectory() as tmp:
        split_path = Path(tmp) / "stroke_split.json"
        train_ids, val_ids, test_ids = create_split(records, str(split_path))

        train_set = set(train_ids)
        val_set = set(val_ids)
        test_set = set(test_ids)

        assert train_set.isdisjoint(val_set), "Train and val must not share subjects"
        assert train_set.isdisjoint(test_set), "Train and test must not share subjects"
        assert val_set.isdisjoint(test_set), "Val and test must not share subjects"
        assert len(train_set) + len(val_set) + len(test_set) == 10

        # Dataset only yields items from its assigned split
        train_ds = StrokeDataset(records, train_ids, waveform_sec=10.0, long_window_sec=300.0)
        for i in range(min(20, len(train_ds))):
            _ = train_ds[i]  # no crash


def test_stroke_dataset_labels_binary():
    """All label_t values are 0.0 or 1.0."""
    records = _stroke_records(num_subjects=6, duration_sec=400.0)
    subject_ids = [r["subject_id"] for r in records]
    ds = StrokeDataset(records, subject_ids, waveform_sec=10.0, long_window_sec=300.0)

    assert len(ds) > 0
    for i in range(len(ds)):
        _, _, label_t, _ = ds[i]
        assert label_t.item() in (0.0, 1.0), f"Unexpected label {label_t.item()} at index {i}"


def test_stroke_dataset_short_record_single_window():
    """A record < 300s but >= 10s produces exactly 1 window."""
    rec = _short_stroke_record(fs=250.0, duration_sec=60.0)
    ds = StrokeDataset([rec], [rec["subject_id"]], waveform_sec=10.0, long_window_sec=300.0)

    assert len(ds) == 1, f"Expected 1 window for short record, got {len(ds)}"
    short_t, long_t, label_t, fs_out = ds[0]
    assert short_t.shape == (1, int(10 * 250)), f"short_t shape {short_t.shape}"
    # long_t length == signal length (< 300s)
    assert long_t.shape[0] == 1
    assert long_t.shape[1] == int(60 * 250), f"long_t shape {long_t.shape}"
    assert label_t.item() == 1.0


def test_get_stroke_dataloaders_returns_three_loaders():
    """get_stroke_dataloaders returns (train, val, test) DataLoaders."""
    import yaml

    records = _stroke_records(num_subjects=10, duration_sec=400.0)
    with tempfile.TemporaryDirectory() as tmp:
        split_path = str(Path(tmp) / "stroke_split.json")
        create_split(records, split_path)

        cfg_path = str(Path(tmp) / "config_stroke.yaml")
        cfg = {
            "data": {"waveform_sec": 10, "hrv_window_sec": 300, "stride_sec": 300},
            "training": {"batch_size": 4},
            "paths": {"phase1_split": split_path},
        }
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)

        train_loader, val_loader, test_loader = get_stroke_dataloaders(
            records, config_path=cfg_path, split_path=split_path
        )

    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None


def test_get_stroke_dataloaders_create_split_if_missing():
    """create_split_if_missing=True creates the split file and returns loaders."""
    import yaml

    records = _stroke_records(num_subjects=10, duration_sec=400.0)
    with tempfile.TemporaryDirectory() as tmp:
        split_path = str(Path(tmp) / "stroke_split.json")

        cfg_path = str(Path(tmp) / "config_stroke.yaml")
        cfg = {
            "data": {"waveform_sec": 10, "hrv_window_sec": 300, "stride_sec": 300},
            "training": {"batch_size": 4},
            "paths": {"phase1_split": split_path},
        }
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)

        assert not Path(split_path).exists()
        train_loader, val_loader, test_loader = get_stroke_dataloaders(
            records, config_path=cfg_path, split_path=split_path,
            create_split_if_missing=True,
        )
        assert Path(split_path).exists()
        assert train_loader is not None
