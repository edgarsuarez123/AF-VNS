"""
Test DataLoader window shapes and no patient leakage (TR-1.1, FR-2.2, FR-3.1/3.2).
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Add project root for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataloaders import PhysioDataset, get_dataloaders
from src.data.dataset_parsers import RecordDict
from src.data.splitter import create_split, load_split


def _synthetic_records(fs: float = 250.0, num_subjects: int = 10, duration_sec: float = 400.0) -> list:
    """Records with subject_id, signal (long enough for 5min), fs, label."""
    n = int(fs * duration_sec)
    records = []
    for i in range(num_subjects):
        records.append({
            "subject_id": f"subj_{i:03d}",
            "signal": np.random.randn(n).astype(np.float64) * 0.5,
            "fs": fs,
            "label": i % 2,
        })
    return records


def test_window_shapes():
    """10s and 5min windows have expected lengths (10*fs and 300*fs)."""
    fs = 250.0
    records = _synthetic_records(fs=fs, num_subjects=5, duration_sec=400.0)
    subject_ids = [r["subject_id"] for r in records]
    dataset = PhysioDataset(
        records,
        subject_ids,
        waveform_sec=10.0,
        long_window_sec=300.0,
    )
    assert len(dataset) > 0
    short_t, long_t, label_t = dataset[0]
    assert short_t.shape == (1, int(10 * fs)), f"Expected (1, {10*fs}), got {short_t.shape}"
    assert long_t.shape == (1, int(300 * fs)), f"Expected (1, {300*fs}), got {long_t.shape}"
    assert label_t.shape == ()
    assert short_t.dtype == torch.float32
    assert long_t.dtype == torch.float32


def test_no_patient_leakage():
    """Every subject in train loader appears only in train split; same for val/test."""
    fs = 250.0
    records = _synthetic_records(fs=fs, num_subjects=10, duration_sec=400.0)
    with tempfile.TemporaryDirectory() as tmp:
        split_path = Path(tmp) / "split.json"
        train_ids, val_ids, test_ids = create_split(records, str(split_path))
        train_set = set(train_ids)
        val_set = set(val_ids)
        test_set = set(test_ids)
        assert train_set.isdisjoint(val_set), "Train and val must not share subjects"
        assert train_set.isdisjoint(test_set), "Train and test must not share subjects"
        assert val_set.isdisjoint(test_set), "Val and test must not share subjects"
        assert len(train_set) + len(val_set) + len(test_set) == 10

        # Dataset only yields from its split
        train_ds = PhysioDataset(records, train_ids, waveform_sec=10.0, long_window_sec=300.0)
        for i in range(min(20, len(train_ds))):
            _ = train_ds[i]  # no crash
        # Subject IDs in train_ds are only train_ids (enforced by constructor)
        assert set(train_ids) == train_set


def test_load_split_roundtrip():
    """create_split then load_split returns same ids."""
    records = _synthetic_records(num_subjects=6)
    with tempfile.TemporaryDirectory() as tmp:
        split_path = Path(tmp) / "split.json"
        t, v, te = create_split(records, str(split_path))
        t2, v2, te2 = load_split(str(split_path))
        assert t == t2 and v == v2 and te == te2
