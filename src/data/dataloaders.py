"""
PyTorch Dataset and DataLoader for 10s waveform and 5min windows (FR-2.2, FR-3.1, FR-3.2).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .dataset_parsers import RecordDict, load_config, parse_all
from .splitter import load_split


class PhysioDataset(Dataset):
    """
    Rolling windows from parsed records: 10s for CNN, 5min for future HRV/RNN.
    Only includes subjects in subject_ids and records with sufficient length (skip short).
    """

    def __init__(
        self,
        records: List[RecordDict],
        subject_ids: List[str],
        waveform_sec: float = 10.0,
        long_window_sec: float = 300.0,
        stride_sec: Optional[float] = None,
    ):
        self.waveform_sec = waveform_sec
        self.long_window_sec = long_window_sec
        self.stride_sec = stride_sec or waveform_sec  # non-overlapping by default

        # Index: (record_idx, start_sample) for each valid window
        self._samples: List[Tuple[int, int]] = []
        self._records: List[RecordDict] = []
        subject_set = set(subject_ids)

        for rec in records:
            if rec["subject_id"] not in subject_set:
                continue
            sig = np.asarray(rec["signal"], dtype=np.float64)
            fs = rec["fs"]
            n_short = int(fs * waveform_sec)
            n_long = int(fs * long_window_sec)
            if sig.size < n_long:
                continue
            self._records.append(rec)
            rec_idx = len(self._records) - 1
            stride = int(fs * self.stride_sec)
            for start in range(0, sig.size - n_long + 1, stride):
                self._samples.append((rec_idx, start))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        rec_idx, start = self._samples[i]
        rec = self._records[rec_idx]
        sig = np.asarray(rec["signal"], dtype=np.float64)
        fs = float(rec["fs"])
        n_short = int(fs * self.waveform_sec)
        n_long = int(fs * self.long_window_sec)

        short = sig[start : start + n_short]
        long_ = sig[start : start + n_long]
        label = rec.get("label")
        if label is None:
            label = -1  # sentinel for unlabeled
        label_val = float(label) if label >= 0 else 0.0

        # Channel-first: (1, T); return fs for pipeline (waveform_to_hrv_sequence, waveform_10s_denoised)
        short_t = torch.from_numpy(short).float().unsqueeze(0)
        long_t = torch.from_numpy(long_).float().unsqueeze(0)
        label_t = torch.tensor(label_val, dtype=torch.float32)
        return short_t, long_t, label_t, fs


def get_dataloaders(
    config_path: str = "config.yaml",
    split_path: Optional[str] = None,
    parsed_records: Optional[List[RecordDict]] = None,
    batch_size: Optional[int] = None,
    shuffle_train: bool = True,
    num_workers: int = 0,
    create_split_if_missing: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test DataLoaders from config and split.
    If parsed_records is None, calls parse_all(). If create_split_if_missing and split file
    does not exist, creates split from parsed records and saves it.
    """
    config = load_config(config_path)
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    split_path = split_path or data_cfg.get("split_path", "models/artifacts/split.json")
    waveform_sec = data_cfg.get("waveform_sec", 10)
    long_sec = data_cfg.get("hrv_window_sec", 300)
    stride_sec = data_cfg.get("stride_sec", None)  # None = non-overlapping
    batch_size = batch_size or train_cfg.get("batch_size", 32)

    if parsed_records is None:
        parsed_records = parse_all(config_path)

    split_file = Path(split_path)
    if not split_file.exists() and create_split_if_missing:
        from .splitter import create_split
        create_split(parsed_records, split_path)
    train_ids, val_ids, test_ids = load_split(split_path)

    train_ds = PhysioDataset(
        parsed_records, train_ids, waveform_sec=waveform_sec, long_window_sec=long_sec, stride_sec=stride_sec
    )
    val_ds = PhysioDataset(
        parsed_records, val_ids, waveform_sec=waveform_sec, long_window_sec=long_sec
    )
    test_ds = PhysioDataset(
        parsed_records, test_ids, waveform_sec=waveform_sec, long_window_sec=long_sec
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
