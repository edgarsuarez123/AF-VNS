"""
StrokeDataset and stroke DataLoaders — thin subclass of PhysioDataset.

StrokeRecordDict is compatible with PhysioDataset (same subject_id/signal/fs/label
fields), so no logic is duplicated here. The training-loop interface is identical:
each __getitem__ returns (short_t, long_t, label_t, fs).
"""

from pathlib import Path
from typing import List, Optional, Tuple

from torch.utils.data import DataLoader

from .dataloaders import PhysioDataset
from .dataset_parsers import load_config
from .splitter import load_split
from .stroke_parsers import StrokeRecordDict


class StrokeDataset(PhysioDataset):
    """Rolling-window dataset for stroke AVNS records.

    Identical interface to PhysioDataset. Subclassed for type clarity and to
    accept List[StrokeRecordDict] explicitly — no logic changes.

    Returns:
        (short_t, long_t, label_t, fs)
        short_t: (1, waveform_sec*fs)  — CNN input
        long_t:  (1, long_window_sec*fs) — RNN/Transformer input
        label_t: scalar float32 (1.0=stroke, 0.0=control)
        fs:      float
    """

    def __init__(
        self,
        records: List[StrokeRecordDict],
        subject_ids: List[str],
        waveform_sec: float = 10.0,
        long_window_sec: float = 300.0,
        stride_sec: Optional[float] = None,
    ):
        super().__init__(records, subject_ids, waveform_sec, long_window_sec, stride_sec)


def get_stroke_dataloaders(
    records: List[StrokeRecordDict],
    config_path: str = "config_stroke.yaml",
    split_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    shuffle_train: bool = True,
    num_workers: int = 0,
    create_split_if_missing: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders from stroke records and config.

    Args:
        records:               Parsed StrokeRecordDicts (from stroke_parsers).
        config_path:           Path to config_stroke.yaml.
        split_path:            Override split JSON path; defaults to
                               config paths.phase1_split.
        batch_size:            Override batch size; defaults to training.batch_size.
        shuffle_train:         Shuffle training loader.
        num_workers:           DataLoader workers.
        create_split_if_missing: If True and split file missing, creates 70/15/15
                               subject-level split and saves it.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    config = load_config(config_path)
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})

    split_path = split_path or paths_cfg.get(
        "phase1_split", "models/artifacts/stroke_phase1_split.json"
    )
    waveform_sec = float(data_cfg.get("waveform_sec", 10))
    long_sec = float(data_cfg.get("hrv_window_sec", 300))
    stride_sec = data_cfg.get("stride_sec", None)
    if stride_sec is not None:
        stride_sec = float(stride_sec)
    batch_size = batch_size or train_cfg.get("batch_size", 32)

    split_file = Path(split_path)
    if not split_file.exists() and create_split_if_missing:
        from .splitter import create_split
        create_split(records, split_path)

    train_ids, val_ids, test_ids = load_split(split_path)

    train_ds = StrokeDataset(
        records, train_ids, waveform_sec=waveform_sec, long_window_sec=long_sec,
        stride_sec=stride_sec,
    )
    val_ds = StrokeDataset(
        records, val_ids, waveform_sec=waveform_sec, long_window_sec=long_sec,
    )
    test_ds = StrokeDataset(
        records, test_ids, waveform_sec=waveform_sec, long_window_sec=long_sec,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
