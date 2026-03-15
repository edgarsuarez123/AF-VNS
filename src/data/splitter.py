"""
70/15/15 train/val/test split by subject_id (TR-1.1). Persist so train.py and dataloaders use the same split.
"""

import json
import os
from pathlib import Path
from typing import List, Tuple

import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.isabs(config_path) and not os.path.isfile(config_path):
        root = Path(__file__).resolve().parents[2]
        config_path = root / config_path
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_split(
    parsed_records: list,
    split_path: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split unique subject_ids into train/val/test. Persist to split_path (JSON).
    parsed_records: list of dicts with 'subject_id' key.
    """
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        raise ImportError("scikit-learn is required for create_split")

    subject_ids = list({r["subject_id"] for r in parsed_records})
    if len(subject_ids) < 3:
        raise ValueError("Need at least 3 subjects for 70/15/15 split")

    # First split: train vs (val+test)
    train_ids, rest = train_test_split(
        subject_ids, train_size=train_ratio, random_state=seed, shuffle=True
    )
    # Second split: val vs test from rest
    val_frac = val_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(
        rest, train_size=val_frac, random_state=seed, shuffle=True
    )

    Path(split_path).parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump({"train": train_ids, "val": val_ids, "test": test_ids}, f, indent=2)

    return train_ids, val_ids, test_ids


def create_split_from_ids(
    subject_ids: List[str],
    split_path: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Create train/val/test split from a list of subject IDs directly.

    Same splitting logic as create_split() but accepts IDs instead of parsed
    records — avoids loading signal data just to extract IDs.
    """
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        raise ImportError("scikit-learn is required for create_split_from_ids")

    unique_ids = sorted(set(subject_ids))
    if len(unique_ids) < 3:
        raise ValueError("Need at least 3 subjects for 70/15/15 split")

    train_ids, rest = train_test_split(
        unique_ids, train_size=train_ratio, random_state=seed, shuffle=True
    )
    val_frac = val_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(
        rest, train_size=val_frac, random_state=seed, shuffle=True
    )

    Path(split_path).parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump({"train": train_ids, "val": val_ids, "test": test_ids}, f, indent=2)

    return train_ids, val_ids, test_ids


def load_split(split_path: str) -> Tuple[List[str], List[str], List[str]]:
    """Load train/val/test subject_id lists from JSON."""
    with open(split_path) as f:
        data = json.load(f)
    return data["train"], data["val"], data["test"]
