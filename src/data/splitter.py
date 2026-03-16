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
        raise ValueError("Need at least 3 subjects for split")

    train_ids, rest = train_test_split(
        unique_ids, train_size=train_ratio, random_state=seed, shuffle=True
    )

    # Handle no-test-set case (e.g., Phase 1 pre-training: 85/15/0)
    if test_ratio == 0.0:
        val_ids = rest
        test_ids = []
    else:
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


def create_phase_splits(
    all_ids: List[str],
    phase1_split_path: str,
    phase2_split_path: str,
    seed: int = 42,
) -> dict:
    """Create separate splits for two-phase transfer learning.

    Phase 1 (MIMIC-3 pre-training): 85/15 train/val — all MIMIC-3 records.
    Phase 2 (fine-tuning): 70/15/15 train/val/test — AFDB/NSRDB/LTAFDB only.

    Returns dict with phase1 and phase2 split tuples.
    """
    import re

    mimic_ids = [sid for sid in all_ids if re.match(r'^p\d{6}_', sid)]
    c17_ids = [sid for sid in all_ids if sid.startswith('c17_')]
    phase2_ids = [sid for sid in all_ids
                  if not re.match(r'^p\d{6}_', sid) and not sid.startswith('c17_')]

    # Phase 1: MIMIC-3 only, 85/15 train/val (no test — pre-training only)
    p1_train, p1_val, _ = create_split_from_ids(
        mimic_ids, phase1_split_path,
        train_ratio=0.85, val_ratio=0.15, test_ratio=0.0,
        seed=seed,
    )

    # Phase 2: AFDB + NSRDB + LTAFDB, 70/15/15 train/val/test
    p2_train, p2_val, p2_test = create_split_from_ids(
        phase2_ids, phase2_split_path,
        train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
        seed=seed,
    )

    return {
        "phase1": {"train": p1_train, "val": p1_val},
        "phase2": {"train": p2_train, "val": p2_val, "test": p2_test},
        "excluded_c17": len(c17_ids),
    }
