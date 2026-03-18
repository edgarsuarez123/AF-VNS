"""
Add MIMIC-III subjects to split.json for mixed-source training.

Strategy:
  - Reserve 10 AF + 10 NSR MIMIC records as holdout (models/artifacts/mimic3_holdout.json)
  - Add remaining 20 AF + 20 NSR to existing split: 14+14 to train, 6+6 to val
  - Test set stays afdb/nsrdb only (clean in-distribution test)

Run: .venv/Scripts/python -m src.data.create_mixed_split
"""

import json
import random
import sys
from pathlib import Path

if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

MIMIC_DIR = Path("data/raw/AF avns/mimic3")
SPLIT_PATH = Path("models/artifacts/split.json")
HOLDOUT_PATH = Path("models/artifacts/mimic3_holdout.json")


def main():
    if not SPLIT_PATH.exists():
        print(f"ERROR: {SPLIT_PATH} not found. Run training once to create it.")
        sys.exit(1)

    with open(SPLIT_PATH) as f:
        split = json.load(f)

    # Check if MIMIC already added
    existing_ids = set(split["train"]) | set(split["val"]) | set(split["test"])
    mimic_ids_in_split = [s for s in existing_ids if s.startswith("p")]
    if mimic_ids_in_split:
        print(f"MIMIC records already in split ({len(mimic_ids_in_split)} found). Skipping.")
        return

    # Collect all MIMIC records with labels
    af_records = []
    nsr_records = []
    for label_file in sorted(MIMIC_DIR.glob("*.label")):
        name = label_file.stem
        if name.upper().startswith("DIAGNOSES"):
            continue
        label = int(label_file.read_text().strip())
        if label == 1:
            af_records.append(name)
        else:
            nsr_records.append(name)

    print(f"Found {len(af_records)} AF + {len(nsr_records)} NSR MIMIC records")

    # Deterministic shuffle with seed
    random.seed(42)
    random.shuffle(af_records)
    random.shuffle(nsr_records)

    # Reserve 10+10 as holdout
    holdout_af = af_records[:10]
    holdout_nsr = nsr_records[:10]
    pool_af = af_records[10:]   # 20 AF for training
    pool_nsr = nsr_records[10:]  # 20 NSR for training

    # Split 70/30 train/val (test stays afdb/nsrdb only)
    n_train_af = 14   # 70% of 20
    n_train_nsr = 14
    new_train = pool_af[:n_train_af] + pool_nsr[:n_train_nsr]
    new_val = pool_af[n_train_af:] + pool_nsr[n_train_nsr:]

    split["train"] = split["train"] + new_train
    split["val"] = split["val"] + new_val

    with open(SPLIT_PATH, "w") as f:
        json.dump(split, f, indent=2)

    holdout = {
        "af": holdout_af,
        "nsr": holdout_nsr,
        "all": holdout_af + holdout_nsr,
    }
    HOLDOUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HOLDOUT_PATH, "w") as f:
        json.dump(holdout, f, indent=2)

    print(f"Updated split.json: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")
    print(f"  Added to train: {len(new_train)} MIMIC ({n_train_af} AF + {n_train_nsr} NSR)")
    print(f"  Added to val:   {len(new_val)} MIMIC ({len(pool_af) - n_train_af} AF + {len(pool_nsr) - n_train_nsr} NSR)")
    print(f"Saved holdout: {len(holdout_af)} AF + {len(holdout_nsr)} NSR -> {HOLDOUT_PATH}")


if __name__ == "__main__":
    main()
