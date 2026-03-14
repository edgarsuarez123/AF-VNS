"""
Create data/splits/train/, data/splits/val/, data/splits/test/ with symlinks to raw files.
Useful for visually differentiating train vs test data. Does not copy data.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from src.data.dataset_parsers import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = "config.yaml"


def main(config_path: str = CONFIG_PATH, split_path: Optional[str] = None, splits_dir: Optional[str] = None):
    config = load_config(config_path)
    data_cfg = config.get("data", {})
    raw_dir = Path(data_cfg.get("raw_dir", "data/raw"))
    split_path = split_path or data_cfg.get("split_path", "models/artifacts/split.json")
    splits_dir = splits_dir or data_cfg.get("splits_dir", "data/splits")

    if not Path(split_path).is_file():
        logger.error("Split file not found: %s. Run training once to create it.", split_path)
        sys.exit(1)

    with open(split_path) as f:
        data = json.load(f)
    train_ids = set(data["train"])
    val_ids = set(data["val"])
    test_ids = set(data["test"])

    def symlink_file(src: Path, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            return
        try:
            os.symlink(src.resolve(), dst)
        except OSError as e:
            if hasattr(e, "winerror") and e.winerror == 1314:  # Admin required for symlinks on Windows
                logger.warning("Symlink failed (try running as admin or enable Developer Mode): %s", dst)
            else:
                raise

    for split_name, id_set in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        out_split = Path(splits_dir) / split_name
        # WFDB: afdb and nsrdb
        for db in ("afdb", "nsrdb"):
            db_path = raw_dir / db
            if not db_path.is_dir():
                continue
            for hea in db_path.glob("*.hea"):
                subject_id = hea.stem
                if subject_id not in id_set:
                    continue
                for ext in (".hea", ".dat"):
                    f = db_path / (hea.stem + ext)
                    if f.exists():
                        symlink_file(f, out_split / db / f.name)
        # MIMIC-III: any file (csv, edf) - stem as subject_id
        mimic_path = raw_dir / "mimic3"
        if mimic_path.is_dir():
            for f in mimic_path.iterdir():
                if f.is_file() and f.suffix.lower() in (".csv", ".edf"):
                    subject_id = f.stem
                    if subject_id not in id_set:
                        continue
                    symlink_file(f, out_split / "mimic3" / f.name)

    logger.info("Materialized split into %s (train=%d, val=%d, test=%d subjects)",
                splits_dir, len(train_ids), len(val_ids), len(test_ids))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Create data/splits/{train,val,test} with symlinks.")
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument("--split-path", default=None)
    p.add_argument("--splits-dir", default=None)
    args = p.parse_args()
    main(config_path=args.config, split_path=args.split_path, splits_dir=args.splits_dir)
