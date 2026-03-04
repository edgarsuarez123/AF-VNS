"""
Download PhysioNet data into data/raw/ for offline training (FR-1.2).

- MIT-BIH AF Database (afdb): AF phenotype classification, rhythm analysis.
- PhysioNet Normal Sinus Rhythm (nsrdb): healthy baseline comparisons.
- MIMIC-III: reads from a configured directory; you place exported files there.
  See README or docstring for expected layout. If directory is missing/empty, logs warning and skips.
"""

import json
import logging
import os
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PhysioNet database identifiers
AFDB = "afdb"   # MIT-BIH Atrial Fibrillation Database
NSRDB = "nsrdb" # Normal Sinus Rhythm Database


def load_config(config_path: str = "config.yaml") -> dict:
    """Load config; use project root if needed."""
    if not os.path.isabs(config_path) and not os.path.isfile(config_path):
        root = Path(__file__).resolve().parents[2]
        config_path = root / config_path
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_wfdb_database(db_name: str, out_dir: str) -> None:
    """Download a WFDB/PhysioNet database by reading each record and writing to out_dir."""
    try:
        import wfdb
    except ImportError:
        raise ImportError("wfdb is required; install with: pip install wfdb")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        record_list = wfdb.get_record_list(db_name)
    except Exception as e:
        logger.warning("Could not get record list for %s: %s. Skipping.", db_name, e)
        return

    for i, record_name in enumerate(record_list):
        try:
            record = wfdb.rdrecord(record_name, pn_dir=db_name)
            # Save to our layout: out_dir/record_name (.dat and .hea)
            sig = record.p_signal if record.p_signal is not None else record.d_signal
            if hasattr(record, "wrsamp"):
                record.wrsamp(write_dir=str(out_path))
            else:
                wfdb.wrsamp(
                    record_name,
                    fs=record.fs,
                    units=getattr(record, "units", None),
                    sig_name=getattr(record, "sig_name", None),
                    p_signal=sig,
                    write_dir=str(out_path),
                )
            logger.info("[%s] %s (%d/%d)", db_name, record_name, i + 1, len(record_list))
        except Exception as e:
            logger.warning("Failed to download %s/%s: %s", db_name, record_name, e)


def ensure_mimic3_dir(mimic3_subdir: str) -> bool:
    """Check MIMIC-III directory exists and has content; log warning if not."""
    path = Path(mimic3_subdir)
    if not path.exists():
        logger.warning(
            "MIMIC-III directory not found: %s. Place exported CSV/EDF files there. Skipping MIMIC-III.",
            path,
        )
        return False
    contents = list(path.iterdir()) if path.is_dir() else []
    if not contents:
        logger.warning(
            "MIMIC-III directory is empty: %s. Add exported files (see README). Skipping MIMIC-III.",
            path,
        )
        return False
    logger.info("MIMIC-III directory present: %s (%d items).", path, len(contents))
    return True


def main() -> None:
    """Download MIT-BIH AF, Normal Sinus Rhythm; check MIMIC-III dir."""
    config = load_config()
    raw_dir = config.get("data", {}).get("raw_dir", "data/raw")
    mimic3_subdir = config.get("data", {}).get("mimic3_subdir", "data/raw/mimic3")
    base = Path(raw_dir)
    base.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading MIT-BIH AF Database to %s/afdb", base)
    download_wfdb_database(AFDB, str(base / "afdb"))

    logger.info("Downloading Normal Sinus Rhythm Database to %s/nsrdb", base)
    download_wfdb_database(NSRDB, str(base / "nsrdb"))

    ensure_mimic3_dir(mimic3_subdir)


if __name__ == "__main__":
    main()
