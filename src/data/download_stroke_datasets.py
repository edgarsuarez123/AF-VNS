"""
Download PhysioNet stroke-related datasets into data/raw/stroke avns/.

Datasets:
  - Cerevasc (cves/1.0.0): 120 records, stroke vs control ECG + multimodal.
    Only ECG-relevant subdirectories are downloaded (skips 98 GB EMG data).
  - SHAREE (shareedb/1.0.0): 139 24-hour Holter ECG recordings, 128 Hz.

Usage:
  .venv\\Scripts\\python -m src.data.download_stroke_datasets --dataset cves
  .venv\\Scripts\\python -m src.data.download_stroke_datasets --dataset shareedb
  .venv\\Scripts\\python -m src.data.download_stroke_datasets --dataset all
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

STROKE_RAW_DIR = Path("data/raw/stroke avns")

DATASETS = {
    "cves": {
        "db_dir": "cves",
        "out_subdir": "cves",
        "description": "Cerebral Vasoregulation in Elderly with Stroke (120 subjects)",
        "ecg_prefixes": [
            "data/head-up-tilt/",
            "data/sit-stand-balance/",
            "data/sit-stand/",
            "data/transcranial-doppler/",
        ],
    },
    "shareedb": {
        "db_dir": "shareedb",
        "out_subdir": "shareedb",
        "description": "SHAREE — 139 24-hour Holter ECG (128 Hz, hypertensive patients)",
        "ecg_prefixes": None,  # download all records
    },
}


def _download_dataset(key: str) -> None:
    try:
        import wfdb
    except ImportError:
        raise ImportError("wfdb is required; install with: pip install wfdb")

    info = DATASETS[key]
    out_dir = STROKE_RAW_DIR / info["out_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    db_dir = info["db_dir"]
    logger.info("=== Downloading %s ===", info["description"])
    logger.info("PhysioNet DB: %s -> %s", db_dir, out_dir)

    record_list = wfdb.get_record_list(db_dir)
    logger.info("Total records in RECORDS file: %d", len(record_list))

    ecg_prefixes = info.get("ecg_prefixes")
    if ecg_prefixes is not None:
        original_count = len(record_list)
        record_list = [
            r for r in record_list
            if any(r.startswith(p) for p in ecg_prefixes)
        ]
        logger.info(
            "Filtered to %d ECG-relevant records (skipped %d non-ECG)",
            len(record_list), original_count - len(record_list),
        )

    existing_hea = set(p.stem for p in out_dir.rglob("*.hea"))
    if existing_hea:
        before = len(record_list)
        record_list = [r for r in record_list if Path(r).name not in existing_hea]
        logger.info("Skipping %d already-downloaded records", before - len(record_list))

    if not record_list:
        logger.info("All records already downloaded. Nothing to do.")
        return

    logger.info("Downloading %d records ...", len(record_list))
    wfdb.dl_database(db_dir, str(out_dir), records=record_list, keep_subdirs=True, overwrite=False)
    logger.info("Done: %s -> %s", key, out_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Download stroke-related PhysioNet datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    args = parser.parse_args()

    STROKE_RAW_DIR.mkdir(parents=True, exist_ok=True)

    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for key in targets:
        try:
            _download_dataset(key)
        except Exception as e:
            logger.error("Failed to download %s: %s", key, e, exc_info=True)


if __name__ == "__main__":
    main()
