"""
Download PhysioNet Long-Term AF Database (ltafdb).

84 records, each ~21 hours, 128 Hz, 2-channel ECG.
Contains rhythm annotations with (AFIB segments.
Parser extracts only AF segments for clean AF training data.

Run: .venv/Scripts/python -m src.data.download_ltafdb
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path("data/raw/ltafdb")


def main():
    try:
        import wfdb
    except ImportError:
        raise ImportError("wfdb is required")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing_hea = list(OUT_DIR.glob("*.hea"))
    if len(existing_hea) >= 80:
        logger.info("ltafdb already downloaded (%d .hea files). Skipping.", len(existing_hea))
        return

    record_list = wfdb.get_record_list("ltafdb")
    logger.info("Downloading %d ltafdb records to %s ...", len(record_list), OUT_DIR)

    for i, record_name in enumerate(record_list):
        hea_file = OUT_DIR / f"{record_name}.hea"
        if hea_file.exists():
            logger.info("[%d/%d] %s — already exists, skipping", i + 1, len(record_list), record_name)
            continue
        try:
            record = wfdb.rdrecord(record_name, pn_dir="ltafdb")
            sig = record.p_signal if record.p_signal is not None else record.d_signal
            wfdb.wrsamp(
                record_name,
                fs=record.fs,
                units=getattr(record, "units", None),
                sig_name=getattr(record, "sig_name", None),
                p_signal=sig,
                write_dir=str(OUT_DIR),
            )
            # Also download annotations
            ann = wfdb.rdann(record_name, "atr", pn_dir="ltafdb")
            ann.wrann(write_dir=str(OUT_DIR))
            logger.info("[%d/%d] %s — %.1f hours downloaded",
                        i + 1, len(record_list), record_name,
                        record.sig_len / record.fs / 3600)
        except Exception as e:
            logger.warning("[%d/%d] %s — FAILED: %s", i + 1, len(record_list), record_name, e)

    n_downloaded = len(list(OUT_DIR.glob("*.hea")))
    logger.info("Done. %d records in %s", n_downloaded, OUT_DIR)


if __name__ == "__main__":
    main()
