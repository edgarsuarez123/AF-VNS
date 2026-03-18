"""
Download PhysioNet Long-Term AF Database (ltafdb) — AF segments only.

84 records, each ~21 hours, 128 Hz, 2-channel ECG.
Instead of downloading full records (~3GB total), this script:
1. Streams each record from PhysioNet
2. Reads rhythm annotations to find (AFIB segments
3. Saves only the AF segments as individual WFDB files with .label=1
4. Discards non-AF portions

Result: ~2,654 AF segment files (>=30s each) in data/raw/ltafdb/
Total disk: ~200-400 MB (AF segments only, not full 3GB)

Run: .venv/Scripts/python -m src.data.download_ltafdb
"""

import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path("data/raw/AF avns/ltafdb")
MIN_DURATION_SEC = 30  # minimum AF segment duration


def main():
    try:
        import wfdb
    except ImportError:
        raise ImportError("wfdb is required")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing_labels = list(OUT_DIR.glob("*.label"))
    if len(existing_labels) >= 2000:
        logger.info("ltafdb AF segments already exist (%d .label files). Skipping.", len(existing_labels))
        return

    record_list = wfdb.get_record_list("ltafdb")
    logger.info("Processing %d ltafdb records (streaming from PhysioNet, saving AF segments only) ...", len(record_list))

    total_segments = 0
    total_af_hours = 0

    for i, record_name in enumerate(record_list):
        # Check if this record's segments already exist
        existing = list(OUT_DIR.glob(f"{record_name}_s*.hea"))
        if existing:
            total_segments += len(existing)
            logger.info("[%d/%d] %s — %d segments already exist, skipping",
                        i + 1, len(record_list), record_name, len(existing))
            continue

        try:
            # Read annotations first (small, fast) to find AF segments
            ann = wfdb.rdann(record_name, "atr", pn_dir="ltafdb")
            header = wfdb.rdheader(record_name, pn_dir="ltafdb")
            fs = header.fs
            sig_len = header.sig_len
        except Exception as e:
            logger.warning("[%d/%d] %s — failed to read annotations: %s",
                           i + 1, len(record_list), record_name, e)
            continue

        # Find AFIB segment boundaries
        afib_segments = []
        current_rhythm = None
        afib_start = None

        for idx in range(len(ann.aux_note)):
            note = ann.aux_note[idx]
            if not note or not note.startswith("("):
                continue
            sample = ann.sample[idx]
            if note in ("(AFIB", "(AFL"):
                if current_rhythm != "AF":
                    afib_start = sample
                    current_rhythm = "AF"
            else:
                if current_rhythm == "AF" and afib_start is not None:
                    if (sample - afib_start) / fs >= MIN_DURATION_SEC:
                        afib_segments.append((afib_start, sample))
                current_rhythm = "other"
                afib_start = None

        if current_rhythm == "AF" and afib_start is not None:
            if (sig_len - afib_start) / fs >= MIN_DURATION_SEC:
                afib_segments.append((afib_start, sig_len))

        if not afib_segments:
            logger.info("[%d/%d] %s — no AF segments >= %ds",
                        i + 1, len(record_list), record_name, MIN_DURATION_SEC)
            continue

        # Now download the full record (we need the signal data)
        try:
            record = wfdb.rdrecord(record_name, pn_dir="ltafdb")
        except Exception as e:
            logger.warning("[%d/%d] %s — failed to read signal: %s",
                           i + 1, len(record_list), record_name, e)
            continue

        signal = record.p_signal if record.p_signal is not None else record.d_signal
        if signal is None:
            continue
        signal = np.asarray(signal, dtype=np.float64)

        # Take first channel only (ECG)
        if signal.ndim == 2:
            signal = signal[:, 0]

        # Save each AF segment as a separate WFDB file
        rec_segs = 0
        for seg_idx, (start, end) in enumerate(afib_segments):
            seg_signal = signal[start:end]
            seg_name = f"{record_name}_s{seg_idx}"

            # Write WFDB file (single-channel)
            wfdb.wrsamp(
                seg_name,
                fs=int(fs),
                units=["mV"],
                sig_name=["ECG"],
                p_signal=seg_signal.reshape(-1, 1),
                write_dir=str(OUT_DIR),
            )
            # Write .label file
            (OUT_DIR / f"{seg_name}.label").write_text("1")
            rec_segs += 1
            total_af_hours += len(seg_signal) / fs / 3600

        total_segments += rec_segs
        logger.info("[%d/%d] %s — saved %d AF segments (%.1f min AF total)",
                    i + 1, len(record_list), record_name,
                    rec_segs, sum((e - s) / fs / 60 for s, e in afib_segments))

    logger.info("Done. %d AF segments saved to %s (%.0f hours AF total)",
                total_segments, OUT_DIR, total_af_hours)


if __name__ == "__main__":
    main()
