"""
Download PhysioNet Computing in Cardiology Challenge 2017 training data.

Dataset: https://physionet.org/content/challenge-2017/1.0.0/
- 8,528 single-lead ECG records (300 Hz, 30-61 seconds each)
- Labels: A (AF), N (Normal), O (Other), ~ (Noisy)
- We keep only A and N; skip O and ~ (not useful for binary AF classification)

Creates .label files (1=AF, 0=Normal) alongside WFDB records for consistency
with the rest of the pipeline (same pattern as MIMIC-III .label files).

Run: .venv/Scripts/python -m src.data.download_challenge2017
"""

import csv
import io
import logging
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# PhysioNet open-access URL (no credentials required)
ZIP_URL = "https://physionet.org/files/challenge-2017/1.0.0/training2017.zip"
OUT_DIR = Path("data/raw/challenge2017")


def _download_with_progress(url: str, dest: Path) -> None:
    """Download file with progress reporting."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            if block_num % 100 == 0:
                logger.info("  %.1f / %.1f MB (%d%%)", mb, total_mb, pct)

    logger.info("Downloading %s ...", url)
    urlretrieve(url, str(dest), reporthook=_report)
    logger.info("Download complete: %s (%.1f MB)", dest, dest.stat().st_size / (1024 * 1024))


def _create_label_files(extract_dir: Path) -> dict:
    """Read REFERENCE.csv and create .label files for AF (1) and Normal (0).

    Returns dict with counts: {'af': N, 'normal': N, 'skipped': N}.
    """
    # REFERENCE.csv may be inside a subfolder (training2017/) after extraction
    ref_path = None
    for candidate in [extract_dir / "REFERENCE.csv",
                      extract_dir / "training2017" / "REFERENCE.csv"]:
        if candidate.exists():
            ref_path = candidate
            break

    if ref_path is None:
        logger.error("REFERENCE.csv not found in %s", extract_dir)
        return {"af": 0, "normal": 0, "skipped": 0}

    # Determine where the actual .hea files live
    hea_dir = ref_path.parent

    counts = {"af": 0, "normal": 0, "skipped": 0}
    with open(ref_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            record_name = row[0].strip()
            label_str = row[1].strip()

            if label_str == "A":
                label = 1
                counts["af"] += 1
            elif label_str == "N":
                label = 0
                counts["normal"] += 1
            else:
                counts["skipped"] += 1
                continue

            # Write .label file next to .hea
            label_path = hea_dir / f"{record_name}.label"
            label_path.write_text(str(label))

    return counts


def main():
    if OUT_DIR.exists() and list(OUT_DIR.glob("*.hea")):
        n_hea = len(list(OUT_DIR.glob("*.hea")))
        n_label = len(list(OUT_DIR.glob("*.label")))
        logger.info("Challenge 2017 data already exists: %d .hea, %d .label files", n_hea, n_label)
        if n_label == 0:
            logger.info("No .label files found — regenerating from REFERENCE.csv")
            counts = _create_label_files(OUT_DIR)
            logger.info("Labels: %d AF, %d Normal, %d skipped (Other/Noisy)", counts["af"], counts["normal"], counts["skipped"])
        return

    # Check if subfolder has data (ZIP extracts to training2017/)
    sub = OUT_DIR / "training2017"
    if sub.exists() and list(sub.glob("*.hea")):
        logger.info("Data in subfolder training2017/ — moving to %s", OUT_DIR)
        # Move files up one level
        for f in sub.iterdir():
            dest = OUT_DIR / f.name
            if not dest.exists():
                f.rename(dest)
        counts = _create_label_files(OUT_DIR)
        logger.info("Labels: %d AF, %d Normal, %d skipped", counts["af"], counts["normal"], counts["skipped"])
        return

    # Download ZIP
    zip_path = OUT_DIR / "training2017.zip"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        _download_with_progress(ZIP_URL, zip_path)

    # Extract
    logger.info("Extracting %s ...", zip_path)
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(OUT_DIR))
    logger.info("Extraction complete")

    # ZIP may extract into training2017/ subfolder — flatten if so
    sub = OUT_DIR / "training2017"
    if sub.exists() and sub.is_dir():
        logger.info("Flattening training2017/ subfolder ...")
        for f in list(sub.iterdir()):
            dest = OUT_DIR / f.name
            if not dest.exists():
                f.rename(dest)
        # Remove empty subfolder
        try:
            sub.rmdir()
        except OSError:
            pass

    # Create .label files
    counts = _create_label_files(OUT_DIR)
    logger.info("Labels created: %d AF, %d Normal, %d skipped (Other/Noisy)",
                counts["af"], counts["normal"], counts["skipped"])

    # Clean up ZIP to save disk space
    if zip_path.exists():
        zip_path.unlink()
        logger.info("Removed %s to save space", zip_path)

    # Verify
    n_hea = len(list(OUT_DIR.glob("*.hea")))
    n_label = len(list(OUT_DIR.glob("*.label")))
    logger.info("Done. %d .hea records, %d .label files in %s", n_hea, n_label, OUT_DIR)


if __name__ == "__main__":
    main()
