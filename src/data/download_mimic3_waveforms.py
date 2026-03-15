"""
Download MIMIC-III Waveform Matched Subset for cross-dataset validation.
Selects AF and non-AF patients, downloads ECG waveform segments via wfdb.
Saves as WFDB records in data/raw/mimic3/ for use by dataset_parsers.py.
"""

import argparse
import base64
import logging
import os
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PHYSIONET_DB = "mimic3wdb-matched/1.0"
RECORDS_URL = f"https://physionet.org/files/{PHYSIONET_DB}/RECORDS"


def _auth_header(username: str, password: str) -> str:
    return "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode("ascii")


def _fetch_records(username: str, password: str):
    """Download RECORDS list from PhysioNet."""
    req = urllib.request.Request(RECORDS_URL)
    req.add_header("Authorization", _auth_header(username, password))
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode().strip().split("\n")


def _get_subject_record_map(records):
    """Map subject_id (int) -> record path (e.g. 'p00/p000020/')."""
    out = {}
    for r in records:
        match = re.search(r"p(\d+)/$", r.strip())
        if match:
            sid = int(match.group(1))
            out[sid] = r.strip()
    return out


def _list_wfdb_records_in_dir(record_path: str, username: str, password: str):
    """List .hea files in a patient's waveform directory on PhysioNet."""
    url = f"https://physionet.org/files/{PHYSIONET_DB}/{record_path}"
    req = urllib.request.Request(url)
    req.add_header("Authorization", _auth_header(username, password))
    try:
        with urllib.request.urlopen(req) as resp:
            html = resp.read().decode()
        # Find .hea file links
        hea_files = re.findall(r'href="([^"]+\.hea)"', html)
        return [h.replace(".hea", "") for h in hea_files]
    except Exception:
        return []


def _download_wfdb_record(record_name: str, record_path: str, out_dir: Path,
                          username: str, password: str) -> bool:
    """Download a single WFDB record (.hea + .dat) to out_dir."""
    auth = _auth_header(username, password)
    base_url = f"https://physionet.org/files/{PHYSIONET_DB}/{record_path}"

    for ext in (".hea", ".dat"):
        url = f"{base_url}{record_name}{ext}"
        req = urllib.request.Request(url)
        req.add_header("Authorization", auth)
        try:
            with urllib.request.urlopen(req) as resp:
                data = resp.read()
            dest = out_dir / f"{record_name}{ext}"
            with open(dest, "wb") as f:
                f.write(data)
        except Exception as e:
            logger.warning("Failed to download %s: %s", url, e)
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Download MIMIC-III waveforms for AF validation.")
    parser.add_argument("--username", default=os.environ.get("PHYSIONET_USER"), help="PhysioNet username (or set PHYSIONET_USER env var)")
    parser.add_argument("--password", default=os.environ.get("PHYSIONET_PASS"), help="PhysioNet password (or set PHYSIONET_PASS env var)")
    parser.add_argument("--n-af", type=int, default=30, help="Number of AF patients to download")
    parser.add_argument("--n-control", type=int, default=30, help="Number of non-AF patients to download")
    parser.add_argument("--diagnoses-csv", default="data/raw/mimic3/DIAGNOSES_ICD.csv",
                        help="Path to DIAGNOSES_ICD.csv")
    parser.add_argument("--out-dir", default="data/raw/mimic3", help="Output directory for waveforms")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for patient selection")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.username or not args.password:
        logger.error("PhysioNet credentials required. Set PHYSIONET_USER/PHYSIONET_PASS or use --username/--password.")
        sys.exit(1)

    # Load diagnoses
    logger.info("Loading diagnoses from %s", args.diagnoses_csv)
    df = pd.read_csv(args.diagnoses_csv)
    af_subjects = set(df[df.ICD9_CODE == "42731"].SUBJECT_ID.unique())
    all_subjects = set(df.SUBJECT_ID.unique())
    non_af_subjects = all_subjects - af_subjects
    logger.info("AF patients: %d, Non-AF patients: %d", len(af_subjects), len(non_af_subjects))

    # Get waveform records
    logger.info("Fetching RECORDS list from PhysioNet...")
    records = _fetch_records(args.username, args.password)
    subject_map = _get_subject_record_map(records)
    wave_subjects = set(subject_map.keys())

    af_with_wave = sorted(af_subjects & wave_subjects)
    non_af_with_wave = sorted(non_af_subjects & wave_subjects)
    logger.info("AF with waveforms: %d, Non-AF with waveforms: %d",
                len(af_with_wave), len(non_af_with_wave))

    # Sample
    rng = np.random.default_rng(args.seed)
    # Over-sample to account for patients without long enough records
    oversample = 3
    af_sample = rng.choice(af_with_wave, size=min(args.n_af * oversample, len(af_with_wave)), replace=False)
    ctrl_sample = rng.choice(non_af_with_wave, size=min(args.n_control * oversample, len(non_af_with_wave)), replace=False)

    # Download
    downloaded = {"af": 0, "control": 0}

    targets = {"af": args.n_af, "control": args.n_control}
    for label, subjects in [("af", af_sample), ("control", ctrl_sample)]:
        for sid in subjects:
            if downloaded[label] >= targets[label]:
                break
            # Skip patients already downloaded
            existing = list(out_dir.glob(f"p{sid:06d}_*.hea"))
            if existing:
                logger.info("Skipping patient %d (%s) — already downloaded", sid, label)
                downloaded[label] += 1
                continue
            record_path = subject_map[sid]
            logger.info("Checking patient %d (%s) at %s", sid, label, record_path)

            # List available records for this patient
            rec_names = _list_wfdb_records_in_dir(record_path, args.username, args.password)
            if not rec_names:
                logger.warning("No WFDB records for patient %d, skipping", sid)
                continue

            # Try each record until we find one with ECG, >=30s, and real waveform (fs>1)
            # Prefer shortest qualifying record (30s-900s) to avoid downloading multi-hour files;
            # fall back to any qualifying record if none found in that range.
            TARGET_MAX_SEC = 900  # 15 min cap — we only use 5 min; avoids 26-hour downloads
            rec_name = None
            best_duration = 0
            fallback_name = None
            fallback_duration = 0
            for candidate in rec_names:
                hea_url = f"https://physionet.org/files/{PHYSIONET_DB}/{record_path}{candidate}.hea"
                req = urllib.request.Request(hea_url)
                req.add_header("Authorization", _auth_header(args.username, args.password))
                try:
                    with urllib.request.urlopen(req) as resp:
                        hea_text = resp.read().decode()
                    first_line = hea_text.strip().split("\n")[0].split()
                    n_samples = int(first_line[3]) if len(first_line) > 3 else 0
                    fs = float(first_line[2]) if len(first_line) > 2 else 0
                    # Skip numerics records (fs=1) — these are vital sign trends, not ECG
                    if fs <= 1:
                        continue
                    duration_sec = n_samples / fs if fs > 0 else 0
                    # Normalize line endings before channel check
                    hea_norm = hea_text.replace("\r\n", "\n").upper()
                    has_ecg = any(ch in hea_norm for ch in [" II\n", " II ", " I\n", " I ", " ECG", " MCL"])
                    if not has_ecg or duration_sec < 30:
                        continue
                    # Prefer records within the target range; among those, pick the longest
                    if duration_sec <= TARGET_MAX_SEC:
                        if duration_sec > best_duration:
                            rec_name = candidate
                            best_duration = duration_sec
                    else:
                        # Outside target range — keep as fallback (shortest over-limit)
                        if fallback_name is None or duration_sec < fallback_duration:
                            fallback_name = candidate
                            fallback_duration = duration_sec
                except Exception:
                    continue
            # Do NOT use the over-limit fallback — skip patients whose shortest record
            # still exceeds TARGET_MAX_SEC.  3x oversampling gives enough candidates.

            if rec_name is None:
                logger.warning("  No waveform record >=30s with ECG for patient %d, skipping", sid)
                continue
            logger.info("  Best record %s: %.0fs", rec_name, best_duration)
            # Create unique filename: p{subject_id}_{record_name}
            unique_name = f"p{sid:06d}_{rec_name}"
            success = _download_wfdb_record(rec_name, record_path, out_dir,
                                            args.username, args.password)
            if success:
                # Rename to unique name to avoid collisions
                for ext in (".hea", ".dat"):
                    src = out_dir / f"{rec_name}{ext}"
                    dst = out_dir / f"{unique_name}{ext}"
                    if src.exists():
                        src.rename(dst)
                # Fix ALL references in .hea to point to renamed .dat
                hea_path = out_dir / f"{unique_name}.hea"
                if hea_path.exists():
                    with open(hea_path, "r") as f:
                        content = f.read()
                    content = content.replace(rec_name, unique_name)
                    with open(hea_path, "w") as f:
                        f.write(content)

                # Write label file so our parser knows AF vs non-AF
                label_val = 1 if label == "af" else 0
                with open(out_dir / f"{unique_name}.label", "w") as f:
                    f.write(str(label_val))

                downloaded[label] += 1
                logger.info("  Downloaded %s (label=%d)", unique_name, label_val)
            else:
                logger.warning("  Failed to download record for patient %d", sid)

    logger.info("Done. Downloaded %d AF + %d control waveforms to %s",
                downloaded["af"], downloaded["control"], out_dir)


if __name__ == "__main__":
    main()
