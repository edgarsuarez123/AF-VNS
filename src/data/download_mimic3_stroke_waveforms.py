"""
Download MIMIC-III Waveform Matched Subset filtered for stroke patients.
Selects stroke (ICD-9 430-438) and non-stroke control patients, downloads
ECG waveform segments via authenticated PhysioNet HTTP.
Saves as WFDB records in data/raw/stroke avns/mimic3_stroke/.

Label semantics: 1 = stroke, 0 = non-stroke control.
(Different from the AF pipeline where 1 = AF.)

Usage:
  $env:PHYSIONET_USER = "your_username"
  $env:PHYSIONET_PASS = "your_password"
  .venv\\Scripts\\python -m src.data.download_mimic3_stroke_waveforms
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

STROKE_ICD9_RANGE = [str(c) for c in range(430, 439)]


def _auth_header(username: str, password: str) -> str:
    return "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode("ascii")


def _fetch_records(username: str, password: str):
    req = urllib.request.Request(RECORDS_URL)
    req.add_header("Authorization", _auth_header(username, password))
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode().strip().split("\n")


def _get_subject_record_map(records):
    out = {}
    for r in records:
        match = re.search(r"p(\d+)/$", r.strip())
        if match:
            sid = int(match.group(1))
            out[sid] = r.strip()
    return out


def _list_wfdb_records_in_dir(record_path: str, username: str, password: str):
    url = f"https://physionet.org/files/{PHYSIONET_DB}/{record_path}"
    req = urllib.request.Request(url)
    req.add_header("Authorization", _auth_header(username, password))
    try:
        with urllib.request.urlopen(req) as resp:
            html = resp.read().decode()
        hea_files = re.findall(r'href="([^"]+\.hea)"', html)
        return [h.replace(".hea", "") for h in hea_files]
    except Exception:
        return []


def _download_wfdb_record(record_name: str, record_path: str, out_dir: Path,
                          username: str, password: str) -> bool:
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


def _find_best_ecg_record(rec_names, record_path, username, password):
    """Pick the best ECG record: >=30s, real waveform (fs>1), within 900s cap."""
    TARGET_MAX_SEC = 900
    best_name = None
    best_duration = 0
    for candidate in rec_names:
        hea_url = f"https://physionet.org/files/{PHYSIONET_DB}/{record_path}{candidate}.hea"
        req = urllib.request.Request(hea_url)
        req.add_header("Authorization", _auth_header(username, password))
        try:
            with urllib.request.urlopen(req) as resp:
                hea_text = resp.read().decode()
            first_line = hea_text.strip().split("\n")[0].split()
            n_samples = int(first_line[3]) if len(first_line) > 3 else 0
            fs = float(first_line[2]) if len(first_line) > 2 else 0
            if fs <= 1:
                continue
            duration_sec = n_samples / fs if fs > 0 else 0
            hea_norm = hea_text.replace("\r\n", "\n").upper()
            has_ecg = any(ch in hea_norm for ch in [" II\n", " II ", " I\n", " I ", " ECG", " MCL"])
            if not has_ecg or duration_sec < 30:
                continue
            if duration_sec <= TARGET_MAX_SEC and duration_sec > best_duration:
                best_name = candidate
                best_duration = duration_sec
        except Exception:
            continue
    return best_name, best_duration


def main():
    parser = argparse.ArgumentParser(description="Download MIMIC-III waveforms for stroke patients.")
    parser.add_argument("--username", default=os.environ.get("PHYSIONET_USER"))
    parser.add_argument("--password", default=os.environ.get("PHYSIONET_PASS"))
    parser.add_argument("--n-stroke", type=int, default=150, help="Number of stroke patients")
    parser.add_argument("--n-control", type=int, default=150, help="Number of non-stroke controls")
    parser.add_argument("--diagnoses-csv", default="data/raw/AF avns/mimic3/DIAGNOSES_ICD.csv")
    parser.add_argument("--out-dir", default="data/raw/stroke avns/mimic3_stroke")
    parser.add_argument("--seed", type=int, default=43, help="Different seed from AF downloader to avoid overlap bias")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.username or not args.password:
        logger.error("PhysioNet credentials required. Set PHYSIONET_USER/PHYSIONET_PASS or use --username/--password.")
        sys.exit(1)

    logger.info("Loading diagnoses from %s", args.diagnoses_csv)
    df = pd.read_csv(args.diagnoses_csv)

    stroke_subjects = set(
        df[df.ICD9_CODE.astype(str).str[:3].isin(STROKE_ICD9_RANGE)].SUBJECT_ID.unique()
    )
    all_subjects = set(df.SUBJECT_ID.unique())
    non_stroke_subjects = all_subjects - stroke_subjects
    logger.info("Stroke patients (ICD9 430-438): %d, Non-stroke: %d",
                len(stroke_subjects), len(non_stroke_subjects))

    logger.info("Fetching RECORDS list from PhysioNet...")
    records = _fetch_records(args.username, args.password)
    subject_map = _get_subject_record_map(records)
    wave_subjects = set(subject_map.keys())

    stroke_with_wave = sorted(stroke_subjects & wave_subjects)
    non_stroke_with_wave = sorted(non_stroke_subjects & wave_subjects)
    logger.info("Stroke with waveforms: %d, Non-stroke with waveforms: %d",
                len(stroke_with_wave), len(non_stroke_with_wave))

    rng = np.random.default_rng(args.seed)
    oversample = 3
    stroke_sample = rng.choice(stroke_with_wave,
                               size=min(args.n_stroke * oversample, len(stroke_with_wave)),
                               replace=False)
    ctrl_sample = rng.choice(non_stroke_with_wave,
                             size=min(args.n_control * oversample, len(non_stroke_with_wave)),
                             replace=False)

    downloaded = {"stroke": 0, "control": 0}
    targets = {"stroke": args.n_stroke, "control": args.n_control}

    for label, subjects in [("stroke", stroke_sample), ("control", ctrl_sample)]:
        for sid in subjects:
            if downloaded[label] >= targets[label]:
                break
            existing = list(out_dir.glob(f"p{sid:06d}_*.hea"))
            if existing:
                logger.info("Skipping patient %d (%s) — already downloaded", sid, label)
                downloaded[label] += 1
                continue
            record_path = subject_map[sid]
            logger.info("Checking patient %d (%s) at %s", sid, label, record_path)

            rec_names = _list_wfdb_records_in_dir(record_path, args.username, args.password)
            if not rec_names:
                logger.warning("No WFDB records for patient %d, skipping", sid)
                continue

            rec_name, best_duration = _find_best_ecg_record(
                rec_names, record_path, args.username, args.password
            )
            if rec_name is None:
                logger.warning("  No qualifying ECG record for patient %d, skipping", sid)
                continue

            logger.info("  Best record %s: %.0fs", rec_name, best_duration)
            unique_name = f"p{sid:06d}_{rec_name}"
            success = _download_wfdb_record(rec_name, record_path, out_dir,
                                            args.username, args.password)
            if success:
                for ext in (".hea", ".dat"):
                    src = out_dir / f"{rec_name}{ext}"
                    dst = out_dir / f"{unique_name}{ext}"
                    if src.exists():
                        src.rename(dst)
                hea_path = out_dir / f"{unique_name}.hea"
                if hea_path.exists():
                    with open(hea_path, "r") as f:
                        content = f.read()
                    content = content.replace(rec_name, unique_name)
                    with open(hea_path, "w") as f:
                        f.write(content)

                # label=1 means stroke, label=0 means non-stroke control
                label_val = 1 if label == "stroke" else 0
                with open(out_dir / f"{unique_name}.label", "w") as f:
                    f.write(str(label_val))

                downloaded[label] += 1
                logger.info("  Downloaded %s (label=%d)", unique_name, label_val)
            else:
                logger.warning("  Failed to download record for patient %d", sid)

    logger.info("Done. Downloaded %d stroke + %d control waveforms to %s",
                downloaded["stroke"], downloaded["control"], out_dir)


if __name__ == "__main__":
    main()
