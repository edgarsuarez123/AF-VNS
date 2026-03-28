"""
Download and extract the WESAD dataset (Wearable Stress and Affect Detection).

WESAD (Schmidt et al., 2018) contains multimodal physiological signals from
15 subjects: wrist BVP/PPG (64 Hz), wrist EDA (4 Hz), chest respiration (700 Hz),
chest EDA (700 Hz), labeled stress/baseline/amusement states.

Usage:
    # Download from a URL (pass the official dataset URL as argument):
    python dl_wesad.py --url <WESAD_ZIP_URL> --out data/raw/tinnitus avns/wesad

    # Extract a pre-downloaded zip:
    python dl_wesad.py --zip path/to/WESAD.zip --out data/raw/tinnitus avns/wesad

    # Verify an already-extracted directory:
    python dl_wesad.py --verify --out data/raw/tinnitus avns/wesad

Dataset reference:
    Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018).
    Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection.
    ICMI 2018. https://doi.org/10.1145/3242969.3242985
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path


EXPECTED_SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
# S12 is excluded — corrupted data per original paper


def verify_wesad_dir(out_dir: Path) -> bool:
    """Check that all expected subject pickle files are present."""
    missing = []
    for s in EXPECTED_SUBJECTS:
        pkl = out_dir / f"S{s}" / f"S{s}.pkl"
        if not pkl.exists():
            missing.append(str(pkl))

    if missing:
        print(f"Missing {len(missing)} subject file(s):")
        for m in missing:
            print(f"  {m}")
        return False

    print(f"All {len(EXPECTED_SUBJECTS)} subject files present in {out_dir}")
    return True


def extract_wesad_zip(zip_path: Path, out_dir: Path) -> None:
    print(f"Extracting {zip_path} → {out_dir} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        total = len(members)
        for i, member in enumerate(members, 1):
            zf.extract(member, out_dir)
            if i % 500 == 0 or i == total:
                print(f"  {i}/{total} files extracted", end="\r")
    print()
    print("Extraction complete.")


def download_wesad(url: str, out_dir: Path) -> Path:
    import requests

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "WESAD.zip"

    print(f"Downloading WESAD from {url} ...")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"  {pct:.1f}% ({downloaded // 1024 // 1024} MB)", end="\r")
    print()
    print(f"Downloaded to {zip_path}")
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Download/extract WESAD dataset")
    parser.add_argument("--url", type=str, default=None, help="Download URL for WESAD.zip")
    parser.add_argument("--zip", type=str, default=None, help="Path to pre-downloaded WESAD.zip")
    parser.add_argument(
        "--out",
        type=str,
        default="data/raw/tinnitus avns/wesad",
        help="Output directory",
    )
    parser.add_argument("--verify", action="store_true", help="Only verify existing directory")
    args = parser.parse_args()

    out_dir = Path(args.out)

    if args.verify:
        ok = verify_wesad_dir(out_dir)
        sys.exit(0 if ok else 1)

    if args.url:
        zip_path = download_wesad(args.url, out_dir)
        extract_wesad_zip(zip_path, out_dir)
    elif args.zip:
        extract_wesad_zip(Path(args.zip), out_dir)
    else:
        print("ERROR: provide --url <download_url> or --zip <path_to_zip>")
        print()
        print("To download WESAD, obtain the dataset URL from:")
        print("  https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection")
        print("  or contact the authors at: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/")
        sys.exit(1)

    ok = verify_wesad_dir(out_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
