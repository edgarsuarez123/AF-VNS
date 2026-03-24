"""
S-63: QRS Amplitude Modulation EDR — Smoke Test vs Reference Respiratory Signal

Compares three EDR methods on 5 CVES records that have hardware resp reference:
  - RSA (vangent2019) — current baseline
  - QRS amplitude modulation (amplitude)
  - PCA fusion (fusion)

Decision gate for S-64 (cache rebuild):
  If QRS-AM correlation with reference <= RSA correlation -> STOP, skip S-64-S-65.
  If QRS-AM shows improvement -> proceed with cache rebuild and retrain.

Usage:
  .venv/Scripts/python scripts/smoke_qrs_am_edr.py [--n-records 5]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _edr_resp_correlation(edr_signal: np.ndarray, resp_signal: np.ndarray, fs: float) -> float:
    """Pearson correlation between EDR and reference resp at 5Hz envelope."""
    from scipy.signal import resample_poly
    from math import gcd

    # Resample both to 5 Hz for comparison
    fs_int = int(round(fs))
    target_hz = 5
    g = gcd(fs_int, target_hz)
    up, down = target_hz // g, fs_int // g

    edr_5hz = resample_poly(edr_signal, up, down)
    resp_5hz = resample_poly(resp_signal, up, down)

    # Trim to same length
    n = min(len(edr_5hz), len(resp_5hz))
    if n < 10:
        return float("nan")
    edr_5hz = edr_5hz[:n]
    resp_5hz = resp_5hz[:n]

    # Normalize
    for sig in (edr_5hz, resp_5hz):
        std = sig.std()
        if std == 0:
            return float("nan")

    corr = np.corrcoef(edr_5hz, resp_5hz)[0, 1]
    return float(corr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-records", type=int, default=5,
                        help="Number of CVES records to test (default: 5)")
    args = parser.parse_args()

    from src.data.stroke_parsers import parse_cerevasc_dir
    from src.features.edr import extract_edr

    config_path = str(ROOT / "config_stroke.yaml")

    logger.info("Loading CVES records...")
    records = parse_cerevasc_dir(str(ROOT / "data/raw/stroke avns/cves"), config_path)

    # Filter to records with reference resp signal
    with_ref = [r for r in records if r.get("resp_signal") is not None]
    logger.info("%d / %d CVES records have reference resp signal", len(with_ref), len(records))

    if len(with_ref) == 0:
        logger.error("No CVES records with reference resp signal found.")
        sys.exit(1)

    # Sample up to n_records
    sample = with_ref[:args.n_records]
    logger.info("Testing on %d records...\n", len(sample))

    results = []
    methods = ["vangent2019", "amplitude", "fusion"]

    for rec in sample:
        ecg = rec["signal"]
        fs = rec["fs"]
        resp = rec["resp_signal"]
        channel = rec.get("resp_channel", "?")

        logger.info("Record: %s | fs=%.0fHz | resp=%s | len=%.0fs",
                    rec.get("record_id", "?"), fs, channel, len(ecg) / fs)

        row = {"record_id": rec.get("record_id", "?")}
        for method in methods:
            try:
                edr_sig, _, _ = extract_edr(ecg, fs, method=method)
                corr = _edr_resp_correlation(edr_sig, resp, fs)
                row[method] = corr
                logger.info("  %-15s corr=%.3f", method, corr if not np.isnan(corr) else float("nan"))
            except Exception as e:
                logger.warning("  %-15s FAILED: %s", method, e)
                row[method] = float("nan")
        results.append(row)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY — Mean |correlation| with reference resp signal")
    print("="*60)
    for method in methods:
        corrs = [abs(r[method]) for r in results if not np.isnan(r.get(method, float("nan")))]
        if corrs:
            mean_corr = np.mean(corrs)
            print(f"  {method:<18} mean |corr| = {mean_corr:.3f}  (n={len(corrs)})")
        else:
            print(f"  {method:<18} FAILED (no valid results)")

    # Decision
    baseline_corrs = [abs(r["vangent2019"]) for r in results
                      if not np.isnan(r.get("vangent2019", float("nan")))]
    am_corrs = [abs(r["amplitude"]) for r in results
                if not np.isnan(r.get("amplitude", float("nan")))]
    fusion_corrs = [abs(r["fusion"]) for r in results
                    if not np.isnan(r.get("fusion", float("nan")))]

    baseline_mean = np.mean(baseline_corrs) if baseline_corrs else 0.0
    am_mean = np.mean(am_corrs) if am_corrs else 0.0
    fusion_mean = np.mean(fusion_corrs) if fusion_corrs else 0.0
    best_new = max(am_mean, fusion_mean)
    best_method = "amplitude" if am_mean >= fusion_mean else "fusion"

    print("\nDECISION:")
    if best_new > baseline_mean:
        delta = best_new - baseline_mean
        print(f"  PROCEED — {best_method} ({best_new:.3f}) > RSA baseline ({baseline_mean:.3f})"
              f"  [+{delta:.3f}]")
        print(f"  -> Rebuild MIMIC cache with method='{best_method}' (S-64)")
    else:
        print(f"  STOP — best new method ({best_method}: {best_new:.3f})"
              f" <= RSA baseline ({baseline_mean:.3f})")
        print("  -> ECG-only exhalation ceiling confirmed. Skip S-64/S-65.")
        print("  -> Document finding in STROKE_PLAN.md (S-66).")

    print("="*60)


if __name__ == "__main__":
    main()
