"""
Phase detection precompute for stroke AVNS: 2s ECG windows + frame-level
diastole/exhalation labels at 5Hz for train/val/test.

Replaces the superseded stroke-vs-control binary classifier cache.
Core logic is in build_stroke_cache() so tests can inject in-memory records.

Usage:
    .venv/Scripts/python -m src.training.stroke_precompute_cache \\
        --config config_stroke.yaml --dataset training --workers 4
"""

import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from src.data.splitter import create_split, load_split
from src.data.stroke_parsers import (
    StrokeRecordDict,
    parse_cerevasc_dir,
    parse_mimic3_stroke_dir,
    parse_sharee_dir,
)
from src.features.edr import generate_exhalation_labels
from src.features.phase_labels import generate_phase_labels

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = "config_stroke.yaml"

# SHAREE: 17 event-patient record IDs from PhysioNet shareedb/info.txt.
# Kept for backward compat (stroke_evaluate.py imports this constant).
SHAREE_EVENT_PATIENTS = {
    "02033", "02059", "02108", "02121", "02148", "02184", "02185",
    "02289", "02304", "02373", "02412",  # MI
    "02119", "02294", "02348",           # stroke
    "02218", "02339", "02403",           # syncope
}


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------

def process_record(
    record: StrokeRecordDict,
    config_path: str,
    window_sec: float = 2.0,
    stride_sec: float = 0.2,
    min_valid_frames: int = 8,
    frame_rate_hz: float = 5.0,
) -> Optional[dict]:
    """Generate full-signal labels, then slice into 2s windows.

    Returns dict with per-window arrays, or None if both label generators fail.
    """
    signal = np.asarray(record["signal"], dtype=np.float64).ravel()
    fs = float(record["fs"])
    sid = record["subject_id"]

    window_samples = int(fs * window_sec)
    stride_samples = int(fs * stride_sec)
    frame_size = int(fs / frame_rate_hz)  # samples per frame (50 @ 250Hz/5Hz)
    frames_per_window = int(window_sec * frame_rate_hz)  # 10

    if len(signal) < window_samples:
        return None

    # Generate full-signal labels
    try:
        dia_result = generate_phase_labels(
            signal, fs, frame_rate_hz=frame_rate_hz, config_path=config_path,
        )
        dia_ok = dia_result["n_beats"] > 0
    except Exception as e:
        logger.warning("%s: diastole labeling failed: %s", sid, e)
        dia_ok = False
        n_frames = len(signal) // frame_size
        dia_result = {
            "labels": np.full(n_frames, np.nan, dtype=np.float32),
            "quality": np.zeros(n_frames, dtype=np.float32),
        }

    try:
        exh_result = generate_exhalation_labels(
            signal, fs, frame_rate_hz=frame_rate_hz, config_path=config_path,
        )
        exh_ok = exh_result["n_resp_cycles"] > 0
    except Exception as e:
        logger.warning("%s: exhalation labeling failed: %s", sid, e)
        exh_ok = False
        n_frames = len(signal) // frame_size
        exh_result = {
            "labels": np.full(n_frames, np.nan, dtype=np.float32),
            "quality": np.zeros(n_frames, dtype=np.float32),
        }

    if not dia_ok and not exh_ok:
        return None

    dia_labels = dia_result["labels"]
    dia_quality = dia_result["quality"]
    exh_labels = exh_result["labels"]
    exh_quality = exh_result["quality"]

    # Slide windows
    ecg_list = []
    diastole_list = []
    exhalation_list = []
    quality_list = []
    n_skipped = 0

    max_start = len(signal) - window_samples + 1
    for s in range(0, max_start, stride_samples):
        frame_start = s // frame_size
        frame_end = frame_start + frames_per_window

        if frame_end > len(dia_labels) or frame_end > len(exh_labels):
            break

        dia_win = dia_labels[frame_start:frame_end]
        exh_win = exh_labels[frame_start:frame_end]

        # Count valid (non-NaN) frames — at least one task must have enough
        dia_valid = int(np.sum(~np.isnan(dia_win)))
        exh_valid = int(np.sum(~np.isnan(exh_win)))

        if dia_valid < min_valid_frames and exh_valid < min_valid_frames:
            n_skipped += 1
            continue

        ecg_win = signal[s : s + window_samples].astype(np.float32)
        dia_q = dia_quality[frame_start:frame_end]
        exh_q = exh_quality[frame_start:frame_end]
        q_win = np.fmin(dia_q, exh_q)

        ecg_list.append(ecg_win)
        diastole_list.append(dia_win)
        exhalation_list.append(exh_win)
        quality_list.append(q_win)

    if not ecg_list:
        return None

    return {
        "subject_id": sid,
        "ecg": ecg_list,
        "diastole": diastole_list,
        "exhalation": exhalation_list,
        "quality": quality_list,
        "n_windows": len(ecg_list),
        "n_skipped": n_skipped,
        "diastole_ok": dia_ok,
        "exhalation_ok": exh_ok,
    }


def _process_record_wrapper(args):
    """Unpacks args for ProcessPoolExecutor (must be picklable top-level fn)."""
    record, config_path, window_sec, stride_sec, min_valid_frames, frame_rate_hz = args
    return process_record(
        record, config_path, window_sec, stride_sec, min_valid_frames, frame_rate_hz,
    )


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------

def build_stroke_cache(
    records: List[StrokeRecordDict],
    config_path: str,
    split_path: str,
    cache_dir,
    workers: int = 1,
    window_sec: float = 2.0,
    stride_sec: float = 0.2,
    min_valid_frames: int = 8,
    frame_rate_hz: float = 5.0,
) -> dict:
    """Build phase detection cache from pre-loaded records.

    Args:
        records:          Pre-loaded StrokeRecordDicts.
        config_path:      Path to config_stroke.yaml.
        split_path:       Path to split JSON; created (70/15/15) if missing.
        cache_dir:        Directory for .npy output.
        workers:          Parallel workers (1 = sequential).
        window_sec:       ECG window duration (seconds).
        stride_sec:       Stride between windows (seconds).
        min_valid_frames: Min non-NaN frames per window to keep it.
        frame_rate_hz:    Label frame rate (Hz).

    Returns:
        dict with n_train, n_val, n_test, n_records, n_skipped, total_windows.
    """
    if workers == 0:
        workers = os.cpu_count() or 4

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    window_samples = int(250.0 * window_sec)  # 500
    frames_per_window = int(window_sec * frame_rate_hz)  # 10

    # --- Load or create split ---
    if not Path(split_path).exists():
        logger.info("Split not found at %s; creating 70/15/15 subject split...", split_path)
        create_split(records, split_path)
    train_ids, val_ids, test_ids = load_split(split_path)

    split_lookup: Dict[str, str] = {}
    for sid in train_ids:
        split_lookup[sid] = "train"
    for sid in val_ids:
        split_lookup[sid] = "val"
    for sid in test_ids:
        split_lookup[sid] = "test"
    logger.info("Split: %d train, %d val, %d test subjects",
                len(train_ids), len(val_ids), len(test_ids))

    # --- Accumulators ---
    accum = {
        name: {"ecg": [], "diastole": [], "exhalation": [], "quality": []}
        for name in ("train", "val", "test")
    }
    n_records = 0
    n_skipped_records = 0
    total_windows = 0
    total_skipped_windows = 0

    # --- Process records ---
    if workers > 1:
        args_list = [
            (rec, config_path, window_sec, stride_sec, min_valid_frames, frame_rate_hz)
            for rec in records
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_record_wrapper, args): args[0]["subject_id"]
                for args in args_list
            }
            pbar = tqdm(as_completed(futures), total=len(futures),
                        desc="Caching phase labels", unit="rec")
            for fut in pbar:
                sid = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    logger.warning("Record %s failed: %s", sid, e)
                    n_skipped_records += 1
                    continue

                if result is None:
                    n_skipped_records += 1
                    continue

                split_name = split_lookup.get(result["subject_id"])
                if split_name is None:
                    n_skipped_records += 1
                    continue

                accum[split_name]["ecg"].extend(result["ecg"])
                accum[split_name]["diastole"].extend(result["diastole"])
                accum[split_name]["exhalation"].extend(result["exhalation"])
                accum[split_name]["quality"].extend(result["quality"])
                total_windows += result["n_windows"]
                total_skipped_windows += result["n_skipped"]
                n_records += 1
                pbar.set_postfix(recs=n_records, wins=total_windows)
            pbar.close()
    else:
        pbar = tqdm(records, desc="Caching phase labels", unit="rec")
        for rec in pbar:
            sid = rec["subject_id"]
            split_name = split_lookup.get(sid)
            if split_name is None:
                n_skipped_records += 1
                continue

            result = process_record(
                rec, config_path, window_sec, stride_sec,
                min_valid_frames, frame_rate_hz,
            )
            if result is None:
                n_skipped_records += 1
                continue

            accum[split_name]["ecg"].extend(result["ecg"])
            accum[split_name]["diastole"].extend(result["diastole"])
            accum[split_name]["exhalation"].extend(result["exhalation"])
            accum[split_name]["quality"].extend(result["quality"])
            total_windows += result["n_windows"]
            total_skipped_windows += result["n_skipped"]
            n_records += 1
            pbar.set_postfix(recs=n_records, wins=total_windows, split=split_name)
        pbar.close()

    logger.info(
        "Processed %d records → %d windows (skipped %d records, %d windows)",
        n_records, total_windows, n_skipped_records, total_skipped_windows,
    )

    # --- Save arrays per split ---
    split_lengths = {}

    for name in ("train", "val", "test"):
        ecg_list = accum[name]["ecg"]
        n = len(ecg_list)
        split_lengths[name] = n

        if n == 0:
            np.save(cache_dir / f"{name}_ecg.npy",
                    np.zeros((0, window_samples), dtype=np.float32))
            np.save(cache_dir / f"{name}_diastole.npy",
                    np.zeros((0, frames_per_window), dtype=np.float32))
            np.save(cache_dir / f"{name}_exhalation.npy",
                    np.zeros((0, frames_per_window), dtype=np.float32))
            np.save(cache_dir / f"{name}_quality.npy",
                    np.zeros((0, frames_per_window), dtype=np.float32))
            logger.warning("Empty %s set; wrote empty arrays.", name)
            continue

        np.save(cache_dir / f"{name}_ecg.npy",
                np.stack(ecg_list, axis=0))
        np.save(cache_dir / f"{name}_diastole.npy",
                np.stack(accum[name]["diastole"], axis=0))
        np.save(cache_dir / f"{name}_exhalation.npy",
                np.stack(accum[name]["exhalation"], axis=0))
        np.save(cache_dir / f"{name}_quality.npy",
                np.stack(accum[name]["quality"], axis=0))
        logger.info("Saved %s: %d windows", name, n)

    del accum

    meta = {
        "window_samples": window_samples,
        "frames_per_window": frames_per_window,
        "window_sec": window_sec,
        "stride_sec": stride_sec,
        "frame_rate_hz": frame_rate_hz,
        "n_train": split_lengths.get("train", 0),
        "n_val": split_lengths.get("val", 0),
        "n_test": split_lengths.get("test", 0),
        "n_records_processed": n_records,
        "n_records_skipped": n_skipped_records,
        "total_windows": total_windows,
    }
    with open(cache_dir / "phase_cache_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Cache written to %s: %s", cache_dir, meta)
    return meta


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> dict:
    import yaml
    if not os.path.isabs(config_path) and not os.path.isfile(config_path):
        root = Path(__file__).resolve().parents[2]
        config_path = root / config_path
    with open(config_path) as f:
        return yaml.safe_load(f)


def main(
    config_path: str = CONFIG_PATH,
    workers: int = 1,
    dataset: str = "training",
):
    """CLI entry: build phase detection cache.

    dataset="training" → CVES + MIMIC-3, 70/15/15 split
    dataset="ood"      → SHaRe only, all test (no split)
    """
    if not os.path.isabs(config_path):
        config_path = str(Path(config_path).resolve())

    config = _load_config(config_path)
    data_cfg = config.get("data", {})
    precompute_cfg = config.get("phase_precompute", {})

    window_sec = float(precompute_cfg.get("window_sec", 2.0))
    stride_sec = float(precompute_cfg.get("stride_sec", 0.2))
    min_valid = int(precompute_cfg.get("min_valid_frames", 8))
    frame_rate_hz = float(precompute_cfg.get("frame_rate_hz", 5.0))

    records: List[StrokeRecordDict] = []

    if dataset == "training":
        cache_dir = Path(precompute_cfg.get("cache_dir", "models/artifacts/cache_phase_detect"))
        split_path = precompute_cfg.get("split_path", "models/artifacts/phase_detect_split.json")

        # Load CVES
        cves_dir = data_cfg.get("cves_subdir", "data/raw/stroke avns/cves")
        cves_records = parse_cerevasc_dir(cves_dir, config_path)
        logger.info("Loaded %d CVES records", len(cves_records))
        records.extend(cves_records)

        # Load MIMIC-3 stroke
        mimic_dir = data_cfg.get("mimic3_stroke_subdir", "data/raw/stroke avns/mimic3_stroke")
        mimic_records = parse_mimic3_stroke_dir(mimic_dir, config_path)
        logger.info("Loaded %d MIMIC-3 stroke records", len(mimic_records))
        records.extend(mimic_records)

    elif dataset == "ood":
        cache_dir = Path(precompute_cfg.get(
            "ood_cache_dir", "models/artifacts/cache_phase_detect_ood"))
        split_path = str(cache_dir / "ood_split.json")

        sharee_dir = data_cfg.get("sharee_subdir", "data/raw/stroke avns/shareedb")
        label_map = {pid: 1 for pid in SHAREE_EVENT_PATIENTS}
        sharee_records = parse_sharee_dir(sharee_dir, label_map, config_path)
        logger.info("Loaded %d SHaRe records", len(sharee_records))
        records.extend(sharee_records)
    elif dataset == "cves":
        cache_dir = Path(precompute_cfg.get("cache_dir", "models/artifacts/cache_phase_detect")).parent / "cache_phase_detect_cves"
        split_path = str(cache_dir / "phase_detect_cves_split.json")

        cves_dir = data_cfg.get("cves_subdir", "data/raw/stroke avns/cves")
        cves_records = parse_cerevasc_dir(cves_dir, config_path)
        logger.info("Loaded %d CVES records", len(cves_records))
        records.extend(cves_records)

    elif dataset == "mimic":
        cache_dir = Path(precompute_cfg.get("cache_dir", "models/artifacts/cache_phase_detect")).parent / "cache_phase_detect_mimic"
        split_path = str(cache_dir / "phase_detect_mimic_split.json")

        mimic_dir = data_cfg.get("mimic3_stroke_subdir", "data/raw/stroke avns/mimic3_stroke")
        mimic_records = parse_mimic3_stroke_dir(mimic_dir, config_path)
        logger.info("Loaded %d MIMIC-3 stroke records", len(mimic_records))
        records.extend(mimic_records)

    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Use 'training', 'ood', 'cves', or 'mimic'.")

    if not records:
        logger.error("No records loaded for dataset=%s. Check data paths.", dataset)
        return

    logger.info("Total records for dataset=%s: %d", dataset, len(records))

    build_stroke_cache(
        records=records,
        config_path=config_path,
        split_path=split_path,
        cache_dir=cache_dir,
        workers=workers,
        window_sec=window_sec,
        stride_sec=stride_sec,
        min_valid_frames=min_valid,
        frame_rate_hz=frame_rate_hz,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Precompute 2s ECG windows + phase labels for stroke AVNS."
    )
    p.add_argument("--config", default=CONFIG_PATH, help="Path to config_stroke.yaml")
    p.add_argument("--workers", type=int, default=1,
                   help="Worker processes (0 = all cores). Default 1 = single-threaded.")
    p.add_argument("--dataset", choices=["training", "ood", "cves", "mimic"], default="training",
                   help="training=CVES+MIMIC-3, ood=SHaRe, cves=CVES only, mimic=MIMIC-3 only")
    args = p.parse_args()
    main(config_path=args.config, workers=args.workers, dataset=args.dataset)
