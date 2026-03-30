"""
Tinnitus AVNS phase detection precompute: 2s PPG windows + frame-level
diastole/exhalation labels at 5 Hz for train/val/test.

Sources: BIDMC (PPG @ 125 Hz + impedance resp) + WESAD (BVP @ 64 Hz + chest resp).
WESAD BVP is resampled to 125 Hz so all windows are exactly 250 samples.

Usage:
    .venv/Scripts/python -m src.training.tinnitus_precompute_cache \\
        --config config_tinnitus.yaml --workers 4
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
from src.data.tinnitus_parsers import (
    TinnitusRecordDict,
    parse_bidmc_ppg_dir,
    parse_wesad_dir,
)
from src.features.ppg_phase_labels import generate_ppg_phase_labels
from src.features.ppg_resp import generate_exhalation_labels_from_ppg
from src.features.resp_labels import generate_exhalation_labels_from_reference

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = "config_tinnitus.yaml"
TARGET_FS = 125.0  # PhaseDetector input_samples=250 = 2s @ 125 Hz


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------

def process_record(
    record: TinnitusRecordDict,
    config_path: str,
    window_sec: float = 2.0,
    stride_sec: float = 0.2,
    min_valid_frames: int = 8,
    frame_rate_hz: float = 5.0,
    target_fs: float = TARGET_FS,
) -> Optional[dict]:
    """Generate full-signal phase labels then slice into 2s PPG windows.

    WESAD records at 64 Hz are resampled to target_fs (125 Hz) so all windows
    are exactly int(target_fs * window_sec) = 250 samples.

    Returns dict with per-window arrays, or None if both label generators fail.
    """
    from scipy.signal import resample as sp_resample

    ppg = np.asarray(record["ppg_signal"], dtype=np.float64).ravel()
    fs = float(record["ppg_fs"])
    sid = record["subject_id"]
    resp_signal = record.get("resp_signal")
    resp_fs = float(record.get("resp_fs") or fs)
    resp_channel = record.get("resp_channel")

    # Resample PPG to target_fs if needed (e.g. WESAD BVP at 64 Hz → 125 Hz)
    if abs(fs - target_fs) > 0.5:
        n_target = int(round(len(ppg) * target_fs / fs))
        ppg = sp_resample(ppg, n_target).astype(np.float64)
        fs = target_fs

    window_samples = int(round(target_fs * window_sec))   # 250
    stride_samples = int(round(target_fs * stride_sec))   # 25
    frame_size = int(round(target_fs / frame_rate_hz))    # 25 (125/5)
    frames_per_window = int(window_sec * frame_rate_hz)   # 10

    if len(ppg) < window_samples:
        return None

    # --- Diastole labels via PPG dicrotic-notch detection ---
    try:
        dia_result = generate_ppg_phase_labels(
            ppg, fs, frame_rate_hz=frame_rate_hz, config_path=config_path,
        )
        dia_ok = dia_result["n_beats"] > 0
    except Exception as e:
        logger.warning("%s: PPG diastole labeling failed: %s", sid, e)
        dia_ok = False
        n_frames = len(ppg) // frame_size
        dia_result = {
            "labels": np.full(n_frames, np.nan, dtype=np.float32),
            "quality": np.zeros(n_frames, dtype=np.float32),
        }

    # --- Exhalation labels: reference resp → PPG-derived fallback ---
    exh_ok = False
    exh_method = "none"

    if resp_signal is not None:
        try:
            exh_result = generate_exhalation_labels_from_reference(
                resp_signal, resp_fs,
                channel_name=resp_channel,
                frame_rate_hz=frame_rate_hz,
                config_path=config_path,
            )
            exh_ok = exh_result["n_resp_cycles"] > 0
            exh_method = "reference" if exh_ok else "reference_failed"
        except Exception as e:
            logger.warning("%s: reference resp labeling failed: %s", sid, e)
            exh_method = "reference_failed"

    if not exh_ok:
        try:
            exh_result = generate_exhalation_labels_from_ppg(
                ppg, fs, frame_rate_hz=frame_rate_hz, config_path=config_path,
            )
            exh_ok = exh_result["n_resp_cycles"] > 0
            exh_method = "ppg_derived" if exh_ok else "ppg_derived_failed"
        except Exception as e:
            logger.warning("%s: PPG-derived exhalation labeling failed: %s", sid, e)

    if not exh_ok:
        n_frames = len(ppg) // frame_size
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

    # --- Sliding window extraction ---
    ppg_list = []
    diastole_list = []
    exhalation_list = []
    quality_list = []
    n_skipped = 0

    max_start = len(ppg) - window_samples + 1
    for s in range(0, max_start, stride_samples):
        frame_start = s // frame_size
        frame_end = frame_start + frames_per_window

        if frame_end > len(dia_labels) or frame_end > len(exh_labels):
            break

        dia_win = dia_labels[frame_start:frame_end]
        exh_win = exh_labels[frame_start:frame_end]

        dia_valid = int(np.sum(~np.isnan(dia_win)))
        exh_valid = int(np.sum(~np.isnan(exh_win)))

        if dia_valid < min_valid_frames and exh_valid < min_valid_frames:
            n_skipped += 1
            continue

        ppg_win = ppg[s: s + window_samples].astype(np.float32)
        q_win = np.fmin(
            dia_quality[frame_start:frame_end],
            exh_quality[frame_start:frame_end],
        )

        ppg_list.append(ppg_win)
        diastole_list.append(dia_win)
        exhalation_list.append(exh_win)
        quality_list.append(q_win)

    if not ppg_list:
        return None

    return {
        "subject_id": sid,
        "ppg": ppg_list,
        "diastole": diastole_list,
        "exhalation": exhalation_list,
        "quality": quality_list,
        "n_windows": len(ppg_list),
        "n_skipped": n_skipped,
        "diastole_ok": dia_ok,
        "exhalation_ok": exh_ok,
        "exh_method": exh_method,
    }


def _process_record_wrapper(args):
    """Top-level picklable wrapper for ProcessPoolExecutor."""
    record, config_path, window_sec, stride_sec, min_valid_frames, frame_rate_hz, target_fs = args
    return process_record(
        record, config_path, window_sec, stride_sec, min_valid_frames, frame_rate_hz, target_fs,
    )


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------

def build_tinnitus_cache(
    records: List[TinnitusRecordDict],
    config_path: str,
    split_path: str,
    cache_dir,
    workers: int = 1,
    window_sec: float = 2.0,
    stride_sec: float = 0.2,
    min_valid_frames: int = 8,
    frame_rate_hz: float = 5.0,
    target_fs: float = TARGET_FS,
) -> dict:
    """Build tinnitus phase detection cache from pre-loaded records.

    Args:
        records:          Pre-loaded TinnitusRecordDicts.
        config_path:      Path to config_tinnitus.yaml.
        split_path:       Path to split JSON; created (70/15/15) if missing.
        cache_dir:        Directory for .npy output.
        workers:          Parallel workers (1 = sequential).
        window_sec:       PPG window duration (seconds).
        stride_sec:       Stride between windows (seconds).
        min_valid_frames: Min non-NaN frames per window to keep.
        frame_rate_hz:    Label frame rate (Hz).
        target_fs:        Target PPG sample rate; records resampled if needed.

    Returns:
        dict with n_train, n_val, n_test, n_records, total_windows, etc.
    """
    if workers == 0:
        workers = os.cpu_count() or 4

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    window_samples = int(round(target_fs * window_sec))  # 250
    frames_per_window = int(window_sec * frame_rate_hz)  # 10

    # --- Load or create subject split ---
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
        name: {"ppg": [], "diastole": [], "exhalation": [], "quality": []}
        for name in ("train", "val", "test")
    }
    n_records = 0
    n_skipped_records = 0
    total_windows = 0
    total_skipped_windows = 0
    exh_method_counts: Dict[str, int] = {}

    # --- Process records ---
    if workers > 1:
        args_list = [
            (rec, config_path, window_sec, stride_sec, min_valid_frames, frame_rate_hz, target_fs)
            for rec in records
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_record_wrapper, args): args[0]["subject_id"]
                for args in args_list
            }
            pbar = tqdm(as_completed(futures), total=len(futures),
                        desc="Caching tinnitus phase labels", unit="rec")
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

                accum[split_name]["ppg"].extend(result["ppg"])
                accum[split_name]["diastole"].extend(result["diastole"])
                accum[split_name]["exhalation"].extend(result["exhalation"])
                accum[split_name]["quality"].extend(result["quality"])
                total_windows += result["n_windows"]
                total_skipped_windows += result["n_skipped"]
                m = result.get("exh_method", "none")
                exh_method_counts[m] = exh_method_counts.get(m, 0) + 1
                n_records += 1
                pbar.set_postfix(recs=n_records, wins=total_windows)
            pbar.close()
    else:
        pbar = tqdm(records, desc="Caching tinnitus phase labels", unit="rec")
        for rec in pbar:
            sid = rec["subject_id"]
            split_name = split_lookup.get(sid)
            if split_name is None:
                n_skipped_records += 1
                continue

            result = process_record(
                rec, config_path, window_sec, stride_sec,
                min_valid_frames, frame_rate_hz, target_fs,
            )
            if result is None:
                n_skipped_records += 1
                continue

            accum[split_name]["ppg"].extend(result["ppg"])
            accum[split_name]["diastole"].extend(result["diastole"])
            accum[split_name]["exhalation"].extend(result["exhalation"])
            accum[split_name]["quality"].extend(result["quality"])
            total_windows += result["n_windows"]
            total_skipped_windows += result["n_skipped"]
            m = result.get("exh_method", "none")
            exh_method_counts[m] = exh_method_counts.get(m, 0) + 1
            n_records += 1
            pbar.set_postfix(recs=n_records, wins=total_windows, split=split_name)
        pbar.close()

    logger.info(
        "Processed %d records → %d windows (skipped %d records, %d windows)",
        n_records, total_windows, n_skipped_records, total_skipped_windows,
    )

    # --- Save per-split arrays ---
    split_lengths: Dict[str, int] = {}

    for name in ("train", "val", "test"):
        ppg_list = accum[name]["ppg"]
        n = len(ppg_list)
        split_lengths[name] = n

        if n == 0:
            np.save(cache_dir / f"{name}_ppg.npy",
                    np.zeros((0, window_samples), dtype=np.float32))
            np.save(cache_dir / f"{name}_diastole.npy",
                    np.zeros((0, frames_per_window), dtype=np.float32))
            np.save(cache_dir / f"{name}_exhalation.npy",
                    np.zeros((0, frames_per_window), dtype=np.float32))
            np.save(cache_dir / f"{name}_quality.npy",
                    np.zeros((0, frames_per_window), dtype=np.float32))
            logger.warning("Empty %s set; wrote empty arrays.", name)
            continue

        np.save(cache_dir / f"{name}_ppg.npy",
                np.stack(ppg_list, axis=0))
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
        "target_fs": target_fs,
        "n_train": split_lengths.get("train", 0),
        "n_val": split_lengths.get("val", 0),
        "n_test": split_lengths.get("test", 0),
        "n_records_processed": n_records,
        "n_records_skipped": n_skipped_records,
        "total_windows": total_windows,
        "exh_method_counts": exh_method_counts,
    }
    with open(cache_dir / "tinnitus_cache_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Cache written to %s", cache_dir)
    logger.info("Meta: %s", meta)
    return meta


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> dict:
    import yaml
    if not os.path.isabs(config_path) and not os.path.isfile(config_path):
        root = Path(__file__).resolve().parents[2]
        config_path = str(root / config_path)
    with open(config_path) as f:
        return yaml.safe_load(f)


def main(
    config_path: str = CONFIG_PATH,
    workers: int = 1,
    dataset: str = "training",
):
    """CLI entry: build tinnitus phase detection cache.

    dataset="training" → BIDMC + WESAD (default)
    dataset="bidmc"    → BIDMC only
    dataset="wesad"    → WESAD only
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

    cache_dir = Path(data_cfg.get("cache_dir", "models/artifacts/cache_tinnitus_phase"))
    split_path = data_cfg.get("split_path", "models/artifacts/tinnitus_phase_split.json")

    records: List[TinnitusRecordDict] = []

    if dataset in ("training", "bidmc"):
        bidmc_dir = data_cfg.get("bidmc_subdir", "data/raw/stroke avns/bidmc")
        bidmc_records = parse_bidmc_ppg_dir(bidmc_dir, config_path)
        logger.info("Loaded %d BIDMC PPG records", len(bidmc_records))
        records.extend(bidmc_records)

    if dataset in ("training", "wesad"):
        wesad_dir = data_cfg.get("wesad_subdir", "data/raw/tinnitus avns/wesad/WESAD")
        wesad_records = parse_wesad_dir(wesad_dir, config_path)
        logger.info("Loaded %d WESAD records", len(wesad_records))
        records.extend(wesad_records)

    if not records:
        logger.error("No records loaded for dataset=%s. Check data paths in config.", dataset)
        return

    logger.info("Total records for dataset=%s: %d", dataset, len(records))

    build_tinnitus_cache(
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
        description="Precompute 2s PPG windows + phase labels for tinnitus AVNS."
    )
    p.add_argument("--config", default=CONFIG_PATH, help="Path to config_tinnitus.yaml")
    p.add_argument("--workers", type=int, default=1,
                   help="Worker processes (0 = all cores). Default 1 = single-threaded.")
    p.add_argument("--dataset", choices=["training", "bidmc", "wesad"], default="training",
                   help="training=BIDMC+WESAD (default), bidmc=BIDMC only, wesad=WESAD only")
    args = p.parse_args()
    main(config_path=args.config, workers=args.workers, dataset=args.dataset)
