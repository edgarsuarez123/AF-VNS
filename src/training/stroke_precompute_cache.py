"""
Streaming precompute for stroke AVNS: HRV and denoised 10s cache for train/val/test.

Mirrors precompute_cache.py but uses stroke parsers as the data source.
Core logic is in build_stroke_cache() so tests can inject in-memory records directly.

Usage:
    .venv/Scripts/python -m src.training.stroke_precompute_cache \\
        --config config_stroke.yaml --phase 1 --workers 1
"""

import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional

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
from src.features.pipeline import waveform_10s_denoised, waveform_to_hrv_sequence
from src.features.scaler import fit_scaler, load_scaler, transform

from .build_model import load_config
from .precompute_worker import process_chunk as _process_chunk

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = "config_stroke.yaml"
MAX_SHORT_LEN_CAP = 3600  # 12s at 300 Hz — same cap as AF pipeline
N_FEATURES = 7
CHUNK_SIZE = 8

# SHAREE: 17 event-patient record IDs from PhysioNet shareedb/info.txt (vascular event column).
# 11 myocardial infarction, 3 stroke, 3 syncope — all treated as event=1 for OOD evaluation.
# Source: https://physionet.org/physiobank/database/shareedb/info.txt
SHAREE_EVENT_PATIENTS = {
    "02033", "02059", "02108", "02121", "02148", "02184", "02185",
    "02289", "02304", "02373", "02412",  # MI
    "02119", "02294", "02348",           # stroke
    "02218", "02339", "02403",           # syncope
}


def build_stroke_cache(
    records: List[StrokeRecordDict],
    config_path: str,
    split_path: str,
    cache_dir: Path,
    scaler_path: str,
    workers: int = 1,
    chunk_size: int = CHUNK_SIZE,
) -> dict:
    """Build HRV + denoised-waveform cache from pre-loaded stroke records.

    Accepts an in-memory list of StrokeRecordDicts so tests can inject synthetic
    records without writing WFDB files to disk.

    Args:
        records:      Pre-loaded StrokeRecordDicts (from parse_mimic3_stroke_dir, etc.)
        config_path:  Path to config_stroke.yaml (for HRV pipeline params).
        split_path:   Path to split JSON; created (70/15/15) if missing.
        cache_dir:    Directory to write .npy files and cache_meta.json.
        scaler_path:  Path to write fitted StandardScaler pkl.
        workers:      Parallel worker processes (1 = single-threaded).
        chunk_size:   Sub-chunk size for parallel mode.

    Returns:
        dict with keys: n_train, n_val, n_test, max_short_len
    """
    if workers == 0:
        workers = os.cpu_count() or 4

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    data_cfg = config.get("data", {})
    waveform_sec = float(data_cfg.get("waveform_sec", 10))
    long_sec = float(data_cfg.get("hrv_window_sec", 300))
    stride_sec = float(data_cfg.get("stride_sec", long_sec))

    # --- Load or create split ---
    if not Path(split_path).exists():
        logger.info("Split not found at %s; creating 70/15/15 subject split...", split_path)
        create_split(records, split_path)
    train_ids, val_ids, test_ids = load_split(split_path)

    split_lookup = {}
    for sid in train_ids:
        split_lookup[sid] = "train"
    for sid in val_ids:
        split_lookup[sid] = "val"
    for sid in test_ids:
        split_lookup[sid] = "test"
    logger.info("Split: %d train, %d val, %d test subjects",
                len(train_ids), len(val_ids), len(test_ids))

    # --- Accumulators ---
    accum = {name: {"short": [], "hrv": [], "labels": [], "hrv_lengths": []}
             for name in ("train", "val", "test")}
    max_short_len = 0
    n_records = 0
    n_windows = 0
    n_skipped = 0

    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None

    try:
        pbar = tqdm(records, desc="Caching stroke records", unit="rec")
        for rec in pbar:
            sid = rec["subject_id"]
            split_name = split_lookup.get(sid)
            if split_name is None:
                n_skipped += 1
                continue

            label_val = rec.get("label")
            if label_val is None or int(label_val) < 0:
                n_skipped += 1
                continue

            sig = np.asarray(rec["signal"], dtype=np.float64)
            fs = float(rec["fs"])
            label = float(label_val)

            n_short = int(fs * waveform_sec)
            n_long = int(fs * long_sec)
            stride = int(fs * stride_sec)

            if sig.size < n_short:
                n_skipped += 1
                continue

            # Build window starts — same logic as PhysioDataset and precompute_cache.py
            if sig.size >= n_long:
                starts = list(range(0, sig.size - n_long + 1, stride))
            else:
                starts = [0]  # short record: single window

            if not starts:
                n_skipped += 1
                continue

            chunk_data = []
            for start in starts:
                short_np = sig[start: start + n_short]
                long_np = sig[start: start + n_long] if sig.size >= start + n_long else sig
                chunk_data.append((short_np, long_np, fs, label))

            # Process windows
            if executor is None or len(chunk_data) <= chunk_size:
                for short_np, long_np, fs_val, lbl in chunk_data:
                    hrv = waveform_to_hrv_sequence(long_np, fs_val, config_path=config_path)
                    short_d = np.asarray(
                        waveform_10s_denoised(short_np, fs_val, config_path=config_path),
                        dtype=np.float32,
                    )
                    hrv_len = max(1, int(np.sum(~np.all(np.isnan(hrv), axis=-1))))
                    accum[split_name]["short"].append(short_d)
                    accum[split_name]["hrv"].append(hrv)
                    accum[split_name]["labels"].append(lbl)
                    accum[split_name]["hrv_lengths"].append(hrv_len)
                    if short_d.size > max_short_len:
                        max_short_len = short_d.size
            else:
                futures = []
                for sub_start in range(0, len(chunk_data), chunk_size):
                    sub = chunk_data[sub_start: sub_start + chunk_size]
                    futures.append(executor.submit(_process_chunk, (sub, config_path)))
                for fut in futures:
                    for short_d, hrv, lbl, hrv_len in fut.result():
                        accum[split_name]["short"].append(short_d)
                        accum[split_name]["hrv"].append(hrv)
                        accum[split_name]["labels"].append(lbl)
                        accum[split_name]["hrv_lengths"].append(hrv_len)
                        if short_d.size > max_short_len:
                            max_short_len = short_d.size

            n_windows += len(chunk_data)
            n_records += 1
            pbar.set_postfix(recs=n_records, wins=n_windows, split=split_name)
            del sig, chunk_data, rec

        pbar.close()
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    logger.info("Processed %d records → %d windows (skipped %d)", n_records, n_windows, n_skipped)

    # --- Save arrays per split ---
    max_short_len = min(max_short_len, MAX_SHORT_LEN_CAP)
    if max_short_len == 0:
        raise ValueError("No samples produced; cannot build cache.")

    hrv_steps = int(long_sec / float(config.get("hrv", {}).get("subwindow_sec", 60)))
    split_lengths = {}

    for name in ("train", "val", "test"):
        short_list = accum[name]["short"]
        hrv_list = accum[name]["hrv"]
        labels_list = accum[name]["labels"]
        hrv_lengths_list = accum[name]["hrv_lengths"]
        n = len(labels_list)
        split_lengths[name] = n

        if n == 0:
            np.save(cache_dir / f"{name}_hrv.npy",
                    np.zeros((0, hrv_steps, N_FEATURES), dtype=np.float32))
            np.save(cache_dir / f"{name}_labels.npy", np.zeros(0, dtype=np.float32))
            np.save(cache_dir / f"{name}_short.npy", np.zeros((0, 0), dtype=np.float32))
            np.save(cache_dir / f"{name}_hrv_lengths.npy", np.zeros(0, dtype=np.int32))
            logger.warning("Empty %s set; wrote empty arrays.", name)
            continue

        hrv_arr = np.stack(hrv_list, axis=0).astype(np.float32)
        np.save(cache_dir / f"{name}_hrv.npy", hrv_arr)
        np.save(cache_dir / f"{name}_labels.npy", np.array(labels_list, dtype=np.float32))
        np.save(cache_dir / f"{name}_hrv_lengths.npy", np.array(hrv_lengths_list, dtype=np.int32))

        out = np.zeros((n, max_short_len), dtype=np.float32)
        for i, s in enumerate(short_list):
            L = min(s.size, max_short_len)
            out[i, :L] = s.ravel()[:L]
        np.save(cache_dir / f"{name}_short.npy", out)
        logger.info("Saved %s: %d windows, short=(%d,%d), hrv=%s",
                    name, n, n, max_short_len, hrv_arr.shape)

    del accum

    # --- Fit scaler on train HRV only ---
    train_hrv_path = cache_dir / "train_hrv.npy"
    if not train_hrv_path.exists() or split_lengths.get("train", 0) == 0:
        raise ValueError("Train set empty; cannot fit scaler.")
    train_hrv = np.load(train_hrv_path)
    X = train_hrv.reshape(-1, N_FEATURES)
    fit_scaler(X, path=scaler_path, config_path=config_path)
    logger.info("Fitted scaler on %d HRV rows → %s", X.shape[0], scaler_path)

    # Scale all splits
    scaler = load_scaler(path=scaler_path, config_path=config_path)
    for split_name in ("train", "val", "test"):
        path_hrv = cache_dir / f"{split_name}_hrv.npy"
        if not path_hrv.exists():
            continue
        hrv = np.load(path_hrv).astype(np.float32)
        if hrv.size == 0:
            np.save(cache_dir / f"{split_name}_hrv_scaled.npy", hrv)
            continue
        hrv = transform(hrv, scaler)
        nan_frac = np.isnan(hrv).any(axis=-1).mean()
        logger.info("  %s: %.1f%% HRV rows had NaN before imputation", split_name, nan_frac * 100)
        np.nan_to_num(hrv, nan=0.0, copy=False)
        np.save(cache_dir / f"{split_name}_hrv_scaled.npy", hrv)

    meta = {
        "max_short_len": int(max_short_len),
        "n_train": split_lengths.get("train", 0),
        "n_val": split_lengths.get("val", 0),
        "n_test": split_lengths.get("test", 0),
    }
    with open(cache_dir / "cache_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Cache written to %s: %s", cache_dir, meta)
    return meta


def main(
    config_path: str = CONFIG_PATH,
    workers: int = 1,
    chunk_size: int = CHUNK_SIZE,
    phase: int = 1,
):
    if not os.path.isabs(config_path):
        config_path = str(Path(config_path).resolve())

    config = load_config(config_path)
    paths_cfg = config.get("paths", {})
    data_cfg = config.get("data", {})

    # Phase-aware path selection
    if phase == 1:
        cache_dir = Path(paths_cfg.get("phase1_cache_dir", "models/artifacts/cache_stroke_phase1"))
        scaler_path = paths_cfg.get("phase1_scaler", "models/artifacts/stroke_phase1_scaler.pkl")
        split_path = paths_cfg.get("phase1_split", "models/artifacts/stroke_phase1_split.json")
    elif phase == 2:
        cache_dir = Path(paths_cfg.get("phase2_cache_dir", "models/artifacts/cache_stroke_phase2"))
        scaler_path = paths_cfg.get("phase2_scaler", "models/artifacts/stroke_phase2_scaler.pkl")
        split_path = paths_cfg.get("phase2_split", "models/artifacts/stroke_phase2_split.json")
    else:  # phase == 0: all available datasets
        cache_dir = Path(paths_cfg.get("cache_dir", "models/artifacts/cache_stroke"))
        scaler_path = paths_cfg.get("scaler", "models/artifacts/stroke_scaler.pkl")
        split_path = paths_cfg.get("split", "models/artifacts/stroke_split.json")

    # Load records for this phase
    records: List[StrokeRecordDict] = []

    if phase in (0, 1):
        mimic_dir = data_cfg.get("mimic3_stroke_subdir", "data/raw/stroke avns/mimic3_stroke")
        mimic_records = parse_mimic3_stroke_dir(mimic_dir, config_path)
        logger.info("Loaded %d MIMIC-3 stroke records", len(mimic_records))
        records.extend(mimic_records)

    if phase == 2:
        cves_dir = data_cfg.get("cves_subdir", "data/raw/stroke avns/cves")
        cves_records = parse_cerevasc_dir(cves_dir, config_path)
        logger.info("Loaded %d CVES records for Phase 2", len(cves_records))
        records.extend(cves_records)

    if phase == 0:  # SHAREE: include in combined (phase-0) run only
        sharee_dir = data_cfg.get("sharee_subdir", "data/raw/stroke avns/shareedb")
        label_map = {pid: 1 for pid in SHAREE_EVENT_PATIENTS}
        sharee_records = parse_sharee_dir(sharee_dir, label_map, config_path)
        logger.info("Loaded %d SHAREE records", len(sharee_records))
        records.extend(sharee_records)

    if not records:
        logger.error("No records loaded for phase %d. Check data paths in config.", phase)
        return

    logger.info("Total records for phase %d: %d", phase, len(records))

    build_stroke_cache(
        records=records,
        config_path=config_path,
        split_path=split_path,
        cache_dir=cache_dir,
        scaler_path=scaler_path,
        workers=workers,
        chunk_size=chunk_size,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Precompute HRV and denoised 10s cache for stroke AVNS.")
    p.add_argument("--config", default=CONFIG_PATH, help="Path to config_stroke.yaml")
    p.add_argument("--workers", type=int, default=1,
                   help="Worker processes (0 = all cores). Default 1 = single-threaded.")
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                   help="Sub-chunk size for parallel mode.")
    p.add_argument("--phase", type=int, choices=[0, 1, 2], default=1,
                   help="1=MIMIC-3 stroke only (pre-train), 0=all available, 2=CereVasc (blocked)")
    args = p.parse_args()
    main(config_path=args.config, workers=args.workers,
         chunk_size=args.chunk_size, phase=args.phase)
