"""
Streaming precompute: HRV and denoised 10s for train/val/test.

Loads one record at a time (not all at once) to avoid OOM on large datasets.
Run once after data exists; then use train --use-cache for fast epochs.
Use --workers N (or 0 for all cores) for parallel speedup.
"""

import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from src.data.dataset_parsers import collect_all_subject_ids, iter_all_records
from src.data.splitter import create_split_from_ids, load_split
from src.features.pipeline import waveform_10s_denoised, waveform_to_hrv_sequence
from src.features.scaler import fit_scaler, load_scaler, transform

from .build_model import load_config
from .precompute_worker import process_chunk as _process_chunk

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = "config.yaml"
MAX_SHORT_LEN_CAP = 3600  # 12 s at 300 Hz for variable fs
N_FEATURES = 7
CHUNK_SIZE = 8


def main(config_path: str = CONFIG_PATH, workers: int = 1, chunk_size: int = CHUNK_SIZE):
    if workers == 0:
        workers = os.cpu_count() or 4
    # Resolve to absolute so worker processes always find config
    if not os.path.isabs(config_path):
        config_path = str(Path(config_path).resolve())
    config = load_config(config_path)
    paths_cfg = config.get("paths", {})
    data_cfg = config.get("data", {})
    cache_dir = Path(paths_cfg.get("cache_dir", "models/artifacts/cache"))
    scaler_path = paths_cfg.get("scaler", "models/artifacts/scaler.pkl")
    cache_dir.mkdir(parents=True, exist_ok=True)
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)

    waveform_sec = float(data_cfg.get("waveform_sec", 10))
    long_sec = float(data_cfg.get("hrv_window_sec", 300))
    # Default stride = long_sec (non-overlapping 5-min windows)
    stride_sec = float(data_cfg.get("stride_sec", long_sec))

    # --- Load or create split (lightweight, no signal loading) ---
    split_path = data_cfg.get("split_path", paths_cfg.get("split", "models/artifacts/split.json"))
    if not Path(split_path).exists():
        logger.info("Split file not found; collecting subject IDs for split creation...")
        all_ids = collect_all_subject_ids(config_path)
        logger.info("Collected %d subject IDs, creating 70/15/15 split", len(all_ids))
        create_split_from_ids(all_ids, split_path)

    train_ids, val_ids, test_ids = load_split(split_path)
    split_lookup = {}
    for sid in train_ids:
        split_lookup[sid] = "train"
    for sid in val_ids:
        split_lookup[sid] = "val"
    for sid in test_ids:
        split_lookup[sid] = "test"
    logger.info("Split loaded: %d train, %d val, %d test subjects",
                len(train_ids), len(val_ids), len(test_ids))

    # --- Accumulators for processed results (small per window) ---
    accum = {name: {"short": [], "hrv": [], "labels": []} for name in ("train", "val", "test")}
    max_short_len = 0
    n_records = 0
    n_windows = 0
    n_skipped = 0

    # --- Create executor once for parallel mode ---
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None

    try:
        pbar = tqdm(iter_all_records(config_path), desc="Streaming records", unit="rec")
        for rec in pbar:
            sid = rec["subject_id"]
            split_name = split_lookup.get(sid)
            if split_name is None:
                n_skipped += 1
                continue

            if rec.get("label") is None or rec["label"] < 0:
                n_skipped += 1
                continue

            sig = rec["signal"]
            fs = float(rec["fs"])
            label = float(rec["label"])

            n_short = int(fs * waveform_sec)
            n_long = int(fs * long_sec)
            stride = int(fs * stride_sec)

            if sig.size < n_short:
                n_skipped += 1
                continue

            # Build window starts (same logic as PhysioDataset.__init__)
            if sig.size >= n_long:
                starts = list(range(0, sig.size - n_long + 1, stride))
            else:
                # Short record (>= 10s but < 300s): single window
                starts = [0]

            if not starts:
                n_skipped += 1
                continue

            # Build chunk_data: list of (short_np, long_np, fs, label)
            chunk_data = []
            for start in starts:
                short_np = sig[start: start + n_short]
                if sig.size >= start + n_long:
                    long_np = sig[start: start + n_long]
                else:
                    long_np = sig  # short record: use full signal
                chunk_data.append((short_np, long_np, fs, label))

            # Process windows
            if executor is None or len(chunk_data) <= chunk_size:
                # Single-threaded (or few windows — not worth pool overhead)
                for short_np, long_np, fs_val, lbl in chunk_data:
                    hrv = waveform_to_hrv_sequence(long_np, fs_val, config_path=config_path)
                    short_d = waveform_10s_denoised(short_np, fs_val, config_path=config_path)
                    short_d = np.asarray(short_d, dtype=np.float32)
                    accum[split_name]["short"].append(short_d)
                    accum[split_name]["hrv"].append(hrv)
                    accum[split_name]["labels"].append(lbl)
                    if short_d.size > max_short_len:
                        max_short_len = short_d.size
            else:
                # Parallel: submit sub-chunks to worker pool
                futures = []
                for sub_start in range(0, len(chunk_data), chunk_size):
                    sub = chunk_data[sub_start: sub_start + chunk_size]
                    futures.append(executor.submit(_process_chunk, (sub, config_path)))
                for fut in futures:
                    for short_d, hrv, lbl in fut.result():
                        accum[split_name]["short"].append(short_d)
                        accum[split_name]["hrv"].append(hrv)
                        accum[split_name]["labels"].append(lbl)
                        if short_d.size > max_short_len:
                            max_short_len = short_d.size

            rec_windows = len(chunk_data)
            n_windows += rec_windows
            n_records += 1
            pbar.set_postfix(recs=n_records, wins=n_windows, split=split_name)

            # Free the raw signal
            del sig, chunk_data, rec

        pbar.close()
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    logger.info("Processed %d records -> %d windows (skipped %d records)",
                n_records, n_windows, n_skipped)

    # --- Save HRV, labels, and padded short waveforms per split ---
    max_short_len = min(max_short_len, MAX_SHORT_LEN_CAP)
    if max_short_len == 0:
        raise ValueError("No samples in any split; cannot build cache.")

    split_lengths = {}
    for name in ("train", "val", "test"):
        short_list = accum[name]["short"]
        hrv_list = accum[name]["hrv"]
        labels_list = accum[name]["labels"]
        n = len(labels_list)
        split_lengths[name] = n

        if n == 0:
            hrv_steps = int(long_sec / float(config.get("hrv", {}).get("subwindow_sec", 60)))
            np.save(cache_dir / f"{name}_hrv.npy",
                    np.zeros((0, hrv_steps, N_FEATURES), dtype=np.float32))
            np.save(cache_dir / f"{name}_labels.npy", np.zeros(0, dtype=np.float32))
            np.save(cache_dir / f"{name}_short.npy", np.zeros((0, 0), dtype=np.float32))
            logger.warning("Empty %s set; wrote empty arrays.", name)
            continue

        # HRV
        hrv_arr = np.stack(hrv_list, axis=0).astype(np.float32)
        np.save(cache_dir / f"{name}_hrv.npy", hrv_arr)
        # Labels
        np.save(cache_dir / f"{name}_labels.npy", np.array(labels_list, dtype=np.float32))
        # Padded short waveforms
        out = np.zeros((n, max_short_len), dtype=np.float32)
        for i, s in enumerate(short_list):
            L = min(s.size, max_short_len)
            out[i, :L] = s.ravel()[:L]
        np.save(cache_dir / f"{name}_short.npy", out)
        logger.info("Saved %s: %d windows, short=(%d,%d), hrv=%s",
                     name, n, n, max_short_len, hrv_arr.shape)

    del accum  # free accumulated results

    # --- Fit scaler on train HRV only ---
    train_hrv_path = cache_dir / "train_hrv.npy"
    if not train_hrv_path.exists() or split_lengths.get("train", 0) == 0:
        raise ValueError("Train set empty or missing; cannot fit scaler.")
    train_hrv = np.load(train_hrv_path)
    X = train_hrv.reshape(-1, N_FEATURES)
    fit_scaler(X, path=scaler_path, config_path=config_path)
    logger.info("Fitted scaler on %d HRV rows, saved to %s", X.shape[0], scaler_path)

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
        logger.info("  %s: %.1f%% of HRV rows had NaN before imputation", split_name, nan_frac * 100)
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


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Precompute HRV and denoised 10s for fast training.")
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (0 = use all CPU cores). Default 1 = single-threaded.",
    )
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                    help="Samples per chunk when using --workers > 1.")
    args = p.parse_args()
    main(config_path=args.config, workers=args.workers, chunk_size=args.chunk_size)
