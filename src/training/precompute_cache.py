"""
One-time precompute: HRV and denoised 10s for train/val/test. Saves cache for fast training.
Run once after data/split exist; then use train --use-cache for ~1 h/epoch.
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

from src.data.dataloaders import get_dataloaders
from src.features.pipeline import waveform_10s_denoised, waveform_to_hrv_sequence
from src.features.scaler import fit_scaler, load_scaler, transform

from .build_model import load_config
from .precompute_worker import process_chunk as _process_chunk

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = "config.yaml"
MAX_SHORT_LEN_CAP = 3600  # 12 s at 300 Hz for variable fs
N_FEATURES = 7
CHUNK_SIZE = 32


def main(config_path: str = CONFIG_PATH, workers: int = 1, chunk_size: int = CHUNK_SIZE):
    if workers == 0:
        workers = os.cpu_count() or 4
    # Resolve to absolute so worker processes (which may have different cwd) always find config
    if not os.path.isabs(config_path):
        config_path = str(Path(config_path).resolve())
    config = load_config(config_path)
    paths_cfg = config.get("paths", {})
    cache_dir = Path(paths_cfg.get("cache_dir", "models/artifacts/cache"))
    scaler_path = paths_cfg.get("scaler", "models/artifacts/scaler.pkl")
    cache_dir.mkdir(parents=True, exist_ok=True)
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(
        config_path=config_path,
        create_split_if_missing=True,
    )
    splits = [
        ("train", train_loader.dataset),
        ("val", val_loader.dataset),
        ("test", test_loader.dataset),
    ]

    split_lengths = {}
    short_lists = {"train": [], "val": [], "test": []}

    for split_name, dataset in splits:
        n = len(dataset)
        if n == 0:
            logger.warning("Empty %s set; skipping.", split_name)
            split_lengths[split_name] = 0
            continue
        short_list = []
        hrv_list = []
        labels_list = []

        if workers == 1:
            for i in tqdm(range(n), desc=f"Precompute {split_name}", unit="sample"):
                short_t, long_t, label_t, fs = dataset[i]
                long_np = long_t.squeeze(0).numpy()
                short_np = short_t.squeeze(0).numpy()
                fs = float(fs)
                hrv = waveform_to_hrv_sequence(long_np, fs, config_path=config_path)
                short_d = waveform_10s_denoised(short_np, fs, config_path=config_path)
                short_list.append(np.asarray(short_d, dtype=np.float32))
                hrv_list.append(hrv)
                labels_list.append(float(label_t.item()))
        else:
            def _chunk_iter():
                for start in range(0, n, chunk_size):
                    indices = range(start, min(start + chunk_size, n))
                    chunk_data = []
                    for i in indices:
                        short_t, long_t, label_t, fs = dataset[i]
                        long_np = long_t.squeeze(0).numpy()
                        short_np = short_t.squeeze(0).numpy()
                        chunk_data.append((short_np, long_np, float(fs), float(label_t.item())))
                    yield (chunk_data, config_path)

            n_chunks = (n + chunk_size - 1) // chunk_size
            with ProcessPoolExecutor(max_workers=workers) as executor:
                chunk_results = list(
                    tqdm(
                        executor.map(_process_chunk, _chunk_iter()),
                        total=n_chunks,
                        desc=f"Precompute {split_name}",
                        unit="chunk",
                    )
                )
            for results in chunk_results:
                for short_d, hrv, label in results:
                    short_list.append(short_d)
                    hrv_list.append(hrv)
                    labels_list.append(label)

        short_lists[split_name] = short_list
        split_lengths[split_name] = n

        hrv_arr = np.stack(hrv_list, axis=0).astype(np.float32)
        labels_arr = np.array(labels_list, dtype=np.float32)
        np.save(cache_dir / f"{split_name}_hrv.npy", hrv_arr)
        np.save(cache_dir / f"{split_name}_labels.npy", labels_arr)

    # Global max length for 10s (cap at MAX_SHORT_LEN_CAP)
    all_shorts = short_lists["train"] + short_lists["val"] + short_lists["test"]
    if not all_shorts:
        raise ValueError("No samples in any split; cannot build cache.")
    max_short_len = min(max(s.size for s in all_shorts), MAX_SHORT_LEN_CAP)

    def pad_and_save(short_list, split_name):
        n = len(short_list)
        out = np.zeros((n, max_short_len), dtype=np.float32)
        for i, s in enumerate(short_list):
            L = min(s.size, max_short_len)
            out[i, :L] = s.ravel()[:L]
        np.save(cache_dir / f"{split_name}_short.npy", out)

    for name in ("train", "val", "test"):
        pad_and_save(short_lists.get(name, []), name)

    # Write empty arrays for splits that had no samples so loaders can always open files
    for name in ("train", "val", "test"):
        if split_lengths.get(name, 0) == 0:
            hrv_steps = int(300 / float(config.get("hrv", {}).get("subwindow_sec", 60)))
            np.save(cache_dir / f"{name}_hrv.npy", np.zeros((0, hrv_steps, N_FEATURES), dtype=np.float32))
            np.save(cache_dir / f"{name}_labels.npy", np.zeros(0, dtype=np.float32))

    # Fit scaler on train HRV only
    train_hrv_path = cache_dir / "train_hrv.npy"
    if not train_hrv_path.exists():
        raise ValueError("Train set empty or missing; cannot fit scaler.")
    train_hrv = np.load(train_hrv_path)
    X = train_hrv.reshape(-1, 7)
    fit_scaler(X, path=scaler_path, config_path=config_path)
    logger.info("Fitted scaler on %d HRV rows (per-column NaN handled internally), saved to %s", X.shape[0], scaler_path)

    scaler = load_scaler(path=scaler_path, config_path=config_path)
    for split_name in ("train", "val", "test"):
        path_hrv = cache_dir / f"{split_name}_hrv.npy"
        if not path_hrv.exists():
            continue
        hrv = np.load(path_hrv).astype(np.float32)
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
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Samples per chunk when using --workers > 1.")
    args = p.parse_args()
    main(config_path=args.config, workers=args.workers, chunk_size=args.chunk_size)
