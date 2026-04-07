"""
F20 — Tinnitus exhalation-only 6s precompute cache.

Extracts 6s PPG windows (750 samples at 125 Hz) with a two-channel input:
  channel 0: raw denoised PPG (z-normalized)
  channel 1: RIIV respiratory proxy (z-normalized)

Labels: per-frame exhalation binary array (N_windows, 30) at 5 Hz.
Quality: per-frame quality array (N_windows, 30).

Uses the same BIDMC + WESAD records and subject split as the 2s phase cache.

Usage:
    python -m src.training.tinnitus_precompute_exh_cache --config config_tinnitus.yaml
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

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
from src.features.ppg_resp import extract_ppg_respiration, generate_exhalation_labels_from_ppg
from src.features.resp_labels import generate_exhalation_labels_from_reference

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TARGET_FS = 125.0
CONFIG_PATH = "config_tinnitus.yaml"


# ---------------------------------------------------------------------------
# Two-channel window builder
# ---------------------------------------------------------------------------

def _build_2ch_window(ppg_win: np.ndarray, riiv_win: np.ndarray) -> np.ndarray:
    """Z-normalize each channel and return stacked (2, N) float32 array."""
    out = []
    for ch in (ppg_win.astype(np.float32), riiv_win.astype(np.float32)):
        std = float(ch.std())
        out.append((ch - ch.mean()) / std if std > 1e-8 else ch - ch.mean())
    return np.stack(out, axis=0)  # (2, N)


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------

def process_record_exh(
    record: TinnitusRecordDict,
    config_path: str,
    window_sec: float = 6.0,
    stride_sec: float = 0.2,
    min_valid_frames: int = 20,
    frame_rate_hz: float = 5.0,
    target_fs: float = TARGET_FS,
) -> Optional[dict]:
    """Extract 6s two-channel PPG+RIIV windows with exhalation labels.

    Returns dict with per-window arrays, or None if exhalation labeling fails.
    """
    from scipy.signal import resample as sp_resample

    ppg = np.asarray(record["ppg_signal"], dtype=np.float64).ravel()
    fs = float(record["ppg_fs"])
    sid = record["subject_id"]
    resp_signal = record.get("resp_signal")
    resp_fs = float(record.get("resp_fs") or fs)
    resp_channel = record.get("resp_channel")

    # Resample PPG to target_fs if needed (e.g. WESAD 64 Hz → 125 Hz)
    if abs(fs - target_fs) > 0.5:
        n_target = int(round(len(ppg) * target_fs / fs))
        ppg = sp_resample(ppg, n_target).astype(np.float64)
        fs = target_fs

    window_samples = int(round(target_fs * window_sec))   # 750
    stride_samples = int(round(target_fs * stride_sec))   # 25
    frame_size = int(round(target_fs / frame_rate_hz))    # 25 (125/5)
    frames_per_window = int(window_sec * frame_rate_hz)   # 30

    if len(ppg) < window_samples:
        return None

    # --- Exhalation labels for full signal ---
    exh_ok = False
    exh_result = None

    if resp_signal is not None:
        try:
            exh_result = generate_exhalation_labels_from_reference(
                resp_signal, resp_fs,
                channel_name=resp_channel,
                frame_rate_hz=frame_rate_hz,
                config_path=config_path,
            )
            exh_ok = exh_result["n_resp_cycles"] > 0
        except Exception as e:
            logger.debug("%s: reference resp labeling failed: %s", sid, e)

    if not exh_ok:
        try:
            exh_result = generate_exhalation_labels_from_ppg(
                ppg, fs, frame_rate_hz=frame_rate_hz, config_path=config_path,
            )
            exh_ok = exh_result["n_resp_cycles"] > 0
        except Exception as e:
            logger.debug("%s: PPG-derived exhalation labeling failed: %s", sid, e)

    if not exh_ok or exh_result is None:
        return None

    exh_labels = exh_result["labels"]
    exh_quality = exh_result["quality"]

    # --- RIIV respiratory proxy for full signal ---
    try:
        riiv_signal, _, _ = extract_ppg_respiration(ppg, fs, method="riiv", config_path=config_path)
    except Exception as e:
        logger.debug("%s: RIIV extraction failed, using zeros: %s", sid, e)
        riiv_signal = np.zeros(len(ppg), dtype=np.float64)

    # --- Sliding window extraction ---
    ppg_list: list = []
    exhalation_list: list = []
    quality_list: list = []
    n_skipped = 0

    max_start = len(ppg) - window_samples + 1
    for s in range(0, max_start, stride_samples):
        frame_start = s // frame_size
        frame_end = frame_start + frames_per_window

        if frame_end > len(exh_labels):
            break

        exh_win = exh_labels[frame_start:frame_end]
        exh_valid = int(np.sum(~np.isnan(exh_win)))

        if exh_valid < min_valid_frames:
            n_skipped += 1
            continue

        ppg_raw = ppg[s: s + window_samples]
        riiv_raw = riiv_signal[s: s + window_samples]
        ppg_2ch = _build_2ch_window(ppg_raw, riiv_raw)  # (2, 750)
        q_win = exh_quality[frame_start:frame_end]

        ppg_list.append(ppg_2ch)
        exhalation_list.append(exh_win)
        quality_list.append(q_win)

    if not ppg_list:
        return None

    return {
        "subject_id": sid,
        "ppg": ppg_list,
        "exhalation": exhalation_list,
        "quality": quality_list,
        "n_windows": len(ppg_list),
        "n_skipped": n_skipped,
    }


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------

def build_exh_cache(
    records: List[TinnitusRecordDict],
    config_path: str,
    split_path: str,
    cache_dir,
    window_sec: float = 6.0,
    stride_sec: float = 0.2,
    min_valid_frames: int = 20,
    frame_rate_hz: float = 5.0,
    target_fs: float = TARGET_FS,
) -> dict:
    """Build exhalation 6s cache from pre-loaded records."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    window_samples = int(round(target_fs * window_sec))  # 750
    frames_per_window = int(window_sec * frame_rate_hz)  # 30

    # Load or create subject split (reuses diastole split — same subjects)
    if not Path(split_path).exists():
        logger.info("Split not found at %s; creating 70/15/15 split...", split_path)
        create_split(records, split_path)
    train_ids, val_ids, test_ids = load_split(split_path)

    split_lookup: Dict[str, str] = {}
    for sid in train_ids:
        split_lookup[sid] = "train"
    for sid in val_ids:
        split_lookup[sid] = "val"
    for sid in test_ids:
        split_lookup[sid] = "test"

    split_ppg: Dict[str, list] = {"train": [], "val": [], "test": []}
    split_exh: Dict[str, list] = {"train": [], "val": [], "test": []}
    split_qual: Dict[str, list] = {"train": [], "val": [], "test": []}

    n_records_ok = 0
    total_windows = 0

    for rec in records:
        sid = rec["subject_id"]
        split_name = split_lookup.get(sid)
        if split_name is None:
            # Try base subject id (e.g. "wesad_S2_baseline_0" → "S2")
            for part in sid.split("_"):
                if part.startswith("S") and part[1:].isdigit():
                    split_name = split_lookup.get(part)
                    break
        if split_name is None:
            continue

        result = process_record_exh(
            rec, config_path, window_sec, stride_sec, min_valid_frames, frame_rate_hz, target_fs,
        )
        if result is None:
            continue

        split_ppg[split_name].extend(result["ppg"])
        split_exh[split_name].extend(result["exhalation"])
        split_qual[split_name].extend(result["quality"])
        n_records_ok += 1
        total_windows += result["n_windows"]

    logger.info("Processed %d records → %d total windows", n_records_ok, total_windows)

    counts: Dict[str, int] = {}
    for split_name in ["train", "val", "test"]:
        ppg_list = split_ppg[split_name]
        exh_list = split_exh[split_name]
        qual_list = split_qual[split_name]

        if ppg_list:
            ppg_arr = np.stack(ppg_list, axis=0).astype(np.float32)   # (N, 2, 750)
            exh_arr = np.stack(exh_list, axis=0).astype(np.float32)   # (N, 30)
            qual_arr = np.stack(qual_list, axis=0).astype(np.float32) # (N, 30)
        else:
            ppg_arr = np.zeros((0, 2, window_samples), dtype=np.float32)
            exh_arr = np.zeros((0, frames_per_window), dtype=np.float32)
            qual_arr = np.zeros((0, frames_per_window), dtype=np.float32)

        np.save(str(cache_dir / f"{split_name}_ppg.npy"), ppg_arr)
        np.save(str(cache_dir / f"{split_name}_exhalation.npy"), exh_arr)
        np.save(str(cache_dir / f"{split_name}_quality.npy"), qual_arr)
        counts[split_name] = int(len(ppg_arr))
        logger.info("  %s: %d windows", split_name, counts[split_name])

    meta = {
        "window_sec": window_sec,
        "stride_sec": stride_sec,
        "window_samples": window_samples,
        "frames_per_window": frames_per_window,
        "in_channels": 2,
        "channels": ["ppg", "riiv"],
        "counts": counts,
        "total_windows": total_windows,
    }
    with open(str(cache_dir / "exh_cache_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Cache saved to %s", cache_dir)
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description="Precompute tinnitus exhalation 6s cache (F20)")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config_path = args.config
    if not Path(config_path).is_absolute() and not Path(config_path).exists():
        config_path = str(root / config_path)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    exh_pre_cfg = cfg.get("exh_phase_precompute", {})
    paths_exh_cfg = cfg.get("paths_exh", {})
    paths_cfg = cfg.get("paths", {})

    def _resolve(p: str) -> str:
        return p if Path(p).is_absolute() else str(root / p)

    bidmc_dir = _resolve(data_cfg.get("bidmc_subdir", "data/raw/stroke avns/bidmc"))
    wesad_dir = _resolve(data_cfg.get("wesad_subdir", "data/raw/tinnitus avns/wesad/WESAD"))
    cache_dir = _resolve(paths_exh_cfg.get("cache_dir", "models/artifacts/cache_tinnitus_exh"))
    split_path = _resolve(paths_cfg.get("split_path", "models/artifacts/tinnitus_phase_split.json"))

    window_sec = float(exh_pre_cfg.get("window_sec", 6.0))
    stride_sec = float(exh_pre_cfg.get("stride_sec", 0.2))
    min_valid_frames = int(exh_pre_cfg.get("min_valid_frames", 20))
    frame_rate_hz = float(exh_pre_cfg.get("frame_rate_hz", 5.0))

    logger.info("Loading BIDMC records from %s", bidmc_dir)
    bidmc_records = parse_bidmc_ppg_dir(bidmc_dir, config_path=config_path)
    logger.info("Loading WESAD records from %s", wesad_dir)
    wesad_records = parse_wesad_dir(wesad_dir, config_path=config_path)
    all_records = bidmc_records + wesad_records
    logger.info("Total records: %d (BIDMC=%d, WESAD=%d)",
                len(all_records), len(bidmc_records), len(wesad_records))

    build_exh_cache(
        all_records, config_path, split_path, cache_dir,
        window_sec=window_sec,
        stride_sec=stride_sec,
        min_valid_frames=min_valid_frames,
        frame_rate_hz=frame_rate_hz,
    )


if __name__ == "__main__":
    main()
