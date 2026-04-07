"""
F13 — Tinnitus Arousal Precompute Cache.

Extracts 5 EDA arousal features per 60s sliding window from WESAD records.
Saves train/val/test feature arrays with subject-level 70/15/15 split.

Features (per 60s window):
  tonic_scl_mean   — mean tonic skin conductance level
  tonic_scl_std    — std of tonic SCL (stability measure)
  phasic_mean      — mean phasic amplitude
  max_scr_amplitude — peak SCR amplitude in window
  scr_rate         — skin conductance response events per minute

Usage:
    python -m src.training.tinnitus_precompute_arousal --config config_tinnitus.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "tonic_scl_mean",
    "tonic_scl_std",
    "phasic_mean",
    "max_scr_amplitude",
    "scr_rate",
]
FEATURE_NAMES_V2 = FEATURE_NAMES + ["skin_temp_mean"]  # F19: adds skin temperature
N_FEATURES = len(FEATURE_NAMES)
N_FEATURES_V2 = len(FEATURE_NAMES_V2)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> dict:
    root = Path(__file__).resolve().parents[2]
    if not Path(config_path).is_absolute() and not Path(config_path).exists():
        config_path = str(root / config_path)
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_eda_features_window(
    eda_window: np.ndarray,
    fs: float,
    scr_min_amplitude: float = 0.02,
    temp_window: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Extract EDA features from a single window.

    Parameters
    ----------
    eda_window : 1D float64 EDA signal in µS
    fs : sampling rate in Hz
    scr_min_amplitude : minimum phasic amplitude to count as an SCR
    temp_window : optional 1D float64 skin temperature signal (°C). When provided,
                  appends skin_temp_mean as 6th feature (F19).

    Returns
    -------
    (5,) or (6,) float32 — all NaN if decomposition fails or signal too short.
    Number of features equals N_FEATURES (5) when temp_window is None,
    N_FEATURES_V2 (6) when temp_window is provided.
    """
    from src.features.eda import decompose_eda, detect_scr_peaks

    n_out = N_FEATURES_V2 if temp_window is not None else N_FEATURES
    min_samples = int(fs * 10)  # require at least 10s
    if len(eda_window) < min_samples:
        return np.full(n_out, np.nan, dtype=np.float32)

    try:
        decomposed = decompose_eda(eda_window, fs)
        tonic = decomposed["tonic"]
        phasic = decomposed["phasic"]

        tonic_scl_mean = float(np.nanmean(tonic))
        tonic_scl_std = float(np.nanstd(tonic))
        phasic_mean = float(np.nanmean(phasic))

        peaks = detect_scr_peaks(phasic, fs, min_amplitude=scr_min_amplitude)
        max_scr_amplitude = (
            float(np.max(peaks["amplitudes"])) if peaks["count"] > 0 else 0.0
        )
        duration_min = len(eda_window) / fs / 60.0
        scr_rate = float(peaks["count"] / duration_min) if duration_min > 0 else 0.0

        feats = [tonic_scl_mean, tonic_scl_std, phasic_mean, max_scr_amplitude, scr_rate]
        if temp_window is not None:
            skin_temp_mean = float(np.nanmean(temp_window)) if len(temp_window) > 0 else float("nan")
            feats.append(skin_temp_mean)
        return np.array(feats, dtype=np.float32)
    except Exception as exc:
        logger.debug("EDA feature extraction failed: %s", exc)
        return np.full(n_out, np.nan, dtype=np.float32)


# ---------------------------------------------------------------------------
# Subject ID helpers
# ---------------------------------------------------------------------------

def _get_base_subject_id(record: dict) -> str:
    """Extract base subject ID (e.g. 'S2') from TinnitusRecordDict."""
    session_id = record.get("session_id")
    if session_id:
        return str(session_id)
    # Fallback: parse from subject_id like "wesad_S2_baseline_0"
    for part in record.get("subject_id", "").split("_"):
        if part.startswith("S") and part[1:].isdigit():
            return part
    return record.get("subject_id", "unknown")


# ---------------------------------------------------------------------------
# Main precompute
# ---------------------------------------------------------------------------

def precompute_arousal_cache(
    config_path: str = "config_tinnitus.yaml",
    config_section: str = "arousal_classifier",
) -> dict:
    """Extract EDA arousal features from WESAD and save to cache.

    Returns a dict with counts: {"train": N, "val": N, "test": N, "total": N}
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    cfg = _load_config(config_path)
    eda_cfg = cfg.get("eda", {})
    ac_cfg = cfg.get(config_section, cfg.get("arousal_classifier", {}))
    data_cfg = cfg.get("data", {})
    use_temp = "skin_temp_mean" in ac_cfg.get("features", [])

    wesad_dir = data_cfg.get("wesad_subdir", "data/raw/tinnitus avns/wesad/WESAD")
    window_sec = float(ac_cfg.get("window_sec", 60))
    hop_sec = float(ac_cfg.get("hop_sec", 30))
    scr_min_amplitude = float(eda_cfg.get("scr_min_amplitude", 0.02))
    cache_dir = ac_cfg.get("cache_dir", "models/artifacts/cache_tinnitus_arousal")
    split_path = ac_cfg.get("split_path", "models/artifacts/tinnitus_arousal_split.json")

    root = Path(__file__).resolve().parents[2]
    for attr_name, val in [("wesad_dir", wesad_dir), ("cache_dir", cache_dir), ("split_path", split_path)]:
        pass  # resolved below

    def _resolve(p: str) -> str:
        return p if Path(p).is_absolute() else str(root / p)

    wesad_dir = _resolve(wesad_dir)
    cache_dir = _resolve(cache_dir)
    split_path = _resolve(split_path)

    from src.data.tinnitus_parsers import parse_wesad_dir
    from src.data.splitter import create_split_from_ids, load_split

    logger.info("Loading WESAD records from %s", wesad_dir)
    records = parse_wesad_dir(wesad_dir, config_path=config_path)
    logger.info("Loaded %d WESAD records", len(records))

    if len(records) == 0:
        logger.error("No WESAD records found. Check data.wesad_subdir in config.")
        return {"train": 0, "val": 0, "test": 0, "total": 0}

    # Unique base subject IDs for splitting
    all_sids = sorted({_get_base_subject_id(r) for r in records})

    if Path(split_path).exists():
        train_ids, val_ids, test_ids = load_split(split_path)
        logger.info("Loaded existing split: %d/%d/%d subjects",
                    len(train_ids), len(val_ids), len(test_ids))
    else:
        train_ids, val_ids, test_ids = create_split_from_ids(all_sids, split_path)
        logger.info("Created new split: %d/%d/%d subjects",
                    len(train_ids), len(val_ids), len(test_ids))

    splits = {"train": set(train_ids), "val": set(val_ids), "test": set(test_ids)}

    # Accumulate windows per split
    split_features: dict = {"train": [], "val": [], "test": []}
    split_labels: dict = {"train": [], "val": [], "test": []}

    skipped = 0
    for rec in records:
        base_sid = _get_base_subject_id(rec)
        split_name = next((s for s, ids in splits.items() if base_sid in ids), None)
        if split_name is None:
            skipped += 1
            continue

        eda = rec.get("eda_signal")
        eda_fs = rec.get("eda_fs")
        temp = rec.get("temp_signal")  # F19: optional skin temperature
        temp_fs = rec.get("temp_fs")
        label = int(rec.get("label", 0))

        if eda is None or eda_fs is None or len(eda) == 0:
            skipped += 1
            continue

        win_samples = int(window_sec * eda_fs)
        hop_samples = max(1, int(hop_sec * eda_fs))

        def _get_temp_win(start_eda: int, end_eda: int) -> Optional[np.ndarray]:
            """Slice temp aligned to EDA window (both at same FS)."""
            if not use_temp or temp is None or temp_fs is None:
                return None
            ratio = float(temp_fs) / float(eda_fs)
            t_start = int(start_eda * ratio)
            t_end = int(end_eda * ratio)
            t_end = min(t_end, len(temp))
            return temp[t_start:t_end].astype(np.float64) if t_end > t_start else None

        if len(eda) < win_samples:
            # Short epoch: use whole signal if at least 10s
            feats = extract_eda_features_window(
                eda, eda_fs, scr_min_amplitude, _get_temp_win(0, len(eda)))
            split_features[split_name].append(feats)
            split_labels[split_name].append(label)
        else:
            start = 0
            while start + win_samples <= len(eda):
                win = eda[start:start + win_samples]
                feats = extract_eda_features_window(
                    win, eda_fs, scr_min_amplitude, _get_temp_win(start, start + win_samples))
                split_features[split_name].append(feats)
                split_labels[split_name].append(label)
                start += hop_samples

    if skipped > 0:
        logger.warning("Skipped %d records (no split assignment or missing EDA)", skipped)

    # Save cache
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    counts: dict = {}

    for split_name in ["train", "val", "test"]:
        feats_list = split_features[split_name]
        labels_list = split_labels[split_name]

        n_feat = N_FEATURES_V2 if use_temp else N_FEATURES
        if feats_list:
            feats_arr = np.stack(feats_list, axis=0).astype(np.float32)
            labels_arr = np.array(labels_list, dtype=np.int32)
        else:
            feats_arr = np.zeros((0, n_feat), dtype=np.float32)
            labels_arr = np.zeros(0, dtype=np.int32)

        np.save(str(Path(cache_dir) / f"{split_name}_features.npy"), feats_arr)
        np.save(str(Path(cache_dir) / f"{split_name}_labels.npy"), labels_arr)
        counts[split_name] = int(len(feats_arr))

    total = sum(counts.values())

    active_feature_names = FEATURE_NAMES_V2 if use_temp else FEATURE_NAMES
    meta = {
        "feature_names": active_feature_names,
        "n_features": len(active_feature_names),
        "window_sec": window_sec,
        "hop_sec": hop_sec,
        "counts": counts,
        "total": total,
    }
    with open(str(Path(cache_dir) / "arousal_cache_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "Cache saved to %s | total=%d (train=%d val=%d test=%d)",
        cache_dir, total, counts["train"], counts["val"], counts["test"],
    )
    return {**counts, "total": total}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute tinnitus arousal EDA feature cache from WESAD"
    )
    parser.add_argument("--config", default="config_tinnitus.yaml",
                        help="Path to config YAML (default: config_tinnitus.yaml)")
    parser.add_argument("--config-section", default="arousal_classifier",
                        help="Config section to read (e.g. arousal_classifier_v2 for F19)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    precompute_arousal_cache(args.config, config_section=args.config_section)


if __name__ == "__main__":
    main()
