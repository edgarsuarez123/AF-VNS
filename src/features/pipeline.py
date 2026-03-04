"""
High-level pipeline: one 5-min waveform -> sequence of HRV vectors for RNN.
Feature order: rmssd, sdnn, lf, hf, lf_hf_ratio, sampen, dfa_alpha1 (7 features).
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from .artifact_scrubber import should_reject_window
from .hrv_freq import compute_hrv_freq
from .hrv_nonlinear import compute_hrv_nonlinear
from .hrv_time import compute_hrv_time
from .peak_detector import get_rr_intervals
from .wavelet_filter import denoise

FEATURE_ORDER = ("rmssd", "sdnn", "lf", "hf", "lf_hf_ratio", "sampen", "dfa_alpha1")
N_FEATURES = len(FEATURE_ORDER)
WINDOW_5MIN_SEC = 300.0


def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.isabs(config_path) and not os.path.isfile(config_path):
        root = Path(__file__).resolve().parents[2]
        config_path = root / config_path
    with open(config_path) as f:
        return yaml.safe_load(f)


def _hrv_dict_to_vector(d_time: dict, d_freq: dict, d_nl: dict) -> np.ndarray:
    """Build one feature vector in fixed order; NaNs where missing."""
    vec = np.full(N_FEATURES, np.nan, dtype=np.float32)
    vec[0] = d_time.get("rmssd", np.nan)
    vec[1] = d_time.get("sdnn", np.nan)
    vec[2] = d_freq.get("lf", np.nan)
    vec[3] = d_freq.get("hf", np.nan)
    vec[4] = d_freq.get("lf_hf_ratio", np.nan)
    vec[5] = d_nl.get("sampen", np.nan)
    vec[6] = d_nl.get("dfa_alpha1", np.nan)
    return vec


def waveform_to_hrv_sequence(
    signal_5min: np.ndarray,
    fs: float,
    subwindow_sec: float = 30.0,
    config_path: str = "config.yaml",
) -> np.ndarray:
    """
    Convert a 5-minute waveform to a sequence of HRV feature vectors (one per sub-window).
    Returns array of shape (n_time_steps, n_features), dtype float32.
    n_time_steps = floor(300 / subwindow_sec). Rejected or insufficient windows yield NaN rows.
    """
    signal_5min = np.asarray(signal_5min, dtype=np.float64).ravel()
    config = load_config(config_path)
    subwindow_sec = float(subwindow_sec or config.get("hrv", {}).get("subwindow_sec", 30.0))
    n_steps = int(WINDOW_5MIN_SEC / subwindow_sec)
    out = np.full((n_steps, N_FEATURES), np.nan, dtype=np.float32)

    denoised = denoise(signal_5min, fs, config_path=config_path)
    n_per_sub = int(fs * subwindow_sec)
    if denoised.size < n_steps * n_per_sub:
        return out

    for i in range(n_steps):
        start = i * n_per_sub
        seg = denoised[start : start + n_per_sub]
        rr = get_rr_intervals(seg, fs)
        if should_reject_window(seg, rr, config_path=config_path):
            continue  # leave row as NaN
        if len(rr) < 5:
            continue
        d_time = compute_hrv_time(rr)
        d_freq = compute_hrv_freq(rr)
        d_nl = compute_hrv_nonlinear(rr)
        out[i] = _hrv_dict_to_vector(d_time, d_freq, d_nl)

    return out


def waveform_10s_denoised(signal_10s: np.ndarray, fs: float, config_path: str = "config.yaml") -> np.ndarray:
    """Denoise only a 10 s segment for the CNN branch. Returns 1D float64."""
    return denoise(
        np.asarray(signal_10s, dtype=np.float64).ravel(),
        fs,
        config_path=config_path,
    ).astype(np.float64)
