"""
Signal denoising via Continuous Wavelet Transform (CWT). FR-2.1.
Removes baseline wander (<0.5 Hz) and high-frequency noise.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pywt
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.isabs(config_path) and not os.path.isfile(config_path):
        root = Path(__file__).resolve().parents[2]
        config_path = root / config_path
    with open(config_path) as f:
        return yaml.safe_load(f)


def denoise(
    signal: np.ndarray,
    fs: float,
    family: Optional[str] = None,
    scale_range: Optional[Tuple[float, float]] = None,
    config_path: str = "config.yaml",
    lowcut_hz: float = 0.5,
    highcut_hz: float = 45.0,
) -> np.ndarray:
    """
    Remove baseline wander (< lowcut_hz) and high-frequency noise (> highcut_hz)
    using CWT. Returns 1D array same length as signal.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if family is None or scale_range is None:
        config = load_config(config_path)
        wav = config.get("wavelet", {})
        family = family or wav.get("family", "cmor")
        sr = wav.get("scale_range", [1, 64])
        scale_range = scale_range or (float(sr[0]), float(sr[1]))

    dt = 1.0 / fs
    # Build scale array for CWT; cmor has center frequency ~1
    scales = np.arange(scale_range[0], scale_range[1] + 1, dtype=np.float64)
    wavelet = f"{family}1.5-1.0" if family == "cmor" else f"{family}2"
    try:
        coeffs, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=dt)
    except Exception:
        wavelet = "mexh"  # fallback
        coeffs, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=dt)

    # Keep only passband: lowcut_hz < freq < highcut_hz
    mask = (freqs >= lowcut_hz) & (freqs <= highcut_hz)
    coeffs_masked = np.where(mask[:, np.newaxis], coeffs, 0.0)

    # Reconstruct: approximate inverse CWT by summing selected scales (normalized)
    n_keep = max(1, np.sum(mask))
    reconstructed = np.real(np.sum(coeffs_masked, axis=0)) / n_keep
    # Preserve length (cwt can change length in some wavelets; trim/pad to match)
    if len(reconstructed) != len(signal):
        reconstructed = np.resize(reconstructed, len(signal))
    return reconstructed.astype(np.float64)
