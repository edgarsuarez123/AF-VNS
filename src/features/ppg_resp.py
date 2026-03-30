"""
PPG-Derived Respiration — exhalation phase label generation for tinnitus aVNS.

Extracts a respiratory proxy signal from PPG using one of three methods:
  - RIIV (Respiratory-Induced Intensity Variation): PPG peak amplitudes
    oscillate at the respiratory rate due to venous return modulation.
  - RIFV (Respiratory-Induced Frequency Variation): inter-beat intervals
    vary with respiration via RSA (respiratory sinus arrhythmia).
  - baseline: low-frequency PPG baseline is modulated by intrathoracic pressure.

Return interface matches edr.generate_exhalation_labels() for drop-in use in
the PhaseDetector multi-task training pipeline.

SBIR reference: >80% agreement with impedance pneumography (BIDMC validation).
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml
from scipy.interpolate import interp1d
from scipy.signal import detrend, find_peaks

from src.features.ppg_filter import denoise_ppg
from src.features.ppg_phase_labels import get_ppg_peak_indices
from src.features.resp_labels import (
    _bandpass_filter,
    _build_resp_phase_array,
    _downsample_to_frames,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config + empty result
# ---------------------------------------------------------------------------

def _load_config(config_path: str = "config_tinnitus.yaml") -> dict:
    if not os.path.isabs(config_path) and not os.path.isfile(config_path):
        root = Path(__file__).resolve().parents[2]
        config_path = root / config_path
    with open(config_path) as f:
        return yaml.safe_load(f)


def _empty_result(n_frames: int) -> dict:
    """Return all-NaN result dict for unprocessable signals."""
    return {
        "labels": np.full(n_frames, np.nan, dtype=np.float32),
        "quality": np.zeros(n_frames, dtype=np.float32),
        "n_resp_cycles": 0,
        "resp_rate_bpm": np.nan,
    }


# ---------------------------------------------------------------------------
# Extraction methods (private)
# ---------------------------------------------------------------------------

def _riiv_respiratory_signal(
    ppg_signal: np.ndarray,
    peak_indices: np.ndarray,
    fs: float,
    bandpass_low: float,
    bandpass_high: float,
    bandpass_order: int,
) -> np.ndarray:
    """Extract respiratory proxy via Respiratory-Induced Intensity Variation.

    PPG pulse amplitudes are modulated by respiration: the venous return
    (and thus pulse height) oscillates at the breathing rate. Interpolate
    peak amplitudes to a continuous signal and bandpass to the resp band.

    Adapted from edr._qrs_amplitude_edr(), replacing R-peaks with PPG systolic
    peaks.
    """
    if len(peak_indices) < 3:
        return np.zeros(len(ppg_signal), dtype=np.float64)

    amplitudes = ppg_signal[peak_indices].astype(np.float64)
    times = peak_indices / fs
    t_full = np.arange(len(ppg_signal)) / fs

    try:
        interp_fn = interp1d(
            times, amplitudes, kind="cubic",
            bounds_error=False, fill_value=(amplitudes[0], amplitudes[-1]),
        )
        amp_continuous = interp_fn(t_full)
    except ValueError:
        # Fallback to linear if cubic fails (too few support points)
        interp_fn = interp1d(
            times, amplitudes, kind="linear",
            bounds_error=False, fill_value=(amplitudes[0], amplitudes[-1]),
        )
        amp_continuous = interp_fn(t_full)

    # Linear detrend before bandpass — removes slow baseline drift that can
    # alias into the respiratory band on long (>5 min) recordings.
    amp_continuous = detrend(amp_continuous, type="linear")

    return _bandpass_filter(amp_continuous, fs, bandpass_low, bandpass_high, bandpass_order)


def _rifv_respiratory_signal(
    peak_indices: np.ndarray,
    fs: float,
    bandpass_low: float,
    bandpass_high: float,
    bandpass_order: int,
    n_samples: int,
) -> np.ndarray:
    """Extract respiratory proxy via Respiratory-Induced Frequency Variation.

    Heart rate oscillates at the respiratory rate (RSA). Interpolate the
    inter-beat interval (IBI) series to a continuous signal and bandpass to
    the resp band.
    """
    if len(peak_indices) < 3:
        return np.zeros(n_samples, dtype=np.float64)

    ibi = np.diff(peak_indices) / fs  # seconds
    # Place each IBI at the midpoint between the two bounding peaks
    ibi_times = (peak_indices[:-1] + peak_indices[1:]) / (2.0 * fs)
    t_full = np.arange(n_samples) / fs

    try:
        interp_fn = interp1d(
            ibi_times, ibi, kind="cubic",
            bounds_error=False, fill_value=(ibi[0], ibi[-1]),
        )
        ibi_continuous = interp_fn(t_full)
    except ValueError:
        interp_fn = interp1d(
            ibi_times, ibi, kind="linear",
            bounds_error=False, fill_value=(ibi[0], ibi[-1]),
        )
        ibi_continuous = interp_fn(t_full)

    ibi_continuous = detrend(ibi_continuous, type="linear")
    return _bandpass_filter(ibi_continuous, fs, bandpass_low, bandpass_high, bandpass_order)


def _baseline_respiratory_signal(
    ppg_signal: np.ndarray,
    fs: float,
    bandpass_low: float,
    bandpass_high: float,
    bandpass_order: int,
) -> np.ndarray:
    """Extract respiratory proxy via baseline wander.

    Intrathoracic pressure changes from breathing directly modulate the
    PPG DC baseline. A simple bandpass to the respiratory band isolates it.
    """
    return _bandpass_filter(ppg_signal, fs, bandpass_low, bandpass_high, bandpass_order)


def _find_resp_peaks(
    resp_signal: np.ndarray,
    fs: float,
    max_resp_rate_bpm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect respiratory peaks (end of inhale) and troughs (end of exhale).

    The resp_signal is already bandpassed, so no re-filtering is applied.
    Uses scipy.signal.find_peaks with auto-prominence (10% of signal range)
    and minimum peak distance derived from the maximum respiratory rate.

    Returns (peak_indices, trough_indices).
    """
    min_dist_samples = max(int(fs * 60.0 / max_resp_rate_bpm), 1)

    sig_range = np.ptp(resp_signal)
    prominence = sig_range * 0.1 if sig_range > 0 else None

    peaks, _ = find_peaks(resp_signal, distance=min_dist_samples, prominence=prominence)
    troughs, _ = find_peaks(-resp_signal, distance=min_dist_samples, prominence=prominence)

    return peaks, troughs


# ---------------------------------------------------------------------------
# Public: extraction dispatcher
# ---------------------------------------------------------------------------

def extract_ppg_respiration(
    ppg_signal: np.ndarray,
    fs: float,
    method: str = "riiv",
    config_path: str = "config_tinnitus.yaml",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract respiratory proxy signal from PPG and detect resp peaks/troughs.

    Parameters
    ----------
    ppg_signal : 1D float64 PPG waveform
    fs : sampling rate in Hz (125 for BIDMC, 64 for WESAD)
    method : "riiv" | "rifv" | "baseline"
    config_path : path to config YAML

    Returns
    -------
    (resp_signal, peak_indices, trough_indices)
      resp_signal   : 1D float64, same length as input — the respiratory proxy
      peak_indices  : int64 — resp peaks (end of inhale)
      trough_indices: int64 — resp troughs (end of exhale)
    """
    valid_methods = ("riiv", "rifv", "baseline")
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got {method!r}")

    try:
        cfg = _load_config(config_path).get("ppg_resp", {})
    except Exception:
        cfg = {}

    low = cfg.get("bandpass_low_hz", 0.1)
    high = cfg.get("bandpass_high_hz", 0.5)
    order = cfg.get("bandpass_order", 4)
    max_resp_rate = cfg.get("max_resp_rate_bpm", 30.0)

    # Denoise PPG to clean pulse morphology before peak detection
    try:
        clean_ppg = denoise_ppg(ppg_signal, fs, config_path=config_path)
    except Exception:
        clean_ppg = ppg_signal

    peak_indices = get_ppg_peak_indices(clean_ppg, fs)

    if len(peak_indices) < 3:
        empty = np.array([], dtype=np.int64)
        return np.zeros(len(ppg_signal), dtype=np.float64), empty, empty

    if method == "riiv":
        resp_signal = _riiv_respiratory_signal(clean_ppg, peak_indices, fs, low, high, order)
    elif method == "rifv":
        resp_signal = _rifv_respiratory_signal(peak_indices, fs, low, high, order, len(ppg_signal))
    else:  # baseline
        resp_signal = _baseline_respiratory_signal(ppg_signal, fs, low, high, order)

    resp_peaks, resp_troughs = _find_resp_peaks(resp_signal, fs, max_resp_rate)

    return resp_signal, resp_peaks, resp_troughs


# ---------------------------------------------------------------------------
# Public: frame-level exhalation labels
# ---------------------------------------------------------------------------

def generate_exhalation_labels_from_ppg(
    ppg_signal: np.ndarray,
    fs: float,
    frame_rate_hz: float = 5.0,
    config_path: Optional[str] = "config_tinnitus.yaml",
) -> dict:
    """Generate per-frame exhalation labels from PPG via respiratory modulation.

    Parameters
    ----------
    ppg_signal : 1D PPG waveform (any dtype, coerced to float64)
    fs : sampling rate in Hz
    frame_rate_hz : output frame rate (default 5 Hz = 200ms frames)
    config_path : path to config YAML

    Returns
    -------
    dict with keys matching edr.generate_exhalation_labels():
        labels        (n_frames,) float32 — 1=exhale, 0=inhale, NaN=unknown
        quality       (n_frames,) float32 — per-frame confidence [0, 1]
        n_resp_cycles int
        resp_rate_bpm float
    """
    ppg_signal = np.asarray(ppg_signal, dtype=np.float64).ravel()
    n_samples = len(ppg_signal)
    frame_size = int(fs / frame_rate_hz)
    n_frames = n_samples // frame_size

    if config_path is None:
        config_path = "config_tinnitus.yaml"

    try:
        cfg = _load_config(config_path).get("ppg_resp", {})
    except Exception:
        cfg = {}

    method = cfg.get("method", "riiv")
    min_resp_rate = cfg.get("min_resp_rate_bpm", 6.0)
    max_resp_rate = cfg.get("max_resp_rate_bpm", 30.0)
    min_cycles = cfg.get("min_resp_cycles", 2)

    if n_samples < int(10 * fs):
        logger.warning("Signal too short (%.1fs) for PPG resp, returning NaN", n_samples / fs)
        return _empty_result(n_frames)

    try:
        resp_signal, resp_peaks, resp_troughs = extract_ppg_respiration(
            ppg_signal, fs, method=method, config_path=config_path
        )
    except Exception as e:
        logger.warning("PPG resp extraction failed: %s", e)
        return _empty_result(n_frames)

    n_peaks = len(resp_peaks)
    if n_peaks < min_cycles:
        logger.warning(
            "Only %d resp cycles found (need %d), returning NaN", n_peaks, min_cycles
        )
        return _empty_result(n_frames)

    if n_peaks >= 2:
        resp_rate = 60.0 / np.mean(np.diff(resp_peaks) / fs)
    else:
        resp_rate = np.nan

    if not np.isnan(resp_rate) and (resp_rate < min_resp_rate or resp_rate > max_resp_rate):
        logger.warning(
            "Resp rate %.1f bpm outside [%.0f, %.0f], returning NaN",
            resp_rate, min_resp_rate, max_resp_rate,
        )
        return _empty_result(n_frames)

    # PPG-derived resp: amplitude peaks at end of EXPIRATION (pulsus paradoxus —
    # inspiration decreases left-heart output → smaller pulse amplitude).
    # Swap convention: RIIV/RIFV/baseline troughs = end of inhale (peaks in
    # _build_resp_phase_array), peaks = end of exhale (troughs in that function).
    phase_samples, quality_samples = _build_resp_phase_array(resp_troughs, resp_peaks, n_samples)
    labels, quality = _downsample_to_frames(phase_samples, quality_samples, frame_size)

    return {
        "labels": labels,
        "quality": quality,
        "n_resp_cycles": int(n_peaks),
        "resp_rate_bpm": float(resp_rate) if not np.isnan(resp_rate) else np.nan,
    }
