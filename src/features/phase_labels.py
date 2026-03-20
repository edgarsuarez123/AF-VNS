"""
Diastolic phase label generation from ECG.

Detects R-peaks and T-wave ends to label each 200ms frame (5Hz) as
systole (0) or diastole (1). Used as ground truth for the Phase Detector CNN.

Grant reference: Aim 2 — >85% diastolic phase classification accuracy.
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def _load_config(config_path: str = "config_stroke.yaml") -> dict:
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
        "n_beats": 0,
        "fallback_fraction": 1.0,
        "mean_hr_bpm": np.nan,
    }


# ---------------------------------------------------------------------------
# Public: R-peak detection
# ---------------------------------------------------------------------------

def get_rpeak_indices(signal: np.ndarray, fs: float) -> np.ndarray:
    """Return R-peak sample indices as int64 array.

    Thin wrapper around nk.ecg_peaks(). Does NOT return R-R intervals
    (use peak_detector.get_rr_intervals for that). Kept separate to avoid
    modifying the AF pipeline's peak_detector.py.
    """
    try:
        import neurokit2 as nk
    except ImportError:
        raise ImportError("neurokit2 is required for get_rpeak_indices")

    signal = np.asarray(signal, dtype=np.float64).ravel()
    _, info = nk.ecg_peaks(signal, sampling_rate=fs)
    peaks = np.asarray(info["ECG_R_Peaks"], dtype=np.int64)
    return peaks


# ---------------------------------------------------------------------------
# Public: T-wave offset detection with fallback
# ---------------------------------------------------------------------------

def get_twave_offsets(
    signal: np.ndarray,
    rpeaks: np.ndarray,
    fs: float,
    fallback_fraction: float = 0.40,
    method: str = "dwt",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return T-wave end sample indices and a boolean fallback mask.

    Parameters
    ----------
    signal : 1D float64 ECG
    rpeaks : int64 array of R-peak sample indices
    fs : sampling rate
    fallback_fraction : fraction of RR interval used when delineation fails
    method : neurokit2 delineation method ("dwt" recommended)

    Returns
    -------
    t_offsets : int64 array, same length as rpeaks
    fallback_mask : bool array, True where fallback was used
    """
    try:
        import neurokit2 as nk
    except ImportError:
        raise ImportError("neurokit2 is required for get_twave_offsets")

    n_beats = len(rpeaks)
    t_offsets = np.empty(n_beats, dtype=np.float64)
    t_offsets[:] = np.nan
    fallback_mask = np.ones(n_beats, dtype=bool)

    # Attempt neurokit2 delineation
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            _, waves = nk.ecg_delineate(
                signal, rpeaks.tolist(), sampling_rate=fs, method=method
            )
        raw_offsets = waves.get("ECG_T_Offsets", [])
        for i, val in enumerate(raw_offsets):
            if i < n_beats and val is not None and not np.isnan(val):
                t_offsets[i] = int(val)
                fallback_mask[i] = False
    except Exception as e:
        logger.warning("ecg_delineate failed, using RR fallback for all beats: %s", e)

    # Fill NaN entries with fallback: R + fallback_fraction * RR
    for i in range(n_beats):
        if np.isnan(t_offsets[i]):
            if i < n_beats - 1:
                rr = rpeaks[i + 1] - rpeaks[i]
            elif i > 0:
                rr = rpeaks[i] - rpeaks[i - 1]
            else:
                rr = int(fs)  # 1 second default if single beat
            t_offsets[i] = rpeaks[i] + int(fallback_fraction * rr)

    # Clamp: t_offset must be in (rpeak[i], rpeak[i+1]) to prevent overlap
    t_offsets = t_offsets.astype(np.int64)
    for i in range(n_beats):
        # Must be after its own R-peak
        t_offsets[i] = max(t_offsets[i], rpeaks[i] + 1)
        # Must be before next R-peak (if exists)
        if i < n_beats - 1:
            t_offsets[i] = min(t_offsets[i], rpeaks[i + 1] - 1)
        # Must be within signal bounds
        t_offsets[i] = min(t_offsets[i], len(signal) - 1)

    return t_offsets, fallback_mask


# ---------------------------------------------------------------------------
# Private: sample-level phase array
# ---------------------------------------------------------------------------

def _build_sample_phase_array(
    rpeaks: np.ndarray,
    t_offsets: np.ndarray,
    fallback_mask: np.ndarray,
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map R-peaks + T-wave ends to per-sample phase and quality arrays.

    Returns
    -------
    phase : float32 (n_samples,) — 0=systole, 1=diastole, NaN=unknown
    quality : float32 (n_samples,) — 1.0=detected, 0.5=fallback, 0.0=unknown
    """
    phase = np.full(n_samples, np.nan, dtype=np.float32)
    quality = np.zeros(n_samples, dtype=np.float32)
    n_beats = len(rpeaks)

    for i in range(n_beats):
        q = 0.5 if fallback_mask[i] else 1.0

        # Systole: R-peak to T-wave end
        sys_start = rpeaks[i]
        sys_end = t_offsets[i]
        phase[sys_start:sys_end] = 0.0
        quality[sys_start:sys_end] = q

        # Diastole: T-wave end to next R-peak (or end of signal for last beat)
        dia_start = t_offsets[i]
        if i < n_beats - 1:
            dia_end = rpeaks[i + 1]
        else:
            dia_end = n_samples
        phase[dia_start:dia_end] = 1.0
        quality[dia_start:dia_end] = q

    return phase, quality


# ---------------------------------------------------------------------------
# Private: downsample to frames
# ---------------------------------------------------------------------------

def _downsample_to_frames(
    phase_samples: np.ndarray,
    quality_samples: np.ndarray,
    frame_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Majority-vote reduction from sample-level to frame-level labels.

    Parameters
    ----------
    phase_samples : float32 (n_samples,) — 0/1/NaN
    quality_samples : float32 (n_samples,)
    frame_size : samples per frame (e.g., 50 for 250Hz/5Hz)

    Returns
    -------
    labels : float32 (n_frames,) — 0=systole, 1=diastole, NaN=unknown
    quality : float32 (n_frames,)
    """
    n_frames = len(phase_samples) // frame_size
    if n_frames == 0:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        )

    # Trim to exact multiple of frame_size
    trimmed = phase_samples[: n_frames * frame_size].reshape(n_frames, frame_size)
    q_trimmed = quality_samples[: n_frames * frame_size].reshape(n_frames, frame_size)

    labels = np.empty(n_frames, dtype=np.float32)
    quality = np.empty(n_frames, dtype=np.float32)

    for f in range(n_frames):
        frame = trimmed[f]
        nan_frac = np.isnan(frame).mean()
        if nan_frac > 0.5:
            labels[f] = np.nan
        else:
            # Majority vote: round(nanmean) gives 1 if diastole dominates
            labels[f] = np.round(np.nanmean(frame))
        quality[f] = q_trimmed[f].mean()

    return labels, quality


# ---------------------------------------------------------------------------
# Public: main entry point
# ---------------------------------------------------------------------------

def generate_phase_labels(
    signal: np.ndarray,
    fs: float,
    frame_rate_hz: float = 5.0,
    fallback_fraction: float = 0.40,
    config_path: str = "config_stroke.yaml",
) -> dict:
    """Generate per-frame diastolic phase labels from ECG.

    Parameters
    ----------
    signal : 1D ECG waveform (float64, any length)
    fs : sampling rate in Hz
    frame_rate_hz : output frame rate (default 5 Hz = 200ms)
    fallback_fraction : T-end fallback as fraction of RR interval
    config_path : path to config YAML

    Returns
    -------
    dict with keys:
        labels : (n_frames,) float32 — 1=diastole, 0=systole, NaN=unknown
        quality : (n_frames,) float32 — per-frame confidence [0, 1]
        n_beats : int — R-peaks detected
        fallback_fraction : float — fraction of T-offsets that used RR fallback
        mean_hr_bpm : float — mean heart rate
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    n_samples = len(signal)
    frame_size = int(fs / frame_rate_hz)
    n_frames = n_samples // frame_size

    # Load config for bounds
    try:
        cfg = _load_config(config_path).get("phase_detection", {})
    except Exception:
        cfg = {}
    min_beats = cfg.get("min_beats", 2)
    min_hr = cfg.get("min_hr_bpm", 40.0)
    max_hr = cfg.get("max_hr_bpm", 200.0)
    method = cfg.get("delineate_method", "dwt")
    do_denoise = cfg.get("denoise_before_delineate", True)

    # Validate minimum duration (need at least 2s for meaningful detection)
    if n_samples < 2 * fs:
        logger.warning("Signal too short (%.1fs), returning NaN labels", n_samples / fs)
        return _empty_result(n_frames)

    # Denoise for better T-wave detection
    if do_denoise:
        try:
            from src.features.wavelet_filter import denoise
            processed = denoise(signal, fs)
        except Exception as e:
            logger.warning("Denoise failed, using raw signal: %s", e)
            processed = signal
    else:
        processed = signal

    # Detect R-peaks
    try:
        rpeaks = get_rpeak_indices(processed, fs)
    except Exception as e:
        logger.warning("R-peak detection failed: %s", e)
        return _empty_result(n_frames)

    if len(rpeaks) < min_beats:
        logger.warning("Only %d R-peaks found (need %d), returning NaN", len(rpeaks), min_beats)
        return _empty_result(n_frames)

    # Heart rate sanity check
    rr_sec = np.diff(rpeaks) / fs
    mean_hr = 60.0 / np.mean(rr_sec)
    if mean_hr < min_hr or mean_hr > max_hr:
        logger.warning("Mean HR %.1f bpm outside [%.0f, %.0f], returning NaN", mean_hr, min_hr, max_hr)
        return _empty_result(n_frames)

    # Detect T-wave offsets
    t_offsets, fallback_mask = get_twave_offsets(
        processed, rpeaks, fs,
        fallback_fraction=fallback_fraction,
        method=method,
    )

    # Build sample-level phase array
    phase_samples, quality_samples = _build_sample_phase_array(
        rpeaks, t_offsets, fallback_mask, n_samples
    )

    # Downsample to frame-level labels
    labels, quality = _downsample_to_frames(phase_samples, quality_samples, frame_size)

    fb_frac = float(fallback_mask.sum()) / len(fallback_mask) if len(fallback_mask) > 0 else 1.0

    return {
        "labels": labels,
        "quality": quality,
        "n_beats": int(len(rpeaks)),
        "fallback_fraction": fb_frac,
        "mean_hr_bpm": float(mean_hr),
    }
