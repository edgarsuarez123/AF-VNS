"""
ECG-Derived Respiration (EDR) — exhalation phase label generation.

Extracts a respiratory proxy from ECG heart-rate variability, detects
inhale/exhale phases, and labels each 200ms frame (5Hz) as exhale (1)
or inhale (0). Used as ground truth for the Phase Detector CNN.

Grant reference: Aim 2 — >85% exhalation phase classification accuracy.
EDR accuracy ceiling vs real respirometry: 80-90%.
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
        "n_resp_cycles": 0,
        "resp_rate_bpm": np.nan,
    }


def _downsample_to_frames(
    phase_samples: np.ndarray,
    quality_samples: np.ndarray,
    frame_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Majority-vote reduction from sample-level to frame-level labels."""
    n_frames = len(phase_samples) // frame_size
    if n_frames == 0:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        )

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
            labels[f] = np.round(np.nanmean(frame))
        quality[f] = q_trimmed[f].mean()

    return labels, quality


# ---------------------------------------------------------------------------
# Public: EDR extraction
# ---------------------------------------------------------------------------

def extract_edr(
    signal: np.ndarray,
    fs: float,
    method: str = "vangent2019",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract ECG-Derived Respiration signal and detect respiratory peaks/troughs.

    Parameters
    ----------
    signal : 1D float64 ECG (should be denoised for best results)
    fs : sampling rate in Hz
    method : nk.ecg_rsp method ("vangent2019", "soni2019", "charlton2016", "sarkar2015")

    Returns
    -------
    edr_signal : 1D float64 respiratory proxy (same length as signal)
    peak_indices : int64 array of respiratory peak sample indices
    trough_indices : int64 array of respiratory trough sample indices
    """
    try:
        import neurokit2 as nk
    except ImportError:
        raise ImportError("neurokit2 is required for extract_edr")

    signal = np.asarray(signal, dtype=np.float64).ravel()

    # R-peak detection
    from src.features.phase_labels import get_rpeak_indices
    rpeaks = get_rpeak_indices(signal, fs)

    if len(rpeaks) < 2:
        return (
            np.zeros(len(signal), dtype=np.float64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )

    # Heart rate signal (continuous, same length as ECG)
    info_dict = {"ECG_R_Peaks": rpeaks.tolist()}
    ecg_rate = nk.signal_rate(
        info_dict, sampling_rate=fs, desired_length=len(signal)
    )

    # ECG-Derived Respiration via bandpass-filtered HR variability
    edr_signal = nk.ecg_rsp(ecg_rate, sampling_rate=fs, method=method)
    edr_signal = np.asarray(edr_signal, dtype=np.float64)

    # Detect respiratory peaks and troughs
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        rsp_info = nk.rsp_findpeaks(edr_signal, sampling_rate=fs)

    peak_indices = np.asarray(rsp_info.get("RSP_Peaks", []), dtype=np.int64)
    trough_indices = np.asarray(rsp_info.get("RSP_Troughs", []), dtype=np.int64)

    return edr_signal, peak_indices, trough_indices


# ---------------------------------------------------------------------------
# Public: main entry point
# ---------------------------------------------------------------------------

def generate_exhalation_labels(
    signal: np.ndarray,
    fs: float,
    frame_rate_hz: float = 5.0,
    config_path: str = "config_stroke.yaml",
) -> dict:
    """Generate per-frame exhalation phase labels from ECG via EDR.

    Parameters
    ----------
    signal : 1D ECG waveform (float64, any length)
    fs : sampling rate in Hz
    frame_rate_hz : output frame rate (default 5 Hz = 200ms)
    config_path : path to config YAML

    Returns
    -------
    dict with keys:
        labels : (n_frames,) float32 — 1=exhale, 0=inhale, NaN=unknown
        quality : (n_frames,) float32 — per-frame confidence [0, 1]
        n_resp_cycles : int — respiratory cycles detected
        resp_rate_bpm : float — estimated breaths per minute
    """
    try:
        import neurokit2 as nk
    except ImportError:
        raise ImportError("neurokit2 is required for generate_exhalation_labels")

    signal = np.asarray(signal, dtype=np.float64).ravel()
    n_samples = len(signal)
    frame_size = int(fs / frame_rate_hz)
    n_frames = n_samples // frame_size

    # Load config
    try:
        cfg = _load_config(config_path).get("edr", {})
    except Exception:
        cfg = {}
    method = cfg.get("method", "vangent2019")
    min_resp_rate = cfg.get("min_resp_rate_bpm", 6.0)
    max_resp_rate = cfg.get("max_resp_rate_bpm", 30.0)
    min_cycles = cfg.get("min_resp_cycles", 2)
    do_denoise = cfg.get("denoise_before_edr", True)

    # Validate minimum duration (10s for at least 2 respiratory cycles)
    if n_samples < 10 * fs:
        logger.warning("Signal too short (%.1fs) for EDR, returning NaN", n_samples / fs)
        return _empty_result(n_frames)

    # Denoise
    if do_denoise:
        try:
            from src.features.wavelet_filter import denoise
            processed = denoise(signal, fs)
        except Exception as e:
            logger.warning("Denoise failed, using raw signal: %s", e)
            processed = signal
    else:
        processed = signal

    # Extract EDR + respiratory peaks/troughs
    try:
        edr_signal, peak_indices, trough_indices = extract_edr(processed, fs, method)
    except Exception as e:
        logger.warning("EDR extraction failed: %s", e)
        return _empty_result(n_frames)

    n_peaks = len(peak_indices)
    if n_peaks < min_cycles:
        logger.warning(
            "Only %d respiratory cycles found (need %d), returning NaN",
            n_peaks, min_cycles,
        )
        return _empty_result(n_frames)

    # Respiratory rate sanity check
    if n_peaks >= 2:
        rr_intervals = np.diff(peak_indices) / fs
        resp_rate = 60.0 / np.mean(rr_intervals)
    else:
        resp_rate = np.nan

    if not np.isnan(resp_rate) and (resp_rate < min_resp_rate or resp_rate > max_resp_rate):
        logger.warning(
            "Resp rate %.1f bpm outside [%.0f, %.0f], returning NaN",
            resp_rate, min_resp_rate, max_resp_rate,
        )
        return _empty_result(n_frames)

    # Sample-level phase via neurokit2
    try:
        rsp_info = {"RSP_Peaks": peak_indices.tolist(), "RSP_Troughs": trough_indices.tolist()}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            phase_df = nk.rsp_phase(rsp_info, desired_length=n_samples)

        # RSP_Phase: 1=inspiration(inhale), 0=expiration(exhale)
        rsp_phase = phase_df["RSP_Phase"].values.astype(np.float64)

        # Invert: our convention is exhale=1, inhale=0
        phase_samples = np.where(np.isnan(rsp_phase), np.nan, 1.0 - rsp_phase).astype(np.float32)

        # Quality: 1.0 where phase is known, 0.0 where NaN
        quality_samples = np.where(np.isnan(rsp_phase), 0.0, 1.0).astype(np.float32)

    except Exception as e:
        logger.warning("rsp_phase failed: %s", e)
        return _empty_result(n_frames)

    # Downsample to frame-level labels
    labels, quality = _downsample_to_frames(phase_samples, quality_samples, frame_size)

    return {
        "labels": labels,
        "quality": quality,
        "n_resp_cycles": int(n_peaks),
        "resp_rate_bpm": float(resp_rate) if not np.isnan(resp_rate) else np.nan,
    }
