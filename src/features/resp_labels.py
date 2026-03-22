"""
Generate exhalation labels from a reference respiratory signal (thermistor / flow_rate).

Replaces EDR-based labels for CVES records where hardware respiratory signals exist.
Return interface matches edr.generate_exhalation_labels() for drop-in use.
"""

import logging
from typing import Tuple

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

from src.data.dataset_parsers import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_resp_config(config_path: str) -> dict:
    config = load_config(config_path)
    return config.get("resp_labels", {})


def _empty_result(n_frames: int) -> dict:
    return {
        "labels": np.full(n_frames, np.nan, dtype=np.float32),
        "quality": np.zeros(n_frames, dtype=np.float32),
        "n_resp_cycles": 0,
        "resp_rate_bpm": np.nan,
    }


def _bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Butterworth bandpass to isolate respiratory band."""
    nyq = fs / 2.0
    # Clamp to valid range
    low = max(low_hz / nyq, 1e-6)
    high = min(high_hz / nyq, 0.999)
    if low >= high:
        return signal
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, signal).astype(np.float64)


def _detect_resp_peaks(
    signal: np.ndarray,
    fs: float,
    channel_name: str,
    cfg: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect respiratory peaks (end of inhale) and troughs (end of exhale).

    For thermst: signal is inverted so that peaks = end of inhale (same
    convention as flow_rate where positive flow = inhale).

    Returns (peak_indices, trough_indices) as int arrays.
    """
    # Invert thermst: its peaks are warm exhaled air = end of exhale
    if cfg.get("thermst_invert", True) and channel_name == "thermst":
        signal = -signal

    # Bandpass filter
    low = cfg.get("bandpass_low_hz", 0.08)
    high = cfg.get("bandpass_high_hz", 0.6)
    order = cfg.get("bandpass_order", 4)
    filtered = _bandpass_filter(signal, fs, low, high, order)

    # Peak detection
    min_dist_sec = cfg.get("min_peak_distance_sec", 1.5)
    min_dist_samples = max(int(fs * min_dist_sec), 1)

    # Auto-prominence: 20% of signal range
    sig_range = np.ptp(filtered)
    prominence = sig_range * 0.1 if sig_range > 0 else None

    peaks, _ = find_peaks(filtered, distance=min_dist_samples, prominence=prominence)
    troughs, _ = find_peaks(-filtered, distance=min_dist_samples, prominence=prominence)

    return peaks, troughs


def _build_resp_phase_array(
    peaks: np.ndarray,
    troughs: np.ndarray,
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sample-level phase labels from peaks and troughs.

    Convention:
    - peaks   = end of inhale (max positive flow / min negative thermst)
    - troughs = end of exhale (min positive flow / max negative thermst)
    - peak → trough = exhale phase (label=1)
    - trough → peak = inhale phase (label=0)
    - Before first landmark / after last landmark = NaN
    """
    phase = np.full(n_samples, np.nan, dtype=np.float32)
    quality = np.zeros(n_samples, dtype=np.float32)

    if len(peaks) == 0 and len(troughs) == 0:
        return phase, quality

    # Merge landmarks into sorted timeline with type tags
    landmarks = []
    for p in peaks:
        landmarks.append((int(p), "peak"))
    for t in troughs:
        landmarks.append((int(t), "trough"))
    landmarks.sort(key=lambda x: x[0])

    # Label between consecutive landmarks
    for i in range(len(landmarks) - 1):
        start_idx, start_type = landmarks[i]
        end_idx, _ = landmarks[i + 1]

        if start_type == "peak":
            # peak → next landmark = exhale
            label_val = 1.0
        else:
            # trough → next landmark = inhale
            label_val = 0.0

        s = max(0, start_idx)
        e = min(n_samples, end_idx)
        phase[s:e] = label_val
        quality[s:e] = 1.0

    return phase, quality


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
# Public API
# ---------------------------------------------------------------------------

def generate_exhalation_labels_from_reference(
    resp_signal: np.ndarray,
    fs: float,
    channel_name: str = "flow_rate",
    frame_rate_hz: float = 5.0,
    config_path: str = "config_stroke.yaml",
) -> dict:
    """Generate exhalation labels from a reference respiratory signal.

    Returns dict matching edr.generate_exhalation_labels() interface:
        labels        (n_frames,) float32 — 1=exhale, 0=inhale, NaN=unknown
        quality       (n_frames,) float32 — per-frame confidence [0, 1]
        n_resp_cycles int
        resp_rate_bpm float
    """
    cfg = _load_resp_config(config_path)

    n_samples = len(resp_signal)
    frame_size = int(fs / frame_rate_hz)
    n_frames = n_samples // frame_size

    # Minimum duration: 10s
    if n_samples < 10 * fs:
        logger.warning("Resp signal too short (%.1fs) for labeling", n_samples / fs)
        return _empty_result(n_frames)

    # Detect peaks and troughs
    try:
        peaks, troughs = _detect_resp_peaks(resp_signal, fs, channel_name, cfg)
    except Exception as e:
        logger.warning("Resp peak detection failed: %s", e)
        return _empty_result(n_frames)

    min_cycles = cfg.get("min_resp_cycles", 2)
    n_peaks = len(peaks)
    if n_peaks < min_cycles:
        logger.warning("Only %d resp cycles found (need %d)", n_peaks, min_cycles)
        return _empty_result(n_frames)

    # Respiratory rate
    if n_peaks >= 2:
        peak_intervals = np.diff(peaks) / fs
        resp_rate = 60.0 / np.mean(peak_intervals)
    else:
        resp_rate = np.nan

    min_rate = cfg.get("min_resp_rate_bpm", 6.0)
    max_rate = cfg.get("max_resp_rate_bpm", 30.0)
    if not np.isnan(resp_rate) and (resp_rate < min_rate or resp_rate > max_rate):
        logger.warning(
            "Resp rate %.1f bpm outside [%.0f, %.0f]", resp_rate, min_rate, max_rate,
        )
        return _empty_result(n_frames)

    # Build sample-level phase array
    phase_samples, quality_samples = _build_resp_phase_array(peaks, troughs, n_samples)

    # Downsample to frame-level
    labels, quality = _downsample_to_frames(phase_samples, quality_samples, frame_size)

    return {
        "labels": labels,
        "quality": quality,
        "n_resp_cycles": int(n_peaks),
        "resp_rate_bpm": float(resp_rate) if not np.isnan(resp_rate) else np.nan,
    }
