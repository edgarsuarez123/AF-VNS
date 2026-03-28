"""
PPG diastolic phase label generation for the tinnitus aVNS pipeline.

Detects systolic peaks and dicrotic notches from PPG to label each 200ms
frame (5 Hz) as systole (0) or diastole (1). Mirrors the interface of
phase_labels.generate_phase_labels() so the same PhaseDetector CNN and
precompute cache infrastructure can be reused.

Diastole detection method:
  - Systolic peaks detected via neurokit2 ppg_findpeaks()
  - Dicrotic notch: local minimum in descending limb of each pulse,
    searched after notch_search_start_fraction × beat_interval from peak
  - Diastole window: notch sample → (next_peak - notch_guard_samples)
  - Systole window: everything else

Reference: SBIR spec requires <50ms timing accuracy for diastole detection.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _load_config(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def _empty_result(n_frames: int) -> dict:
    return {
        "labels": np.full(n_frames, np.nan, dtype=np.float32),
        "quality": np.zeros(n_frames, dtype=np.float32),
        "n_beats": 0,
        "mean_hr_bpm": np.nan,
    }


def get_ppg_peak_indices(signal: np.ndarray, fs: float) -> np.ndarray:
    """Return systolic peak sample indices from PPG waveform.

    Uses neurokit2.ppg_findpeaks() — the same backend as peak_detector.py
    but returns raw indices rather than RR intervals.
    """
    try:
        import neurokit2 as nk
    except ImportError:
        raise ImportError("neurokit2 is required for get_ppg_peak_indices")

    signal = np.asarray(signal, dtype=np.float64).ravel()
    peaks_dict = nk.ppg_findpeaks(signal, sampling_rate=fs)
    return np.asarray(peaks_dict["PPG_Peaks"], dtype=np.int64)


def _detect_dicrotic_notch(
    signal: np.ndarray,
    peak_idx: int,
    next_peak_idx: int,
    notch_start_fraction: float = 0.30,
) -> int:
    """Find dicrotic notch as local minimum in descending pulse limb.

    Searches between (peak + notch_start_fraction × beat_interval) and
    (next_peak - notch_start_fraction × beat_interval) to avoid the
    systolic upstroke of the next beat.

    Returns the sample index of the notch, or the midpoint if no local
    minimum is found (fallback).
    """
    beat_interval = next_peak_idx - peak_idx
    search_start = peak_idx + int(notch_start_fraction * beat_interval)
    search_end = next_peak_idx - int(notch_start_fraction * beat_interval)

    if search_start >= search_end:
        # Beat too short — fall back to midpoint
        return (peak_idx + next_peak_idx) // 2

    segment = signal[search_start:search_end]
    local_min_offset = int(np.argmin(segment))
    return search_start + local_min_offset


def _build_ppg_phase_samples(
    signal: np.ndarray,
    peaks: np.ndarray,
    fs: float,
    notch_start_fraction: float,
    notch_guard_samples: int,
) -> tuple:
    """Build sample-level diastole (1) / systole (0) / NaN array from PPG peaks.

    Returns
    -------
    phase_samples : float32 (n_samples,) — 1=diastole, 0=systole, NaN=unknown
    quality_samples : float32 (n_samples,) — 1.0 for labelled samples, 0 elsewhere
    """
    n = len(signal)
    phase = np.full(n, np.nan, dtype=np.float32)
    quality = np.zeros(n, dtype=np.float32)

    for i in range(len(peaks) - 1):
        p_curr = peaks[i]
        p_next = peaks[i + 1]

        # Systole: from previous beat end to dicrotic notch
        notch = _detect_dicrotic_notch(signal, p_curr, p_next, notch_start_fraction)

        # Diastole end: guard samples before next systolic peak
        dia_end = max(notch, p_next - notch_guard_samples)

        # Systole region: current peak down to notch
        sys_start = p_curr
        sys_end = notch
        if sys_start < sys_end <= n:
            phase[sys_start:sys_end] = 0.0
            quality[sys_start:sys_end] = 1.0

        # Diastole region: notch to dia_end
        if notch < dia_end <= n:
            phase[notch:dia_end] = 1.0
            quality[notch:dia_end] = 1.0

    return phase, quality


def _reduce_to_frames(
    phase_samples: np.ndarray,
    quality_samples: np.ndarray,
    frame_size: int,
) -> tuple:
    """Majority-vote reduction from sample-level to frame-level labels.

    Matches phase_labels._reduce_to_frames() interface.
    """
    n_frames = len(phase_samples) // frame_size
    if n_frames == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

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


def generate_ppg_phase_labels(
    signal: np.ndarray,
    fs: float,
    frame_rate_hz: float = 5.0,
    config_path: Optional[str] = None,
    notch_start_fraction: float = 0.30,
    notch_guard_ms: float = 100.0,
    min_beats: int = 2,
    min_hr_bpm: float = 40.0,
    max_hr_bpm: float = 200.0,
) -> dict:
    """Generate per-frame diastolic phase labels from PPG waveform.

    Parameters
    ----------
    signal : 1D PPG waveform (float64)
    fs : sampling rate in Hz
    frame_rate_hz : output frame rate (default 5 Hz = 200ms frames)
    config_path : if provided, load params from config_tinnitus.yaml
    notch_start_fraction : search for dicrotic notch after this fraction of beat interval
    notch_guard_ms : reserve this many ms before next systolic peak as diastole end
    min_beats : minimum peaks required; else return all-NaN
    min_hr_bpm / max_hr_bpm : HR range validity check

    Returns
    -------
    dict with keys:
        labels : (n_frames,) float32 — 1=diastole, 0=systole, NaN=unknown
        quality : (n_frames,) float32 — per-frame confidence [0, 1]
        n_beats : int — systolic peaks detected
        mean_hr_bpm : float — mean heart rate
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    frame_size = int(fs / frame_rate_hz)
    n_frames = len(signal) // frame_size

    if config_path is not None:
        cfg = _load_config(config_path).get("ppg_diastole", {})
        notch_start_fraction = float(cfg.get("notch_search_start_fraction", notch_start_fraction))
        notch_guard_ms = float(cfg.get("notch_guard_ms", notch_guard_ms))
        min_beats = int(cfg.get("min_beats", min_beats))
        min_hr_bpm = float(cfg.get("min_hr_bpm", min_hr_bpm))
        max_hr_bpm = float(cfg.get("max_hr_bpm", max_hr_bpm))
        frame_rate_hz = float(cfg.get("frame_rate_hz", frame_rate_hz))
        frame_size = int(fs / frame_rate_hz)
        n_frames = len(signal) // frame_size

    if n_frames == 0:
        return _empty_result(1)

    notch_guard_samples = int(notch_guard_ms / 1000.0 * fs)

    # Detect systolic peaks
    try:
        peaks = get_ppg_peak_indices(signal, fs)
    except Exception as exc:
        logger.warning("PPG peak detection failed: %s", exc)
        return _empty_result(n_frames)

    if len(peaks) < min_beats:
        logger.debug("Too few PPG peaks (%d < %d)", len(peaks), min_beats)
        return _empty_result(n_frames)

    # HR validity check
    rr_intervals = np.diff(peaks) / fs  # seconds
    mean_hr = 60.0 / np.mean(rr_intervals) if len(rr_intervals) > 0 else np.nan
    if np.isnan(mean_hr) or mean_hr < min_hr_bpm or mean_hr > max_hr_bpm:
        logger.debug("PPG HR %.1f bpm outside valid range [%.0f, %.0f]", mean_hr, min_hr_bpm, max_hr_bpm)
        return _empty_result(n_frames)

    # Build sample-level labels
    phase_samples, quality_samples = _build_ppg_phase_samples(
        signal, peaks, fs, notch_start_fraction, notch_guard_samples
    )

    # Reduce to frame-level
    labels, quality = _reduce_to_frames(phase_samples, quality_samples, frame_size)

    return {
        "labels": labels,
        "quality": quality,
        "n_beats": int(len(peaks)),
        "mean_hr_bpm": float(mean_hr),
    }
