"""
EDA Feature Extraction — tonic/phasic decomposition + arousal-in-band gate.

Implements per-subject baseline calibration for the tinnitus aVNS tri-fold
trigger: stimulation fires only when the patient is in a calm-but-awake
arousal band (not drowsy, not acutely stressed).

Pipeline:
  1. EDA signal cleaned via neurokit2
  2. Decomposed into tonic (SCL) + phasic (SCR) components — cvxEDA by default
  3. Tonic SCL calibrated against a 120s baseline period → (mean, std)
  4. Arousal gate: in-band = SCL within [mean - low*std, mean + high*std]
  5. Output: binary arousal_in_band at 1 Hz

WESAD validation target:
  - Baseline self-calibration: >80% frames in-band
  - Stress cross-calibration: >50% frames out-of-band

SBIR reference: aVNS Tinnitus.md — tri-fold gate section
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config + empty result
# ---------------------------------------------------------------------------

def _load_config(config_path: str = "config_tinnitus.yaml") -> dict:
    """Load the 'eda' section from the tinnitus config YAML."""
    if not os.path.isabs(config_path) and not os.path.isfile(config_path):
        root = Path(__file__).resolve().parents[2]
        config_path = str(root / config_path)
    with open(config_path) as f:
        return yaml.safe_load(f)


def _empty_result(n_frames: int) -> dict:
    """Return all-NaN result dict for unprocessable signals."""
    return {
        "arousal_in_band": np.full(n_frames, np.nan, dtype=np.float32),
        "tonic_scl": np.full(n_frames, np.nan, dtype=np.float32),
        "phasic_scr": np.full(n_frames, np.nan, dtype=np.float32),
        "scr_count": 0,
        "calibration_mean": np.nan,
        "calibration_std": np.nan,
        "low_threshold": np.nan,
        "high_threshold": np.nan,
        "in_band_fraction": np.nan,
    }


# ---------------------------------------------------------------------------
# EDA decomposition
# ---------------------------------------------------------------------------

def decompose_eda(
    signal: np.ndarray,
    fs: float,
    method: str = "cvxeda",
) -> dict:
    """Decompose EDA into tonic (SCL) and phasic (SCR) components.

    Uses neurokit2: eda_clean → eda_phasic. Falls back to 'highpass' if
    cvxEDA fails (requires cvxopt which may not be installed).

    Parameters
    ----------
    signal : 1D float64 EDA waveform in µS
    fs : sampling rate in Hz (4 Hz for WESAD Empatica E4)
    method : "cvxeda" (default) | "highpass" | "median"

    Returns
    -------
    dict with keys 'tonic' and 'phasic' — both (n_samples,) float64 arrays
    """
    import neurokit2 as nk

    signal = np.asarray(signal, dtype=np.float64).ravel()
    fs_int = int(round(fs))

    # Clean: artifact removal + lowpass
    try:
        cleaned = nk.eda_clean(signal, sampling_rate=fs_int)
    except Exception as e:
        logger.warning("eda_clean failed (%s), using raw signal", e)
        cleaned = signal.copy()

    # Decompose: cvxEDA with highpass fallback
    active_method = method
    decomposed = None

    try:
        decomposed = nk.eda_phasic(cleaned, sampling_rate=fs_int, method=active_method)
    except Exception as e:
        if active_method == "cvxeda":
            logger.warning(
                "cvxEDA decomposition failed (%s), falling back to highpass", e
            )
            active_method = "highpass"
            try:
                decomposed = nk.eda_phasic(cleaned, sampling_rate=fs_int, method=active_method)
            except Exception as e2:
                logger.warning("highpass EDA decomposition also failed: %s", e2)
        else:
            logger.warning("EDA decomposition failed: %s", e)

    if decomposed is None:
        # Last resort: trivial split (tonic = moving average, phasic = residual)
        from scipy.ndimage import uniform_filter1d
        window = max(1, fs_int * 10)
        tonic = uniform_filter1d(cleaned, size=window, mode="nearest")
        phasic = np.clip(cleaned - tonic, 0, None)
        return {"tonic": tonic.astype(np.float64), "phasic": phasic.astype(np.float64)}

    # neurokit2 returns a DataFrame with EDA_Tonic, EDA_Phasic
    tonic = decomposed["EDA_Tonic"].to_numpy(dtype=np.float64)
    phasic = decomposed["EDA_Phasic"].to_numpy(dtype=np.float64)

    return {"tonic": tonic, "phasic": phasic}


# ---------------------------------------------------------------------------
# SCR peak detection
# ---------------------------------------------------------------------------

def detect_scr_peaks(
    phasic: np.ndarray,
    fs: float,
    min_amplitude: float = 0.02,
) -> dict:
    """Detect skin conductance response (SCR) peaks in phasic EDA component.

    Parameters
    ----------
    phasic : 1D float64 phasic EDA component
    fs : sampling rate in Hz
    min_amplitude : minimum SCR peak amplitude in µS to count (default 0.02)

    Returns
    -------
    dict with:
        peak_indices  : int64 ndarray — sample indices of detected SCR peaks
        amplitudes    : float64 ndarray — peak amplitudes in µS
        count         : int — number of peaks passing amplitude threshold
    """
    import neurokit2 as nk

    phasic = np.asarray(phasic, dtype=np.float64).ravel()
    fs_int = int(round(fs))

    empty = {"peak_indices": np.array([], dtype=np.int64),
             "amplitudes": np.array([], dtype=np.float64),
             "count": 0}

    if len(phasic) < 2 or np.ptp(phasic) == 0:
        return empty

    try:
        peaks_info = nk.eda_findpeaks(phasic, sampling_rate=fs_int)
    except Exception as e:
        logger.warning("eda_findpeaks failed: %s", e)
        return empty

    # neurokit2 returns dict with 'SCR_Peaks' key containing peak indices
    peak_key = "SCR_Peaks"
    if not isinstance(peaks_info, dict) or peak_key not in peaks_info or peaks_info[peak_key] is None:
        return empty

    raw_indices = np.asarray(peaks_info[peak_key], dtype=np.int64)
    if len(raw_indices) == 0:
        return empty

    # Filter by minimum amplitude
    valid_mask = phasic[raw_indices] >= min_amplitude
    filtered_indices = raw_indices[valid_mask]
    filtered_amplitudes = phasic[filtered_indices]

    return {
        "peak_indices": filtered_indices,
        "amplitudes": filtered_amplitudes,
        "count": int(len(filtered_indices)),
    }


# ---------------------------------------------------------------------------
# Baseline calibration
# ---------------------------------------------------------------------------

def calibrate_baseline(
    tonic: np.ndarray,
    fs: float,
    calibration_sec: float = 120.0,
) -> Tuple[float, float]:
    """Estimate per-subject baseline SCL statistics from the first N seconds.

    Parameters
    ----------
    tonic : 1D float64 tonic EDA (SCL) component
    fs : sampling rate in Hz
    calibration_sec : duration (s) to use for calibration (clipped to signal length)

    Returns
    -------
    (mean, std) — float tuple; std floored at 1e-6 to prevent division by zero
    """
    tonic = np.asarray(tonic, dtype=np.float64).ravel()
    n_cal = min(len(tonic), int(calibration_sec * fs))
    if n_cal < 1:
        return float(np.nanmean(tonic)), 1e-6

    cal_segment = tonic[:n_cal]
    cal_mean = float(np.nanmean(cal_segment))
    cal_std = float(np.nanstd(cal_segment))
    cal_std = max(cal_std, 1e-6)  # floor to prevent zero-std division

    return cal_mean, cal_std


# ---------------------------------------------------------------------------
# Arousal-in-band computation
# ---------------------------------------------------------------------------

def compute_arousal_in_band(
    tonic: np.ndarray,
    fs: float,
    cal_mean: float,
    cal_std: float,
    low_sigma: float,
    high_sigma: float,
    frame_rate_hz: float = 1.0,
) -> np.ndarray:
    """Downsample tonic SCL to 1 Hz frames and apply arousal threshold.

    Arousal gate fires (1) when SCL is within:
        [cal_mean - low_sigma * cal_std, cal_mean + high_sigma * cal_std]

    Below the lower bound → drowsy (hold stim).
    Above the upper bound → hyperaroused (hold stim).

    Parameters
    ----------
    tonic : 1D float64 tonic EDA at input fs
    fs : sampling rate of tonic in Hz
    cal_mean : baseline SCL mean (µS)
    cal_std : baseline SCL std (µS), must be > 0
    low_sigma : lower bound sigma multiplier
    high_sigma : upper bound sigma multiplier
    frame_rate_hz : output frame rate in Hz (default 1.0)

    Returns
    -------
    (n_frames,) float32 binary array — 1=in-band, 0=out-of-band
    """
    tonic = np.asarray(tonic, dtype=np.float64).ravel()

    low_thresh = cal_mean - low_sigma * cal_std
    high_thresh = cal_mean + high_sigma * cal_std

    # Downsample: average across each 1s frame
    samples_per_frame = max(1, int(round(fs / frame_rate_hz)))
    n_frames = len(tonic) // samples_per_frame

    if n_frames == 0:
        return np.array([], dtype=np.float32)

    # Reshape and mean — trim tail to fit full frames
    trimmed = tonic[: n_frames * samples_per_frame]
    frame_means = trimmed.reshape(n_frames, samples_per_frame).mean(axis=1)

    in_band = ((frame_means >= low_thresh) & (frame_means <= high_thresh)).astype(np.float32)

    return in_band


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_eda_features(
    signal: np.ndarray,
    fs: float,
    calibration_signal: Optional[np.ndarray] = None,
    calibration_fs: Optional[float] = None,
    config_path: str = "config_tinnitus.yaml",
) -> dict:
    """Extract EDA arousal-in-band features for the tinnitus aVNS tri-fold gate.

    Parameters
    ----------
    signal : 1D float64 EDA waveform (µS) — the recording to classify
    fs : EDA sampling rate in Hz (4 Hz for WESAD Empatica E4)
    calibration_signal : optional separate baseline recording for cross-calibration.
        If None, self-calibrate from the first `calibration_sec` of `signal`.
        Use for stress records (label=1): calibrate from paired baseline.
    calibration_fs : sampling rate of calibration_signal (defaults to fs if None)
    config_path : path to config YAML

    Returns
    -------
    dict:
        arousal_in_band   (n_frames,) float32 — 1=in-band, 0=out-of-band at 1 Hz
        tonic_scl         (n_frames,) float32 — mean tonic SCL per 1s frame
        phasic_scr        (n_frames,) float32 — max phasic SCR per 1s frame
        scr_count         int — total SCR peaks in signal
        calibration_mean  float — baseline SCL mean (µS)
        calibration_std   float — baseline SCL std (µS)
        low_threshold     float — cal_mean - low_sigma * cal_std
        high_threshold    float — cal_mean + high_sigma * cal_std
        in_band_fraction  float — fraction of frames in-band
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()

    # Load config
    try:
        cfg = _load_config(config_path).get("eda", {})
    except Exception:
        cfg = {}

    method = cfg.get("decomposition_method", "cvxeda")
    calibration_sec = float(cfg.get("calibration_sec", 120.0))
    low_sigma = float(cfg.get("low_threshold_sigma", 1.5))
    high_sigma = float(cfg.get("high_threshold_sigma", 2.5))
    scr_min_amplitude = float(cfg.get("scr_min_amplitude", 0.02))
    frame_rate_hz = float(cfg.get("frame_rate_hz", 1.0))

    # Compute n_frames for empty result
    samples_per_frame = max(1, int(round(fs / frame_rate_hz)))
    n_frames = len(signal) // samples_per_frame

    # Reject signals that are too short (< 10s)
    min_samples = int(10 * fs)
    if len(signal) < min_samples:
        logger.warning(
            "EDA signal too short (%.1fs, need ≥10s), returning NaN result",
            len(signal) / fs,
        )
        return _empty_result(n_frames)

    # Decompose signal
    try:
        decomposed = decompose_eda(signal, fs, method=method)
    except Exception as e:
        logger.warning("EDA decomposition failed: %s", e)
        return _empty_result(n_frames)

    tonic = decomposed["tonic"]
    phasic = decomposed["phasic"]

    # SCR peak detection on phasic component
    try:
        scr_info = detect_scr_peaks(phasic, fs, min_amplitude=scr_min_amplitude)
    except Exception as e:
        logger.warning("SCR peak detection failed: %s", e)
        scr_info = {"peak_indices": np.array([], dtype=np.int64),
                    "amplitudes": np.array([], dtype=np.float64),
                    "count": 0}

    # Calibration: use separate baseline signal or self-calibrate
    if calibration_signal is not None:
        cal_signal = np.asarray(calibration_signal, dtype=np.float64).ravel()
        cal_fs = float(calibration_fs) if calibration_fs is not None else fs
        try:
            cal_decomposed = decompose_eda(cal_signal, cal_fs, method=method)
            cal_tonic = cal_decomposed["tonic"]
        except Exception as e:
            logger.warning(
                "Calibration signal decomposition failed (%s), self-calibrating", e
            )
            cal_tonic = tonic
            cal_fs = fs
        cal_mean, cal_std = calibrate_baseline(cal_tonic, cal_fs, calibration_sec)
    else:
        # Self-calibrate from first calibration_sec of the signal's tonic
        cal_mean, cal_std = calibrate_baseline(tonic, fs, calibration_sec)

    low_threshold = cal_mean - low_sigma * cal_std
    high_threshold = cal_mean + high_sigma * cal_std

    # Compute arousal-in-band binary at 1 Hz
    arousal_in_band = compute_arousal_in_band(
        tonic, fs, cal_mean, cal_std, low_sigma, high_sigma, frame_rate_hz
    )

    # Build 1 Hz frame-level tonic and phasic summaries
    n_out = len(arousal_in_band)
    if n_out == 0:
        return _empty_result(0)

    trimmed_tonic = tonic[: n_out * samples_per_frame]
    trimmed_phasic = phasic[: n_out * samples_per_frame]

    tonic_frames = (
        trimmed_tonic.reshape(n_out, samples_per_frame)
        .mean(axis=1)
        .astype(np.float32)
    )
    phasic_frames = (
        trimmed_phasic.reshape(n_out, samples_per_frame)
        .max(axis=1)
        .astype(np.float32)
    )

    in_band_fraction = float(np.mean(arousal_in_band)) if len(arousal_in_band) > 0 else np.nan

    return {
        "arousal_in_band": arousal_in_band,
        "tonic_scl": tonic_frames,
        "phasic_scr": phasic_frames,
        "scr_count": scr_info["count"],
        "calibration_mean": float(cal_mean),
        "calibration_std": float(cal_std),
        "low_threshold": float(low_threshold),
        "high_threshold": float(high_threshold),
        "in_band_fraction": in_band_fraction,
    }
