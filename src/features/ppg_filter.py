"""
PPG bandpass filter for the tinnitus aVNS pipeline.

Replaces the ECG wavelet denoiser (wavelet_filter.py) for PPG signals.
PPG morphology has much lower frequency content than ECG — a tight
0.5–8 Hz Butterworth bandpass removes baseline drift and motion/HF noise
while preserving the pulse waveform and dicrotic notch.

Reference: SBIR spec requires ≥125 Hz PPG; passband tuned to cover
0.5 Hz (40 bpm resting HR lower bound) through 8 Hz (sufficient for
dicrotic notch and 2nd harmonic at 240 bpm max).
"""

import logging
from typing import Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt

logger = logging.getLogger(__name__)

# Minimum signal length required for filtfilt (3× filter order + 1)
_MIN_SAMPLES_FACTOR = 10


def _load_ppg_filter_config(config_path: str) -> dict:
    import yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("ppg_filter", {})


def denoise_ppg(
    signal: np.ndarray,
    fs: float,
    config_path: Optional[str] = None,
    bandpass_low: float = 0.5,
    bandpass_high: float = 8.0,
    bandpass_order: int = 4,
) -> np.ndarray:
    """Butterworth bandpass filter for PPG denoising.

    Parameters
    ----------
    signal : 1D float64 PPG waveform
    fs : sampling rate in Hz
    config_path : if provided, load filter params from config_tinnitus.yaml
    bandpass_low : lower cutoff Hz (default 0.5)
    bandpass_high : upper cutoff Hz (default 8.0)
    bandpass_order : Butterworth order (default 4)

    Returns
    -------
    Filtered 1D float64 array, same length as input.
    Returns input unchanged (with warning) if signal is too short to filter.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()

    if config_path is not None:
        cfg = _load_ppg_filter_config(config_path)
        bandpass_low = float(cfg.get("bandpass_low_hz", bandpass_low))
        bandpass_high = float(cfg.get("bandpass_high_hz", bandpass_high))
        bandpass_order = int(cfg.get("bandpass_order", bandpass_order))

    min_len = _MIN_SAMPLES_FACTOR * bandpass_order
    if len(signal) < min_len:
        logger.warning(
            "PPG signal too short (%d samples) for order-%d filter — returning unfiltered",
            len(signal), bandpass_order,
        )
        return signal.copy()

    nyq = fs / 2.0
    low = bandpass_low / nyq
    high = min(bandpass_high / nyq, 0.99)  # clamp below Nyquist

    if low >= high:
        logger.warning(
            "PPG filter: low cutoff (%.2f) >= high cutoff (%.2f) after normalization — "
            "returning unfiltered", low, high,
        )
        return signal.copy()

    sos = butter(bandpass_order, [low, high], btype="band", output="sos")
    filtered = sosfiltfilt(sos, signal)
    return filtered.astype(np.float64)
