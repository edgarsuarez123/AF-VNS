"""
R-peak detection from ECG/PPG. Returns R-R intervals in seconds. FR-1.1 requires fs >= 250 Hz.
"""

from typing import Optional

import numpy as np


def get_rr_intervals(
    signal: np.ndarray,
    fs: float,
    signal_type: str = "ecg",
) -> np.ndarray:
    """
    Detect peaks and return R-R intervals in seconds.
    signal: 1D array (ECG or PPG). fs: sampling rate in Hz (>= 250 Hz recommended).
    signal_type: "ecg" (default) or "ppg".
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    try:
        import neurokit2 as nk
    except ImportError:
        raise ImportError("neurokit2 is required for get_rr_intervals")

    if signal_type.lower() == "ecg":
        _, info = nk.ecg_peaks(signal, sampling_rate=fs)
        peaks = info["ECG_R_Peaks"]
    else:
        peaks = nk.ppg_findpeaks(signal, sampling_rate=fs)["PPG_Peaks"]

    peaks = np.asarray(peaks, dtype=np.int64)
    if len(peaks) < 2:
        return np.array([], dtype=np.float64)

    intervals = np.diff(peaks) / float(fs)  # seconds
    return intervals.astype(np.float64)
