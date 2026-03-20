"""
Frequency-domain HRV from R-R intervals: LF, HF, LF/HF via Welch PSD. FR-2.2.
"""

from typing import Dict

import numpy as np
from scipy import integrate, signal as scipy_signal

LF_LO, LF_HI = 0.04, 0.15   # Hz
HF_LO, HF_HI = 0.15, 0.4   # Hz
MIN_SAMPLES = 32  # minimum for meaningful PSD


def compute_hrv_freq(
    rr_intervals: np.ndarray,
    fs_resample: float = 4.0,
) -> Dict[str, float]:
    """
    Resample R-R to uniform grid at fs_resample Hz, then Welch PSD.
    Returns lf, hf (power in band), lf_hf_ratio. Units: power; ratio dimensionless.
    Returns NaNs if insufficient length.
    """
    rr = np.asarray(rr_intervals, dtype=np.float64).ravel()
    out: Dict[str, float] = {"lf": np.nan, "hf": np.nan, "lf_hf_ratio": np.nan, "norm_hf_power": np.nan}
    if len(rr) < MIN_SAMPLES:
        return out

    # Interpolate to uniform time grid
    t_cum = np.concatenate([[0], np.cumsum(rr)])
    t_unif = np.arange(0, t_cum[-1], 1.0 / fs_resample)
    if len(t_unif) < MIN_SAMPLES:
        return out
    rr_unif = np.interp(t_unif, t_cum[:-1], rr)

    freqs, psd = scipy_signal.welch(rr_unif, fs=fs_resample, nperseg=min(256, len(rr_unif) // 2))

    lf_mask = (freqs >= LF_LO) & (freqs <= LF_HI)
    hf_mask = (freqs >= HF_LO) & (freqs <= HF_HI)
    lf_power = float(integrate.trapezoid(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 0.0)
    hf_power = float(integrate.trapezoid(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0.0)

    out["lf"] = float(lf_power)
    out["hf"] = float(hf_power)
    out["lf_hf_ratio"] = float(lf_power / hf_power) if hf_power > 0 else np.nan
    total = lf_power + hf_power
    out["norm_hf_power"] = float(hf_power / total) if total > 0 else np.nan
    return out
