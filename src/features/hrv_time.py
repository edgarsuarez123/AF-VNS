"""
Time-domain HRV from R-R intervals: RMSSD, SDNN, pNN50, CoV. FR-2.2. Units: seconds.
"""

from typing import Dict

import numpy as np

MIN_INTERVALS = 5  # below this, return NaN-like values


def compute_hrv_time(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Compute RMSSD, SDNN, pNN50, and CoV from R-R intervals (in seconds).
    Returns dict with keys: rmssd, sdnn, pnn50, cov_rr.
    Returns np.nan for each if too few intervals.
    """
    rr = np.asarray(rr_intervals, dtype=np.float64).ravel()
    out: Dict[str, float] = {"rmssd": np.nan, "sdnn": np.nan, "pnn50": np.nan, "cov_rr": np.nan}
    if len(rr) < MIN_INTERVALS:
        return out
    # RMSSD = sqrt(mean(diff(rr)^2))
    diffs = np.diff(rr)
    out["rmssd"] = float(np.sqrt(np.mean(diffs ** 2)))
    out["sdnn"] = float(np.std(rr))
    # pNN50: fraction of successive RR diffs > 50ms
    out["pnn50"] = float(np.mean(np.abs(diffs) > 0.050))
    # CoV: coefficient of variation (std/mean)
    mean_rr = float(np.mean(rr))
    out["cov_rr"] = float(np.std(rr) / mean_rr) if mean_rr > 0 else np.nan
    return out
