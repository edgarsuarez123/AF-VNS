"""
Time-domain HRV from R-R intervals: RMSSD, SDNN. FR-2.2. Units: seconds.
"""

from typing import Dict

import numpy as np

MIN_INTERVALS = 5  # below this, return NaN-like values


def compute_hrv_time(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Compute RMSSD and SDNN from R-R intervals (in seconds).
    Returns dict with keys: rmssd, sdnn. Returns np.nan for each if too few intervals.
    """
    rr = np.asarray(rr_intervals, dtype=np.float64).ravel()
    out: Dict[str, float] = {"rmssd": np.nan, "sdnn": np.nan}
    if len(rr) < MIN_INTERVALS:
        return out
    # RMSSD = sqrt(mean(diff(rr)^2))
    diffs = np.diff(rr)
    out["rmssd"] = float(np.sqrt(np.mean(diffs ** 2)))
    out["sdnn"] = float(np.std(rr))
    return out
