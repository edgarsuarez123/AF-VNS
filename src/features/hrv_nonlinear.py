"""
Non-linear HRV: sample entropy (SampEn), DFA alpha1. Uses NeuroKit2. FR-2.2.
"""

from typing import Dict

import numpy as np


def compute_hrv_nonlinear(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Compute sampen and dfa_alpha1 from R-R intervals (seconds).
    Uses NeuroKit2 nk.entropy_sample and nk.hrv_nonlinear. Returns NaNs for short sequences.
    """
    rr = np.asarray(rr_intervals, dtype=np.float64).ravel()
    out: Dict[str, float] = {"sampen": np.nan, "dfa_alpha1": np.nan}
    if len(rr) < 10:
        return out
    try:
        import neurokit2 as nk
    except ImportError:
        return out
    # R-R in ms for neurokit2 (common convention)
    rr_ms = rr * 1000.0

    # SampEn via entropy_sample (works on raw RR series)
    try:
        se = nk.entropy_sample(rr_ms)
        # neurokit2 may return (value, info_dict) tuple or a scalar
        if isinstance(se, tuple):
            se = se[0]
        out["sampen"] = float(se)
        if not np.isfinite(out["sampen"]):
            out["sampen"] = np.nan
    except Exception:
        pass

    # DFA via hrv_nonlinear (needs peaks as cumulative sample indices, not raw RR)
    try:
        peaks = np.cumsum(rr_ms).astype(int)
        dfa_result = nk.hrv_nonlinear(peaks, sampling_rate=1000, silent=True)
        if dfa_result is not None and hasattr(dfa_result, "columns"):
            if "HRV_DFA_alpha1" in dfa_result.columns:
                val = dfa_result["HRV_DFA_alpha1"].iloc[0]
                if not np.isnan(val):
                    out["dfa_alpha1"] = float(val)
                    if not np.isfinite(out["dfa_alpha1"]):
                        out["dfa_alpha1"] = np.nan
    except Exception:
        pass

    return out
