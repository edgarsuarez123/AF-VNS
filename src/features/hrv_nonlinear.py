"""
Non-linear HRV: sample entropy (SampEn), DFA alpha1. Uses NeuroKit2. FR-2.2.
"""

from typing import Dict

import numpy as np


def compute_hrv_nonlinear(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Compute sampen and dfa_alpha1 from R-R intervals (seconds).
    Uses NeuroKit2 nk.entropy_sample and nk.dfa. Returns NaNs for short sequences.
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
    try:
        out["sampen"] = float(nk.entropy_sample(rr_ms))
    except Exception:
        pass
    try:
        dfa_indices = nk.hrv_nonlinear(rr_ms, silent=True)
        if dfa_indices is not None and hasattr(dfa_indices, "columns"):
            for col in ("DFA_Alpha1", "HRV_DFA_alpha1", "DFA_alpha1"):
                if col in dfa_indices.columns:
                    out["dfa_alpha1"] = float(dfa_indices[col].iloc[0])
                    break
        if hasattr(dfa_indices, "iloc") and np.isnan(out["dfa_alpha1"]):
            row = dfa_indices.iloc[0]
            for k in ("DFA_Alpha1", "HRV_DFA_alpha1", "DFA_alpha1"):
                if k in row:
                    out["dfa_alpha1"] = float(row[k])
                    break
    except Exception:
        try:
            out["dfa_alpha1"] = float(nk.dfa(rr_ms)[0])
        except Exception:
            pass
    return out
