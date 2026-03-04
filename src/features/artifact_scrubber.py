"""
Artifact rejection: label windows as reject when amplitude or R-R is out of range. Plan: X×MAD, Y% R-R deviation.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.isabs(config_path) and not os.path.isfile(config_path):
        root = Path(__file__).resolve().parents[2]
        config_path = root / config_path
    with open(config_path) as f:
        return yaml.safe_load(f)


def should_reject_window(
    signal: np.ndarray,
    rr_intervals: np.ndarray,
    amplitude_mad_multiple: Optional[float] = None,
    rr_deviation_percent: Optional[float] = None,
    config_path: str = "config.yaml",
) -> bool:
    """
    Return True if the window should be rejected (artifact).
    - Amplitude: reject if max absolute deviation exceeds amplitude_mad_multiple * MAD.
    - R-R: reject if any interval deviates from median R-R by more than rr_deviation_percent%.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    rr_intervals = np.asarray(rr_intervals, dtype=np.float64).ravel()

    if amplitude_mad_multiple is None or rr_deviation_percent is None:
        config = load_config(config_path)
        art = config.get("artifact", {})
        amplitude_mad_multiple = amplitude_mad_multiple or art.get("amplitude_mad_multiple", 5.0)
        rr_deviation_percent = rr_deviation_percent or art.get("rr_deviation_percent", 25.0)

    # Amplitude: MAD of signal; use epsilon when MAD=0 so constant signal is not rejected
    med = np.median(signal)
    mad = np.median(np.abs(signal - med))
    max_dev = np.max(np.abs(signal - med))
    mad_eff = max(mad, 1e-12)
    if max_dev > amplitude_mad_multiple * mad_eff:
        return True

    # R-R: deviation from median
    if len(rr_intervals) > 0:
        rr_med = np.median(rr_intervals)
        if rr_med > 0:
            dev_pct = np.abs(rr_intervals - rr_med) / rr_med * 100
            if np.any(dev_pct > rr_deviation_percent):
                return True

    return False
