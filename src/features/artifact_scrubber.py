"""
Artifact detection and correction per Aim 2 methodology.
Amplitude: reject on X×MAD. R-R: correct outlier beats via interpolation,
reject only if fraction of outliers exceeds threshold.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.isabs(config_path) and not os.path.isfile(config_path):
        root = Path(__file__).resolve().parents[2]
        config_path = root / config_path
    with open(config_path) as f:
        return yaml.safe_load(f)


def correct_rr_intervals(
    rr_intervals: np.ndarray,
    rr_deviation_percent: Optional[float] = None,
    config_path: str = "config.yaml",
) -> Tuple[np.ndarray, float]:
    """
    Correct outlier RR intervals by replacing them with linear interpolation
    from neighboring valid beats. Returns (corrected_rr, fraction_corrected).

    Outlier = deviates from median by more than rr_deviation_percent%.
    """
    rr = np.asarray(rr_intervals, dtype=np.float64).ravel()
    if len(rr) == 0:
        return rr, 1.0

    if rr_deviation_percent is None:
        config = load_config(config_path)
        art = config.get("artifact", {})
        rr_deviation_percent = float(art.get("rr_deviation_percent", 60.0))

    rr_med = np.median(rr)
    if rr_med <= 0:
        return rr, 1.0

    dev_pct = np.abs(rr - rr_med) / rr_med * 100
    outlier_mask = dev_pct > rr_deviation_percent
    n_outliers = int(np.sum(outlier_mask))
    fraction = n_outliers / len(rr)

    if n_outliers == 0:
        return rr, 0.0

    corrected = rr.copy()
    valid_indices = np.where(~outlier_mask)[0]
    if len(valid_indices) < 2:
        # Too few valid beats to interpolate; replace with median
        corrected[outlier_mask] = rr_med
    else:
        outlier_indices = np.where(outlier_mask)[0]
        corrected[outlier_indices] = np.interp(
            outlier_indices, valid_indices, rr[valid_indices]
        )

    return corrected, fraction


def should_reject_window(
    signal: np.ndarray,
    rr_intervals: np.ndarray,
    amplitude_mad_multiple: Optional[float] = None,
    rr_deviation_percent: Optional[float] = None,
    rr_fraction_threshold: Optional[float] = None,
    config_path: str = "config.yaml",
) -> bool:
    """
    Return True if the window should be rejected (artifact).
    - Amplitude: reject if max absolute deviation exceeds amplitude_mad_multiple * MAD.
    - R-R: reject only if fraction of outlier beats > rr_fraction_threshold.
      (Fraction-based per Aim 2 "correction not rejection")
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    rr_intervals = np.asarray(rr_intervals, dtype=np.float64).ravel()

    if amplitude_mad_multiple is None or rr_deviation_percent is None or rr_fraction_threshold is None:
        config = load_config(config_path)
        art = config.get("artifact", {})
        amplitude_mad_multiple = amplitude_mad_multiple if amplitude_mad_multiple is not None else art.get("amplitude_mad_multiple", 5.0)
        rr_deviation_percent = rr_deviation_percent if rr_deviation_percent is not None else art.get("rr_deviation_percent", 60.0)
        rr_fraction_threshold = rr_fraction_threshold if rr_fraction_threshold is not None else art.get("rr_fraction_threshold", 0.30)

    # Amplitude: MAD of signal; use epsilon when MAD=0 so constant signal is not rejected
    med = np.median(signal)
    mad = np.median(np.abs(signal - med))
    max_dev = np.max(np.abs(signal - med))
    mad_eff = max(mad, 1e-12)
    if max_dev > amplitude_mad_multiple * mad_eff:
        return True

    # R-R: fraction-based rejection
    if len(rr_intervals) > 0:
        _, fraction = correct_rr_intervals(rr_intervals, rr_deviation_percent, config_path)
        if fraction > rr_fraction_threshold:
            return True

    return False
