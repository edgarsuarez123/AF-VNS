"""
Autonomic State Module — sliding-window HRV feature extractor (FR-2.3).

Input:  60s of R-R intervals (or raw ECG)
Output: autonomic state vector [lf_hf_ratio, norm_hf_power, sampen, dfa_alpha1]

Updates every 60s. Feeds the stim parameter recommender (not ML — control algorithm).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..features.hrv_freq import compute_hrv_freq
from ..features.hrv_nonlinear import compute_hrv_nonlinear

logger = logging.getLogger(__name__)

# Default feature order matches grant Aim 2 requirements
DEFAULT_FEATURES = ("lf_hf_ratio", "norm_hf_power", "sampen", "dfa_alpha1")


@dataclass(frozen=True)
class AutonomicStateConfig:
    window_sec: float = 60.0
    features: tuple[str, ...] = DEFAULT_FEATURES
    fs_resample: float = 4.0  # Hz — Welch PSD resampling rate


class AutonomicState:
    """Sliding-window HRV feature extractor for autonomic nervous system assessment.

    Wraps existing compute_hrv_freq() and compute_hrv_nonlinear() into a single
    interface that produces the 4-feature autonomic state vector required by the grant.
    """

    def __init__(self, cfg: AutonomicStateConfig):
        self.cfg = cfg

    def compute(self, rr_intervals: np.ndarray) -> dict[str, float]:
        """Compute autonomic state from R-R intervals (seconds).

        Args:
            rr_intervals: 1D array of R-R intervals in seconds (from ~60s window).

        Returns:
            Dict with keys matching cfg.features. NaN for failed/insufficient features.
        """
        rr = np.asarray(rr_intervals, dtype=np.float64).ravel()
        state: dict[str, float] = {f: np.nan for f in self.cfg.features}

        freq = compute_hrv_freq(rr, fs_resample=self.cfg.fs_resample)
        nonlinear = compute_hrv_nonlinear(rr)

        # Map computed values to requested features
        lookup = {**freq, **nonlinear}
        for feat in self.cfg.features:
            if feat in lookup:
                state[feat] = lookup[feat]

        return state

    def compute_from_ecg(
        self,
        ecg_signal: np.ndarray,
        fs: float,
        config_path: str = "config_stroke.yaml",
    ) -> dict[str, float]:
        """Convenience: raw ECG → R-peak detection → RR intervals → compute().

        Args:
            ecg_signal: 1D ECG signal (~60s at fs Hz).
            fs: Sampling frequency in Hz.
            config_path: Config path for denoising and artifact correction.

        Returns:
            Autonomic state dict (same as compute()).
        """
        from ..features.wavelet_filter import denoise
        from ..features.peak_detector import get_rr_intervals
        from ..features.artifact_scrubber import correct_rr_intervals

        signal = np.asarray(ecg_signal, dtype=np.float64).ravel()
        denoised = denoise(signal, fs, config_path=config_path)
        rr = get_rr_intervals(denoised, fs)

        if len(rr) < 5:
            return {f: np.nan for f in self.cfg.features}

        rr_corrected, _ = correct_rr_intervals(rr, config_path=config_path)
        if len(rr_corrected) < 5:
            return {f: np.nan for f in self.cfg.features}

        return self.compute(rr_corrected)

    def as_vector(self, state: dict[str, float]) -> np.ndarray:
        """Convert state dict to fixed-order numpy array matching cfg.features.

        Args:
            state: Dict from compute() or compute_from_ecg().

        Returns:
            (n_features,) float32 array.
        """
        return np.array([state.get(f, np.nan) for f in self.cfg.features], dtype=np.float32)


def build_autonomic_state(config_path: str = "config_stroke.yaml") -> AutonomicState:
    """Factory: load config → AutonomicStateConfig → AutonomicState."""
    from ..training.build_model import load_config

    config = load_config(config_path)
    m = config.get("autonomic_state", {})

    cfg = AutonomicStateConfig(
        window_sec=float(m.get("window_sec", 60.0)),
        features=tuple(m.get("features", list(DEFAULT_FEATURES))),
        fs_resample=float(m.get("fs_resample", 4.0)),
    )
    return AutonomicState(cfg)
