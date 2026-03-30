"""ArousalGate — streaming EDA arousal gating for tinnitus aVNS tri-fold trigger.

Wraps eda.py functions in a stateful ring-buffer class for real-time operation.
The gate fires True (OK to stimulate) when tonic SCL is within the calibrated
arousal band: [cal_mean - low_sigma*cal_std, cal_mean + high_sigma*cal_std].

Usage:
    gate = ArousalGate("config_tinnitus.yaml")
    gate.calibrate(baseline_eda, fs=4.0)
    gate.update(new_samples, fs=4.0)
    if gate.is_in_band():
        fire_stim()
"""

import logging
import os
from collections import deque
from pathlib import Path

import numpy as np
import yaml

from ..features.eda import calibrate_baseline, compute_arousal_in_band, decompose_eda

logger = logging.getLogger(__name__)


class ArousalGate:
    """Real-time EDA arousal gate with ring buffer and per-subject calibration.

    Parameters
    ----------
    config_path : str
        Path to config YAML containing the 'eda' section.
    buffer_sec : float
        Duration (s) of the internal ring buffer. Defaults to 60s.
    """

    def __init__(self, config_path: str = "config_tinnitus.yaml", buffer_sec: float = 60.0):
        cfg = self._load_eda_config(config_path)

        self._low_sigma: float = float(cfg.get("low_threshold_sigma", 1.5))
        self._high_sigma: float = float(cfg.get("high_threshold_sigma", 2.5))
        self._frame_rate_hz: float = float(cfg.get("frame_rate_hz", 1.0))
        self._calibration_sec: float = float(cfg.get("calibration_sec", 1200.0))
        self._decomp_method: str = str(cfg.get("decomposition_method", "cvxeda"))
        self._buffer_sec: float = buffer_sec

        # Calibration state
        self._cal_mean: float = float("nan")
        self._cal_std: float = float("nan")
        self._calibrated: bool = False

        # Derived thresholds (set after calibrate())
        self._low_thresh: float = float("nan")
        self._high_thresh: float = float("nan")

        # Current gate state — updated by update()
        self._in_band: bool = False
        self._tonic_scl: float = float("nan")

        # Ring buffer — capacity set on first update() when fs is known
        self._buffer: deque = deque()
        self._buffer_capacity: int = 0

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, eda_signal: np.ndarray, fs: float) -> None:
        """Compute per-subject baseline (mean, std) from a recording segment.

        Parameters
        ----------
        eda_signal : 1D EDA signal in µS for the calibration period
        fs : sampling rate of eda_signal in Hz
        """
        eda_signal = np.asarray(eda_signal, dtype=np.float64).ravel()
        decomposed = decompose_eda(eda_signal, fs, method=self._decomp_method)
        tonic = decomposed["tonic"]

        self._cal_mean, self._cal_std = calibrate_baseline(
            tonic, fs, calibration_sec=self._calibration_sec)

        self._low_thresh = self._cal_mean - self._low_sigma * self._cal_std
        self._high_thresh = self._cal_mean + self._high_sigma * self._cal_std
        self._calibrated = True

        logger.info(
            "ArousalGate calibrated: mean=%.4f  std=%.6f  band=[%.4f, %.4f]",
            self._cal_mean, self._cal_std, self._low_thresh, self._high_thresh)

    # ------------------------------------------------------------------
    # Streaming update
    # ------------------------------------------------------------------

    def update(self, new_samples: np.ndarray, fs: float) -> None:
        """Append new EDA samples to ring buffer, recompute tonic + gate state.

        Parameters
        ----------
        new_samples : 1D EDA samples in µS
        fs : sampling rate of new_samples in Hz
        """
        new_samples = np.asarray(new_samples, dtype=np.float64).ravel()

        # Set buffer capacity on first call
        capacity = max(1, int(self._buffer_sec * fs))
        if self._buffer_capacity != capacity:
            self._buffer_capacity = capacity

        # Append and trim to capacity
        self._buffer.extend(new_samples.tolist())
        while len(self._buffer) > self._buffer_capacity:
            self._buffer.popleft()

        if len(self._buffer) == 0:
            return

        buf = np.array(self._buffer, dtype=np.float64)
        decomposed = decompose_eda(buf, fs, method=self._decomp_method)
        tonic = decomposed["tonic"]

        # Store last tonic value for get_state()
        self._tonic_scl = float(tonic[-1]) if len(tonic) > 0 else float("nan")

        if not self._calibrated:
            return

        # Compute gate using the last frame
        in_band_arr = compute_arousal_in_band(
            tonic, fs, self._cal_mean, self._cal_std,
            self._low_sigma, self._high_sigma,
            frame_rate_hz=self._frame_rate_hz)

        if len(in_band_arr) > 0:
            self._in_band = bool(in_band_arr[-1] > 0.5)

    # ------------------------------------------------------------------
    # Gate query
    # ------------------------------------------------------------------

    def is_in_band(self) -> bool:
        """Return True if current tonic SCL is within the calibrated arousal band.

        Raises
        ------
        RuntimeError if calibrate() has not been called yet.
        """
        if not self._calibrated:
            raise RuntimeError(
                "ArousalGate.calibrate() must be called before is_in_band(). "
                "Provide a baseline EDA recording.")
        return self._in_band

    def get_state(self) -> dict:
        """Return current gate state as a dict for logging/inspection.

        Returns
        -------
        dict with keys: in_band, tonic_scl, low_thresh, high_thresh, calibrated
        """
        return {
            "in_band": self._in_band,
            "tonic_scl": self._tonic_scl,
            "low_thresh": self._low_thresh,
            "high_thresh": self._high_thresh,
            "calibrated": self._calibrated,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_eda_config(config_path: str) -> dict:
        """Load the 'eda' section from a tinnitus config YAML."""
        if not os.path.isabs(config_path) and not os.path.isfile(config_path):
            root = Path(__file__).resolve().parents[2]
            config_path = str(root / config_path)
        with open(config_path) as f:
            full = yaml.safe_load(f)
        return full.get("eda", {})
