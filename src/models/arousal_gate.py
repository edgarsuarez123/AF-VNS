"""ArousalGate — streaming EDA arousal gating for tinnitus aVNS tri-fold trigger.

Wraps eda.py functions in a stateful ring-buffer class for real-time operation.
The gate fires True (OK to stimulate) when tonic SCL is within the calibrated
arousal band: [cal_mean - low_sigma*cal_std, cal_mean + high_sigma*cal_std].

Optional trained classifier (F13): if an ArousalClassifier is supplied, is_in_band()
uses the classifier's prediction on the current 5-feature EDA vector instead of the
rule-based sigma threshold.  The rule-based path is always available as fallback.

Usage (rule-based, existing):
    gate = ArousalGate("config_tinnitus.yaml")
    gate.calibrate(baseline_eda, fs=4.0)
    gate.update(new_samples, fs=4.0)
    if gate.is_in_band():
        fire_stim()

Usage (with trained classifier, F13):
    clf = ArousalClassifier.load("models/checkpoints/arousal_classifier.pkl")
    gate = ArousalGate("config_tinnitus.yaml", classifier=clf)
    gate.update(new_samples, fs=4.0)   # no calibrate() call needed
    if gate.is_in_band():
        fire_stim()
"""

import logging
import os
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from ..features.eda import (
    calibrate_baseline,
    compute_arousal_in_band,
    decompose_eda,
    detect_scr_peaks,
)

logger = logging.getLogger(__name__)


class ArousalGate:
    """Real-time EDA arousal gate with ring buffer and per-subject calibration.

    Parameters
    ----------
    config_path : str
        Path to config YAML containing the 'eda' section.
    buffer_sec : float
        Duration (s) of the internal ring buffer. Defaults to 60s.
    classifier : ArousalClassifier or None
        Optional trained F13 classifier. When provided, is_in_band() uses the
        classifier's prediction instead of the rule-based sigma threshold.
        calibrate() is not required when a classifier is provided.
    """

    def __init__(
        self,
        config_path: str = "config_tinnitus.yaml",
        buffer_sec: float = 60.0,
        classifier=None,
    ):
        cfg = self._load_eda_config(config_path)

        self._low_sigma: float = float(cfg.get("low_threshold_sigma", 1.5))
        self._high_sigma: float = float(cfg.get("high_threshold_sigma", 2.5))
        self._frame_rate_hz: float = float(cfg.get("frame_rate_hz", 1.0))
        self._calibration_sec: float = float(cfg.get("calibration_sec", 1200.0))
        self._decomp_method: str = str(cfg.get("decomposition_method", "cvxeda"))
        self._scr_min_amplitude: float = float(cfg.get("scr_min_amplitude", 0.02))
        self._buffer_sec: float = buffer_sec

        # Optional trained classifier (F13)
        self._classifier = classifier

        # Calibration state (rule-based path)
        self._cal_mean: float = float("nan")
        self._cal_std: float = float("nan")
        self._calibrated: bool = False

        # Derived thresholds (set after calibrate())
        self._low_thresh: float = float("nan")
        self._high_thresh: float = float("nan")

        # Current rule-based gate state — updated by update()
        self._in_band: bool = False
        self._tonic_scl: float = float("nan")

        # Current EDA feature vector for classifier path — updated by update()
        self._current_features: Optional[np.ndarray] = None
        self._eda_fs: float = 0.0  # saved from last update() call

        # Optional temperature ring buffer (F19)
        self._temp_buffer: deque = deque()
        self._temp_buffer_capacity: int = 0

        # Ring buffer — capacity set on first update() when fs is known
        self._buffer: deque = deque()
        self._buffer_capacity: int = 0

    # ------------------------------------------------------------------
    # Calibration (rule-based path only)
    # ------------------------------------------------------------------

    def calibrate(self, eda_signal: np.ndarray, fs: float) -> None:
        """Compute per-subject baseline (mean, std) from a recording segment.

        Parameters
        ----------
        eda_signal : 1D EDA signal in µS for the calibration period
        fs : sampling rate of eda_signal in Hz

        Note: Not required when a trained classifier is provided.
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

    def update(
        self,
        new_samples: np.ndarray,
        fs: float,
        temp_samples: Optional[np.ndarray] = None,
    ) -> None:
        """Append new EDA samples to ring buffer, recompute tonic + gate state.

        Also updates the feature EDA vector used by the classifier path.

        Parameters
        ----------
        new_samples : 1D EDA samples in µS
        fs : sampling rate of new_samples in Hz
        temp_samples : optional 1D skin temperature samples (°C) at same fs (F19)
        """
        new_samples = np.asarray(new_samples, dtype=np.float64).ravel()
        self._eda_fs = float(fs)

        # Set buffer capacity on first call
        capacity = max(1, int(self._buffer_sec * fs))
        if self._buffer_capacity != capacity:
            self._buffer_capacity = capacity
            self._temp_buffer_capacity = capacity

        # Append and trim EDA to capacity
        self._buffer.extend(new_samples.tolist())
        while len(self._buffer) > self._buffer_capacity:
            self._buffer.popleft()

        # Append and trim temperature to capacity (F19)
        if temp_samples is not None:
            temp_arr = np.asarray(temp_samples, dtype=np.float64).ravel()
            self._temp_buffer.extend(temp_arr.tolist())
            while len(self._temp_buffer) > self._temp_buffer_capacity:
                self._temp_buffer.popleft()

        if len(self._buffer) == 0:
            return

        buf = np.array(self._buffer, dtype=np.float64)
        decomposed = decompose_eda(buf, fs, method=self._decomp_method)
        tonic = decomposed["tonic"]
        phasic = decomposed["phasic"]

        # Store last tonic value for get_state()
        self._tonic_scl = float(tonic[-1]) if len(tonic) > 0 else float("nan")

        # Build optional temp buffer array for feature extraction (F19)
        temp_buf = np.array(self._temp_buffer, dtype=np.float64) if len(self._temp_buffer) > 0 else None

        # --- Update feature vector for classifier path ---
        self._current_features = self._compute_features(tonic, phasic, buf, fs, temp_buf)

        # --- Rule-based path ---
        if self._calibrated:
            in_band_arr = compute_arousal_in_band(
                tonic, fs, self._cal_mean, self._cal_std,
                self._low_sigma, self._high_sigma,
                frame_rate_hz=self._frame_rate_hz)
            if len(in_band_arr) > 0:
                self._in_band = bool(in_band_arr[-1] > 0.5)

        # --- Classifier path: update _in_band if classifier available ---
        if self._classifier is not None and self._current_features is not None:
            try:
                self._in_band = self._classifier.predict(self._current_features)
            except Exception as exc:
                logger.debug("Classifier predict failed, keeping rule-based state: %s", exc)

    def _compute_features(
        self,
        tonic: np.ndarray,
        phasic: np.ndarray,
        buf: np.ndarray,
        fs: float,
        temp_buf: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute 5 or 6 EDA features from current buffer decomposition.

        Returns (5,) float32 when temp_buf is None:
            [tonic_scl_mean, tonic_scl_std, phasic_mean, max_scr_amplitude, scr_rate]
        Returns (6,) float32 when temp_buf is provided (F19):
            [...5 features..., skin_temp_mean]
        """
        tonic_scl_mean = float(np.nanmean(tonic))
        tonic_scl_std = float(np.nanstd(tonic))
        phasic_mean = float(np.nanmean(phasic))

        try:
            peaks = detect_scr_peaks(phasic, fs, min_amplitude=self._scr_min_amplitude)
            max_scr_amp = float(np.max(peaks["amplitudes"])) if peaks["count"] > 0 else 0.0
            duration_min = len(buf) / fs / 60.0
            scr_rate = float(peaks["count"] / duration_min) if duration_min > 0 else 0.0
        except Exception:
            max_scr_amp = 0.0
            scr_rate = 0.0

        feats = [tonic_scl_mean, tonic_scl_std, phasic_mean, max_scr_amp, scr_rate]

        # F19: optional skin temperature feature
        if temp_buf is not None and len(temp_buf) > 0:
            feats.append(float(np.nanmean(temp_buf)))

        return np.array(feats, dtype=np.float32)

    # ------------------------------------------------------------------
    # Gate query
    # ------------------------------------------------------------------

    def is_in_band(self) -> bool:
        """Return True if current EDA state is within the stimulation-safe band.

        Decision hierarchy:
          1. Trained classifier (F13): if provided, uses classifier.predict()
          2. Rule-based threshold: requires calibrate() to have been called

        Raises
        ------
        RuntimeError if no classifier AND calibrate() has not been called.
        """
        if self._classifier is not None:
            # Classifier path: requires at least one update() call
            if self._current_features is None:
                return False  # no data yet — hold stimulation
            return self._in_band  # already set by update()

        # Rule-based fallback
        if not self._calibrated:
            raise RuntimeError(
                "ArousalGate.calibrate() must be called before is_in_band(). "
                "Provide a baseline EDA recording, or supply a trained classifier.")
        return self._in_band

    def get_state(self) -> dict:
        """Return current gate state as a dict for logging/inspection.

        Returns
        -------
        dict with keys: in_band, tonic_scl, low_thresh, high_thresh, calibrated,
                        using_classifier, current_features
        """
        return {
            "in_band": self._in_band,
            "tonic_scl": self._tonic_scl,
            "low_thresh": self._low_thresh,
            "high_thresh": self._high_thresh,
            "calibrated": self._calibrated,
            "using_classifier": self._classifier is not None,
            "current_features": (
                self._current_features.tolist()
                if self._current_features is not None else None
            ),
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
