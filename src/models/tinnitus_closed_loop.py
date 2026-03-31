"""
F12 — Tri-Fold Closed-Loop Inference Pipeline for Tinnitus aVNS.

Wires PhaseDetector (PPG) + ArousalGate (EDA) + AutonomicState + StimRecommender
into a single synchronous streaming pipeline.  Caller feeds PPG samples (and
optionally EDA samples) and receives TinnitusStimEvent objects whenever all three
trigger conditions are met simultaneously:

    1. Diastole probability > diastole_threshold  (cardiac gating)
    2. Exhalation probability > exhalation_threshold  (respiratory gating)
    3. ArousalGate.is_in_band() == True  (EDA arousal gating)

Fast path  (~100ms stride):
    last 2s PPG -> denoise_ppg() -> PhaseDetector CNN -> sigmoid ->
    tri-fold check -> TinnitusStimEvent

EDA path  (on each feed() call with eda_samples):
    arousal_gate.update(eda_samples, eda_fs)

Slow path  (every slow_window_sec = 60s):
    last 60s PPG -> denoise_ppg() -> get_rr_intervals(ppg) ->
    correct_rr_intervals() -> autonomic_state.compute(rr) ->
    stim_recommender.recommend() -> update stim params

No threading — caller controls timing.  Suitable for LSL loops, test harnesses,
and offline replay.

Usage:
    pipe = build_tinnitus_closed_loop_pipeline(device="cpu")
    pipe.calibrate_eda(baseline_eda, fs=4.0)
    events = pipe.feed(ppg_chunk, eda_chunk)   # np.ndarray (N,) each
    state  = pipe.get_state()
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TinnitusStimEvent:
    """A tri-fold synchronized stimulation trigger event."""
    timestamp_samples: int    # Stream PPG sample index when this event fired
    amplitude: float          # mA  (from StimRecommender)
    frequency: float          # Hz
    pulse_width: float        # µs
    diastole_prob: float      # Raw sigmoid probability at trigger frame
    exhalation_prob: float    # Raw sigmoid probability at trigger frame
    arousal_in_band: bool     # EDA gate state at trigger time


@dataclass
class TinnitusPipelineState:
    """Snapshot of pipeline counters and history."""
    total_samples_fed: int
    stim_events: list[TinnitusStimEvent]
    last_autonomic_state: Optional[dict]
    last_stim_params: Optional[dict]
    fast_path_calls: int
    slow_path_calls: int
    eda_calibrated: bool


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

# Safe stim defaults used before first autonomic state update (< 60s data).
# Values are within tinnitus SBIR hardware limits (0.8 mA, 30 Hz, 100 µs).
_SAFE_STIM_DEFAULTS = {"amplitude": 0.2, "frequency": 10.0, "pulse_width": 50.0}


class TinnitusClosedLoopPipeline:
    """Synchronous tri-fold closed-loop VNS pipeline for tinnitus aVNS.

    Args:
        phase_detector:       PhaseDetector CNN (eval mode, on target device)
        autonomic_state:      AutonomicState HRV extractor
        stim_recommender:     StimRecommender rule engine
        arousal_gate:         ArousalGate EDA arousal gate
        ppg_fs:               Expected PPG sampling rate in Hz (default 125)
        eda_fs:               Expected EDA sampling rate in Hz (default 4)
        inference_stride_ms:  Fast-path stride in ms (default 100)
        slow_window_sec:      Slow-path window length in seconds (default 60)
        diastole_threshold:   Sigmoid threshold for diastole binary decision
        exhalation_threshold: Sigmoid threshold for exhalation binary decision
        config_path:          Path to config YAML (used for artifact correction)
    """

    def __init__(
        self,
        phase_detector,
        autonomic_state,
        stim_recommender,
        arousal_gate,
        ppg_fs: float = 125.0,
        eda_fs: float = 4.0,
        inference_stride_ms: float = 100.0,
        slow_window_sec: float = 60.0,
        diastole_threshold: float = 0.5,
        exhalation_threshold: float = 0.5,
        config_path: str = "config_tinnitus.yaml",
    ) -> None:
        self._phase_detector = phase_detector
        self._autonomic_state = autonomic_state
        self._stim_recommender = stim_recommender
        self._arousal_gate = arousal_gate
        self._ppg_fs = float(ppg_fs)
        self._eda_fs = float(eda_fs)
        self._dia_threshold = float(diastole_threshold)
        self._exh_threshold = float(exhalation_threshold)
        self._config_path = config_path

        # Window / stride sizes in samples (PPG domain)
        self._fast_window_samples = int(round(2.0 * self._ppg_fs))           # 250 @ 125 Hz
        self._slow_window_samples = int(round(slow_window_sec * self._ppg_fs))  # 7500 @ 125 Hz
        self._stride_samples = max(1, int(round(inference_stride_ms / 1000.0 * self._ppg_fs)))  # 13 @ 125 Hz

        # PPG ring buffer — maxlen ensures automatic eviction of old data
        self._ppg_buffer: collections.deque = collections.deque(
            maxlen=self._slow_window_samples
        )

        # Counters
        self._samples_since_last_inference: int = 0
        self._samples_since_last_slow: int = 0
        self._total_samples: int = 0

        # State
        self._last_stim_params: dict = dict(_SAFE_STIM_DEFAULTS)
        self._last_autonomic_state: Optional[dict] = None
        self._stim_events: list[TinnitusStimEvent] = []
        self._fast_path_calls: int = 0
        self._slow_path_calls: int = 0

        # Get device from phase detector
        try:
            self._device = next(phase_detector.parameters()).device
        except StopIteration:
            self._device = torch.device("cpu")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate_eda(self, baseline_eda: np.ndarray, fs: float) -> None:
        """Calibrate the EDA arousal gate from a baseline recording.

        Args:
            baseline_eda: 1D EDA signal in µS (ideally ≥ 20 minutes for WESAD).
            fs:           Sampling rate of baseline_eda in Hz.
        """
        self._arousal_gate.calibrate(baseline_eda, fs)

    def feed(
        self,
        ppg_samples: np.ndarray,
        eda_samples: Optional[np.ndarray] = None,
    ) -> list[TinnitusStimEvent]:
        """Feed PPG (and optionally EDA) samples into the pipeline.

        Args:
            ppg_samples:  1-D float array of PPG values at ppg_fs Hz.
            eda_samples:  Optional 1-D float array of EDA values at eda_fs Hz.
                          If provided, updates the ArousalGate ring buffer.

        Returns:
            List of TinnitusStimEvent objects triggered during this call
            (usually empty or one per stride).
        """
        ppg_samples = np.asarray(ppg_samples, dtype=np.float32).ravel()
        triggered: list[TinnitusStimEvent] = []

        # Update EDA gate before the PPG loop so the arousal state is current
        if eda_samples is not None:
            eda_arr = np.asarray(eda_samples, dtype=np.float64).ravel()
            if len(eda_arr) > 0:
                self._arousal_gate.update(eda_arr, self._eda_fs)

        for s in ppg_samples:
            self._ppg_buffer.append(float(s))
            self._total_samples += 1
            self._samples_since_last_inference += 1
            self._samples_since_last_slow += 1

            # Fast path — every stride_samples
            if (self._samples_since_last_inference >= self._stride_samples
                    and len(self._ppg_buffer) >= self._fast_window_samples):
                self._samples_since_last_inference = 0
                event = self._run_fast_path()
                if event is not None:
                    triggered.append(event)
                    self._stim_events.append(event)

            # Slow path — every slow_window_samples
            if (self._samples_since_last_slow >= self._slow_window_samples
                    and len(self._ppg_buffer) >= self._slow_window_samples):
                self._samples_since_last_slow = 0
                self._run_slow_path()

        return triggered

    def reset(self) -> None:
        """Clear all buffers and state. Pipeline returns to initial condition."""
        self._ppg_buffer.clear()
        self._samples_since_last_inference = 0
        self._samples_since_last_slow = 0
        self._total_samples = 0
        self._last_stim_params = dict(_SAFE_STIM_DEFAULTS)
        self._last_autonomic_state = None
        self._stim_events = []
        self._fast_path_calls = 0
        self._slow_path_calls = 0

    def get_state(self) -> TinnitusPipelineState:
        """Return a snapshot of the current pipeline state."""
        return TinnitusPipelineState(
            total_samples_fed=self._total_samples,
            stim_events=list(self._stim_events),
            last_autonomic_state=dict(self._last_autonomic_state) if self._last_autonomic_state else None,
            last_stim_params=dict(self._last_stim_params),
            fast_path_calls=self._fast_path_calls,
            slow_path_calls=self._slow_path_calls,
            eda_calibrated=self._arousal_gate._calibrated,
        )

    # ------------------------------------------------------------------
    # Internal paths
    # ------------------------------------------------------------------

    def _run_fast_path(self) -> Optional[TinnitusStimEvent]:
        """Extract last 2s PPG, run PhaseDetector + tri-fold gate check.

        Returns TinnitusStimEvent if diastole AND exhalation both exceed their
        thresholds AND the ArousalGate is in-band, otherwise None.
        """
        from src.features.ppg_filter import denoise_ppg  # lazy import

        self._fast_path_calls += 1

        # Extract last fast_window_samples from buffer
        buf = list(self._ppg_buffer)
        ppg_window = np.array(buf[-self._fast_window_samples:], dtype=np.float64)

        # Bandpass denoise (Butterworth 0.5–8 Hz)
        ppg_denoised = denoise_ppg(ppg_window, self._ppg_fs, config_path=self._config_path)

        # Build tensor: (1, 1, 250)
        tensor = (
            torch.tensor(ppg_denoised, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self._device)
        )

        # Model forward — output (1, 10, 2) logits
        with torch.no_grad():
            logits = self._phase_detector(tensor)

        # Sigmoid probabilities — (10, 2)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

        # Tri-fold check on last frame
        last_dia_prob = float(probs[-1, 0])
        last_exh_prob = float(probs[-1, 1])

        if last_dia_prob <= self._dia_threshold or last_exh_prob <= self._exh_threshold:
            return None

        # EDA arousal gate check — uncalibrated gate blocks all stims
        try:
            in_band = self._arousal_gate.is_in_band()
        except RuntimeError:
            return None

        if not in_band:
            return None

        return TinnitusStimEvent(
            timestamp_samples=self._total_samples,
            amplitude=self._last_stim_params["amplitude"],
            frequency=self._last_stim_params["frequency"],
            pulse_width=self._last_stim_params["pulse_width"],
            diastole_prob=last_dia_prob,
            exhalation_prob=last_exh_prob,
            arousal_in_band=True,
        )

    def _run_slow_path(self) -> None:
        """Extract last 60s PPG, compute autonomic state, update stim params."""
        from src.features.ppg_filter import denoise_ppg
        from src.features.peak_detector import get_rr_intervals
        from src.features.artifact_scrubber import correct_rr_intervals

        self._slow_path_calls += 1

        buf = list(self._ppg_buffer)
        ppg_60s = np.array(buf[-self._slow_window_samples:], dtype=np.float64)

        try:
            denoised = denoise_ppg(ppg_60s, self._ppg_fs, config_path=self._config_path)
            rr = get_rr_intervals(denoised, self._ppg_fs, signal_type="ppg")

            if len(rr) < 5:
                logger.debug("Slow path: insufficient RR intervals (%d), keeping prior params", len(rr))
                return

            rr_corrected, _ = correct_rr_intervals(rr, config_path=self._config_path)
            if len(rr_corrected) < 5:
                logger.debug("Slow path: too few RR intervals after correction, keeping prior params")
                return

            autonomic = self._autonomic_state.compute(rr_corrected)
            self._last_autonomic_state = autonomic
            self._last_stim_params = self._stim_recommender.recommend(autonomic)
            logger.debug(
                "Slow path: autonomic=%s stim=%s", autonomic, self._last_stim_params
            )
        except Exception as exc:  # pragma: no cover — defensive; log and keep prior params
            logger.warning("Slow path failed, keeping prior stim params: %s", exc)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_tinnitus_closed_loop_pipeline(
    config_path: str = "config_tinnitus.yaml",
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
) -> TinnitusClosedLoopPipeline:
    """Build TinnitusClosedLoopPipeline from config.

    Reads the ``closed_loop`` section of config_tinnitus.yaml for ppg_fs,
    eda_fs, inference_stride_ms, slow_window_sec, and thresholds.

    Args:
        config_path:      Path to tinnitus config YAML.
        checkpoint_path:  Optional PhaseDetector checkpoint path (default: from config).
        device:           PyTorch device string ("cpu" or "cuda").

    Returns:
        Fully wired TinnitusClosedLoopPipeline ready to accept PPG + EDA samples.
    """
    import sys
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from src.models.phase_detector import build_phase_detector
    from src.models.autonomic_state import build_autonomic_state
    from src.models.stim_recommender import build_stim_recommender
    from src.models.arousal_gate import ArousalGate
    from src.training.build_model import load_config

    cfg = load_config(config_path)
    cl_cfg = cfg.get("closed_loop", {})

    ppg_fs = float(cl_cfg.get("ppg_fs", 125.0))
    eda_fs = float(cl_cfg.get("eda_fs", 4.0))
    inference_stride_ms = float(cl_cfg.get("inference_stride_ms", 100.0))
    slow_window_sec = float(cl_cfg.get("slow_window_sec", 60.0))
    diastole_threshold = float(cl_cfg.get("diastole_threshold", 0.5))
    exhalation_threshold = float(cl_cfg.get("exhalation_threshold", 0.5))

    phase_detector = build_phase_detector(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    phase_detector.eval()

    autonomic_state = build_autonomic_state(config_path=config_path)
    stim_recommender = build_stim_recommender(config_path=config_path)
    arousal_gate = ArousalGate(config_path=config_path)

    return TinnitusClosedLoopPipeline(
        phase_detector=phase_detector,
        autonomic_state=autonomic_state,
        stim_recommender=stim_recommender,
        arousal_gate=arousal_gate,
        ppg_fs=ppg_fs,
        eda_fs=eda_fs,
        inference_stride_ms=inference_stride_ms,
        slow_window_sec=slow_window_sec,
        diastole_threshold=diastole_threshold,
        exhalation_threshold=exhalation_threshold,
        config_path=config_path,
    )
