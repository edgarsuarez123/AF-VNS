"""
S-31 — Closed-Loop Inference Pipeline.

Wires PhaseDetector + AutonomicState + StimRecommender into a single synchronous
streaming pipeline.  The caller feeds ECG samples (one or many at a time) and
receives StimEvent objects whenever a diastole + exhalation co-occurrence is detected.

Fast path  (every inference_stride_ms):
    last 2s ECG → denoise → PhaseDetector CNN → co-occurrence check → StimEvent

Slow path  (every slow_window_sec):
    last 60s ECG → AutonomicState.compute_from_ecg() → StimRecommender → update stim params

No threading — the caller controls timing.  Suitable for LSL loops, test harnesses,
and offline replay of recorded ECG.

Usage:
    pipe = build_closed_loop_pipeline(device="cpu")
    events = pipe.feed(ecg_chunk)   # np.ndarray (N,)
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
class StimEvent:
    """A stimulation trigger event."""
    timestamp_samples: int    # Stream sample index when this event fired
    amplitude: float          # mA  (from StimRecommender)
    frequency: float          # Hz
    pulse_width: float        # µs
    diastole_prob: float      # Raw sigmoid probability at trigger frame
    exhalation_prob: float    # Raw sigmoid probability at trigger frame


@dataclass
class PipelineState:
    """Snapshot of pipeline counters and history."""
    total_samples_fed: int
    stim_events: list[StimEvent]
    last_autonomic_state: Optional[dict]
    last_stim_params: Optional[dict]
    fast_path_calls: int
    slow_path_calls: int


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

# Safe stim defaults — used before the first autonomic state update (< 60s data)
_SAFE_STIM_DEFAULTS = {"amplitude": 1.0, "frequency": 10.0, "pulse_width": 50.0}


class ClosedLoopPipeline:
    """Synchronous closed-loop VNS pipeline.

    Args:
        phase_detector:      PhaseDetector CNN (eval mode, on target device)
        autonomic_state:     AutonomicState HRV extractor
        stim_recommender:    StimRecommender rule engine
        fs:                  Expected ECG sampling rate in Hz (default 250)
        inference_stride_ms: Fast-path stride in ms (default 100 → 116ms worst-case latency)
        slow_window_sec:     Slow-path window length in seconds (default 60)
        threshold:           Sigmoid threshold for diastole/exhalation binary decision
        config_path:         Path to config YAML (passed to denoise())
    """

    def __init__(
        self,
        phase_detector,
        autonomic_state,
        stim_recommender,
        fs: float = 250.0,
        inference_stride_ms: float = 100.0,
        slow_window_sec: float = 60.0,
        threshold: float = 0.5,
        config_path: str = "config_stroke.yaml",
    ) -> None:
        self._phase_detector = phase_detector
        self._autonomic_state = autonomic_state
        self._stim_recommender = stim_recommender
        self._fs = float(fs)
        self._threshold = float(threshold)
        self._config_path = config_path

        # Window / stride sizes in samples
        self._fast_window_samples = int(round(2.0 * self._fs))          # 500 @ 250Hz
        self._slow_window_samples = int(round(slow_window_sec * self._fs))  # 15000 @ 250Hz
        self._stride_samples = max(1, int(round(inference_stride_ms / 1000.0 * self._fs)))  # 25 @ 250Hz

        # Ring buffer — maxlen ensures automatic eviction of old data
        self._ecg_buffer: collections.deque = collections.deque(
            maxlen=self._slow_window_samples
        )

        # Counters
        self._samples_since_last_inference: int = 0
        self._samples_since_last_slow: int = 0
        self._total_samples: int = 0

        # State
        self._last_stim_params: dict = dict(_SAFE_STIM_DEFAULTS)
        self._last_autonomic_state: Optional[dict] = None
        self._stim_events: list[StimEvent] = []
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

    def feed(self, samples: np.ndarray) -> list[StimEvent]:
        """Feed ECG samples into the pipeline.

        Args:
            samples: 1-D float array of ECG values at ``self._fs`` Hz.

        Returns:
            List of StimEvent objects triggered during this call (usually empty or one).
        """
        samples = np.asarray(samples, dtype=np.float32).ravel()
        triggered: list[StimEvent] = []

        for s in samples:
            self._ecg_buffer.append(float(s))
            self._total_samples += 1
            self._samples_since_last_inference += 1
            self._samples_since_last_slow += 1

            # Fast path — every stride_samples
            if (self._samples_since_last_inference >= self._stride_samples
                    and len(self._ecg_buffer) >= self._fast_window_samples):
                self._samples_since_last_inference = 0
                event = self._run_fast_path()
                if event is not None:
                    triggered.append(event)
                    self._stim_events.append(event)

            # Slow path — every slow_window_samples
            if (self._samples_since_last_slow >= self._slow_window_samples
                    and len(self._ecg_buffer) >= self._slow_window_samples):
                self._samples_since_last_slow = 0
                self._run_slow_path()

        return triggered

    def reset(self) -> None:
        """Clear all buffers and state. Pipeline returns to initial condition."""
        self._ecg_buffer.clear()
        self._samples_since_last_inference = 0
        self._samples_since_last_slow = 0
        self._total_samples = 0
        self._last_stim_params = dict(_SAFE_STIM_DEFAULTS)
        self._last_autonomic_state = None
        self._stim_events = []
        self._fast_path_calls = 0
        self._slow_path_calls = 0

    def get_state(self) -> PipelineState:
        """Return a snapshot of the current pipeline state."""
        return PipelineState(
            total_samples_fed=self._total_samples,
            stim_events=list(self._stim_events),
            last_autonomic_state=dict(self._last_autonomic_state) if self._last_autonomic_state else None,
            last_stim_params=dict(self._last_stim_params),
            fast_path_calls=self._fast_path_calls,
            slow_path_calls=self._slow_path_calls,
        )

    # ------------------------------------------------------------------
    # Internal paths
    # ------------------------------------------------------------------

    def _run_fast_path(self) -> Optional[StimEvent]:
        """Extract last 2s, run PhaseDetector, check co-occurrence.

        Returns StimEvent if diastole AND exhalation both exceed threshold on
        the last frame, otherwise None.
        """
        from src.features.wavelet_filter import denoise  # lazy import

        self._fast_path_calls += 1

        # Extract last fast_window_samples from buffer
        buf = list(self._ecg_buffer)
        ecg_window = np.array(buf[-self._fast_window_samples:], dtype=np.float64)

        # Wavelet denoise (same as latency benchmark, config_stroke.yaml)
        ecg_denoised = denoise(ecg_window, self._fs, config_path=self._config_path)

        # Build tensor: (1, 1, 500)
        tensor = (
            torch.tensor(ecg_denoised, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self._device)
        )

        # Model forward — output (1, 10, 2) logits
        with torch.no_grad():
            logits = self._phase_detector(tensor)

        # Sigmoid probabilities — (10, 2)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

        # Co-occurrence check on last frame
        last_dia_prob = float(probs[-1, 0])
        last_exh_prob = float(probs[-1, 1])

        if last_dia_prob > self._threshold and last_exh_prob > self._threshold:
            return StimEvent(
                timestamp_samples=self._total_samples,
                amplitude=self._last_stim_params["amplitude"],
                frequency=self._last_stim_params["frequency"],
                pulse_width=self._last_stim_params["pulse_width"],
                diastole_prob=last_dia_prob,
                exhalation_prob=last_exh_prob,
            )
        return None

    def _run_slow_path(self) -> None:
        """Extract last 60s ECG, compute autonomic state, update stim params."""
        self._slow_path_calls += 1

        buf = list(self._ecg_buffer)
        ecg_60s = np.array(buf[-self._slow_window_samples:], dtype=np.float64)

        try:
            autonomic = self._autonomic_state.compute_from_ecg(
                ecg_60s, self._fs, self._config_path
            )
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

def build_closed_loop_pipeline(
    config_path: str = "config_stroke.yaml",
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
) -> ClosedLoopPipeline:
    """Build ClosedLoopPipeline from config.

    Reads ``closed_loop`` section of the config for fs / stride / threshold.
    Uses existing build_* factories for each sub-component.

    Args:
        config_path:      Path to config YAML.
        checkpoint_path:  Optional checkpoint for PhaseDetector (default: from config).
        device:           PyTorch device string ("cpu" or "cuda").

    Returns:
        Fully wired ClosedLoopPipeline ready to accept ECG samples.
    """
    import sys
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from src.models.phase_detector import build_phase_detector
    from src.models.autonomic_state import build_autonomic_state
    from src.models.stim_recommender import build_stim_recommender
    from src.training.build_model import load_config

    cfg = load_config(config_path)
    cl_cfg = cfg.get("closed_loop", {})

    fs = float(cl_cfg.get("fs", 250))
    inference_stride_ms = float(cl_cfg.get("inference_stride_ms", 100))
    slow_window_sec = float(cl_cfg.get("slow_window_sec", 60))
    threshold = float(cl_cfg.get("co_occurrence_threshold", 0.5))

    phase_detector = build_phase_detector(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    phase_detector.eval()

    autonomic_state = build_autonomic_state(config_path=config_path)
    stim_recommender = build_stim_recommender(config_path=config_path)

    return ClosedLoopPipeline(
        phase_detector=phase_detector,
        autonomic_state=autonomic_state,
        stim_recommender=stim_recommender,
        fs=fs,
        inference_stride_ms=inference_stride_ms,
        slow_window_sec=slow_window_sec,
        threshold=threshold,
        config_path=config_path,
    )
