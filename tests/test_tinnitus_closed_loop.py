"""
Tests for F12 — TinnitusClosedLoopPipeline (tri-fold trigger: diastole + exhalation + EDA).

Covers: factory build, PPG buffering, fast-path stride, tri-fold gate logic
(all 3 combinations that block), slow-path autonomic update, reset, latency guard.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.models.tinnitus_closed_loop import (
    TinnitusClosedLoopPipeline,
    TinnitusPipelineState,
    TinnitusStimEvent,
    build_tinnitus_closed_loop_pipeline,
)
from src.models.phase_detector import PhaseDetector, PhaseDetectorConfig
from src.models.autonomic_state import AutonomicState, AutonomicStateConfig
from src.models.stim_recommender import StimRecommender, StimConfig
from src.models.arousal_gate import ArousalGate

CONFIG_PATH = "config_tinnitus.yaml"
PPG_FS = 125.0
EDA_FS = 4.0
FAST_WINDOW = 250   # 2s @ 125 Hz
STRIDE = 13         # 100ms @ 125 Hz  (round(100/1000 * 125) = 13)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline(
    diastole_threshold: float = 0.5,
    exhalation_threshold: float = 0.5,
) -> TinnitusClosedLoopPipeline:
    """Build a pipeline with random-weight PhaseDetector and fresh ArousalGate."""
    phase_detector = PhaseDetector(PhaseDetectorConfig(input_samples=250))
    phase_detector.eval()
    autonomic_state = AutonomicState(AutonomicStateConfig())
    stim_recommender = StimRecommender(StimConfig())
    arousal_gate = ArousalGate(config_path=CONFIG_PATH)
    return TinnitusClosedLoopPipeline(
        phase_detector=phase_detector,
        autonomic_state=autonomic_state,
        stim_recommender=stim_recommender,
        arousal_gate=arousal_gate,
        ppg_fs=PPG_FS,
        eda_fs=EDA_FS,
        inference_stride_ms=100.0,
        slow_window_sec=60.0,
        diastole_threshold=diastole_threshold,
        exhalation_threshold=exhalation_threshold,
        config_path=CONFIG_PATH,
    )


def _synthetic_ppg(n_samples: int, fs: float = PPG_FS) -> np.ndarray:
    """Synthetic PPG: 1.2 Hz carrier + 12 Hz harmonic."""
    t = np.arange(n_samples) / fs
    return (np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 12 * t)).astype(np.float32)


def _flat_eda(value: float = 2.0, n_samples: int = 480) -> np.ndarray:
    """Flat synthetic EDA signal (120s at 4 Hz) for ArousalGate calibration."""
    return np.full(n_samples, value, dtype=np.float64)


def _calibrate_gate_in_band(pipe: TinnitusClosedLoopPipeline) -> None:
    """Calibrate and force the arousal gate into the in-band state."""
    baseline = _flat_eda(2.0, n_samples=4800)  # 1200s at 4 Hz
    pipe.calibrate_eda(baseline, fs=EDA_FS)
    # The flat signal sits exactly at cal_mean; in-band requires an update to set _in_band=True
    pipe._arousal_gate.update(_flat_eda(2.0, n_samples=8), fs=EDA_FS)


def _calibrate_gate_out_of_band(pipe: TinnitusClosedLoopPipeline) -> None:
    """Calibrate gate at 2.0 µS then push a high-arousal value out of band."""
    baseline = _flat_eda(2.0, n_samples=4800)
    pipe.calibrate_eda(baseline, fs=EDA_FS)
    # Push tonic well above high_threshold_sigma (2.5 std above mean)
    pipe._arousal_gate.update(_flat_eda(100.0, n_samples=240), fs=EDA_FS)


# ---------------------------------------------------------------------------
# Test 1: factory build
# ---------------------------------------------------------------------------

def test_factory_build():
    """build_tinnitus_closed_loop_pipeline() returns correctly wired pipeline."""
    pipe = build_tinnitus_closed_loop_pipeline(config_path=CONFIG_PATH, device="cpu")
    assert isinstance(pipe, TinnitusClosedLoopPipeline)
    state = pipe.get_state()
    assert state.total_samples_fed == 0
    assert state.fast_path_calls == 0
    assert state.slow_path_calls == 0
    assert state.stim_events == []
    assert state.last_autonomic_state is None
    assert state.eda_calibrated is False


def test_factory_loads_arousal_classifier():
    """Factory wires trained ArousalClassifier into ArousalGate when checkpoint exists."""
    pipe = build_tinnitus_closed_loop_pipeline(config_path=CONFIG_PATH, device="cpu")
    gate = pipe._arousal_gate
    assert gate._classifier is not None, "Classifier should be loaded when checkpoint exists"
    assert gate._classifier._is_fitted is True


def test_factory_rule_based_when_classifier_disabled(tmp_path):
    """Factory uses rule-based gate when use_arousal_classifier is false."""
    import yaml

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["closed_loop"]["use_arousal_classifier"] = False
    tmp_config = tmp_path / "config_no_clf.yaml"
    with open(tmp_config, "w") as f:
        yaml.dump(cfg, f)

    pipe = build_tinnitus_closed_loop_pipeline(config_path=str(tmp_config), device="cpu")
    assert pipe._arousal_gate._classifier is None


# ---------------------------------------------------------------------------
# Test 2: PPG buffer accumulates correctly
# ---------------------------------------------------------------------------

def test_feed_ppg_buffers():
    """PPG samples accumulate in _ppg_buffer; total_samples_fed tracks correctly."""
    pipe = _make_pipeline()
    n = FAST_WINDOW - 1  # below threshold — no fast path
    events = pipe.feed(_synthetic_ppg(n))
    state = pipe.get_state()
    assert state.total_samples_fed == n
    assert state.fast_path_calls == 0
    assert events == []
    assert len(pipe._ppg_buffer) == n


# ---------------------------------------------------------------------------
# Test 3: fast-path stride fires at correct intervals
# ---------------------------------------------------------------------------

def test_fast_path_stride():
    """After 2s of data, one fast-path call per STRIDE thereafter."""
    pipe = _make_pipeline()
    k = 5
    total = FAST_WINDOW + k * STRIDE
    pipe.feed(_synthetic_ppg(total))
    state = pipe.get_state()
    # First call fires when buffer reaches FAST_WINDOW; then once per STRIDE
    assert state.fast_path_calls == k + 1


# ---------------------------------------------------------------------------
# Test 4: tri-fold fires when all 3 conditions met
# ---------------------------------------------------------------------------

def test_trifold_fires():
    """All three gates open (high dia, high exh, in-band EDA) → StimEvent fires."""
    pipe = _make_pipeline(diastole_threshold=0.5, exhalation_threshold=0.5)

    # Mock PhaseDetector: both heads always output high logit
    high_logit = 5.0  # sigmoid(5) ≈ 0.993
    def _mock_forward(x):
        B = x.shape[0]
        return torch.full((B, 10, 2), high_logit)

    pipe._phase_detector.forward = _mock_forward
    _calibrate_gate_in_band(pipe)

    events = pipe.feed(_synthetic_ppg(FAST_WINDOW))
    assert len(events) >= 1
    ev = events[0]
    assert isinstance(ev, TinnitusStimEvent)
    assert ev.diastole_prob > 0.5
    assert ev.exhalation_prob > 0.5
    assert ev.arousal_in_band is True


# ---------------------------------------------------------------------------
# Test 5: tri-fold blocked by EDA gate (out-of-band)
# ---------------------------------------------------------------------------

def test_trifold_blocked_by_eda():
    """High dia + high exh but EDA out-of-band → no StimEvent."""
    pipe = _make_pipeline()

    def _mock_forward(x):
        B = x.shape[0]
        return torch.full((B, 10, 2), 5.0)

    pipe._phase_detector.forward = _mock_forward
    _calibrate_gate_out_of_band(pipe)

    events = pipe.feed(_synthetic_ppg(FAST_WINDOW))
    assert events == []


# ---------------------------------------------------------------------------
# Test 6: tri-fold blocked by phase gate (low exhalation)
# ---------------------------------------------------------------------------

def test_trifold_blocked_by_phase():
    """High dia but low exh (even with EDA in-band) → no StimEvent."""
    pipe = _make_pipeline()

    def _mock_forward(x):
        B = x.shape[0]
        out = torch.zeros(B, 10, 2)
        out[:, :, 0] = 5.0    # diastole high
        out[:, :, 1] = -5.0   # exhalation low
        return out

    pipe._phase_detector.forward = _mock_forward
    _calibrate_gate_in_band(pipe)

    events = pipe.feed(_synthetic_ppg(FAST_WINDOW))
    assert events == []


# ---------------------------------------------------------------------------
# Test 7: uncalibrated EDA gate blocks stim
# ---------------------------------------------------------------------------

def test_eda_not_calibrated_blocks_stim():
    """Fast path with uncalibrated ArousalGate never fires StimEvent."""
    pipe = _make_pipeline()

    def _mock_forward(x):
        B = x.shape[0]
        return torch.full((B, 10, 2), 5.0)

    pipe._phase_detector.forward = _mock_forward
    # Do NOT calibrate gate
    assert pipe.get_state().eda_calibrated is False

    events = pipe.feed(_synthetic_ppg(FAST_WINDOW))
    assert events == []


# ---------------------------------------------------------------------------
# Test 8: slow path triggers after 60s and populates autonomic state
# ---------------------------------------------------------------------------

def test_slow_path_triggers():
    """Feeding 60s of PPG triggers slow path; last_autonomic_state is set."""
    pipe = _make_pipeline()
    slow_samples = int(60.0 * PPG_FS)  # 7500
    pipe.feed(_synthetic_ppg(slow_samples))
    state = pipe.get_state()
    assert state.slow_path_calls >= 1
    # Autonomic state populated (may be all-NaN on synthetic PPG, but not None)
    assert state.last_autonomic_state is not None


# ---------------------------------------------------------------------------
# Test 9: reset clears all state
# ---------------------------------------------------------------------------

def test_reset_clears_state():
    """After reset(), pipeline state is identical to a freshly created pipeline."""
    pipe = _make_pipeline()
    _calibrate_gate_in_band(pipe)
    pipe.feed(_synthetic_ppg(FAST_WINDOW + 2 * STRIDE))
    assert pipe.get_state().total_samples_fed > 0

    pipe.reset()
    state = pipe.get_state()
    assert state.total_samples_fed == 0
    assert state.fast_path_calls == 0
    assert state.slow_path_calls == 0
    assert state.stim_events == []
    assert state.last_autonomic_state is None
    assert len(pipe._ppg_buffer) == 0


# ---------------------------------------------------------------------------
# Test 10: latency regression — fast path mean < 50ms
# ---------------------------------------------------------------------------

def test_latency_regression():
    """Fast path (denoise + inference) runs in < 50ms mean over 50 calls."""
    pipe = _make_pipeline()
    # Pre-fill buffer with 2s of PPG
    pipe.feed(_synthetic_ppg(FAST_WINDOW))
    pipe._ppg_buffer.extend(_synthetic_ppg(FAST_WINDOW).tolist())

    times_ms = []
    for _ in range(50):
        t0 = time.perf_counter()
        pipe._run_fast_path()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(times_ms))
    assert mean_ms < 50.0, f"Fast path mean latency {mean_ms:.1f}ms exceeds 50ms threshold"
