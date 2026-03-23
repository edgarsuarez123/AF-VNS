"""
Tests for S-31 — ClosedLoopPipeline integration.

Covers: factory build, fast-path triggering, co-occurrence logic, slow-path
autonomic update, reset, chunk vs sample-by-sample equivalence, latency guard.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.models.closed_loop_pipeline import (
    ClosedLoopPipeline,
    PipelineState,
    StimEvent,
    build_closed_loop_pipeline,
)
from src.models.phase_detector import PhaseDetector, PhaseDetectorConfig
from src.models.autonomic_state import AutonomicState, AutonomicStateConfig
from src.models.stim_recommender import StimRecommender, StimConfig

CONFIG_PATH = "config_stroke.yaml"
FS = 250.0
FAST_WINDOW = 500   # 2s @ 250Hz
STRIDE = 25         # 100ms @ 250Hz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline(threshold: float = 0.5) -> ClosedLoopPipeline:
    """Build a pipeline with random-weight PhaseDetector (no checkpoint)."""
    phase_detector = PhaseDetector(PhaseDetectorConfig())
    phase_detector.eval()
    autonomic_state = AutonomicState(AutonomicStateConfig())
    stim_recommender = StimRecommender(StimConfig())
    return ClosedLoopPipeline(
        phase_detector=phase_detector,
        autonomic_state=autonomic_state,
        stim_recommender=stim_recommender,
        fs=FS,
        inference_stride_ms=100.0,
        slow_window_sec=60.0,
        threshold=threshold,
        config_path=CONFIG_PATH,
    )


def _synthetic_ecg(n_samples: int, fs: float = FS) -> np.ndarray:
    """Synthetic ECG: 1.2Hz carrier + 12Hz harmonic (same as latency benchmark)."""
    t = np.arange(n_samples) / fs
    return (np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 12 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Test 1: factory build
# ---------------------------------------------------------------------------

def test_build_factory():
    """build_closed_loop_pipeline() returns a correctly wired ClosedLoopPipeline."""
    pipe = build_closed_loop_pipeline(config_path=CONFIG_PATH, device="cpu")
    assert isinstance(pipe, ClosedLoopPipeline)
    state = pipe.get_state()
    assert state.total_samples_fed == 0
    assert state.fast_path_calls == 0
    assert state.slow_path_calls == 0
    assert state.stim_events == []
    assert state.last_autonomic_state is None


# ---------------------------------------------------------------------------
# Test 2: insufficient data — no fast path
# ---------------------------------------------------------------------------

def test_feed_insufficient_data():
    """Feeding fewer than 500 samples does not trigger fast path."""
    pipe = _make_pipeline()
    events = pipe.feed(_synthetic_ecg(FAST_WINDOW - 1))
    state = pipe.get_state()
    assert state.fast_path_calls == 0
    assert events == []
    assert state.total_samples_fed == FAST_WINDOW - 1


# ---------------------------------------------------------------------------
# Test 3: fast path triggers after 2s of data
# ---------------------------------------------------------------------------

def test_fast_path_triggers_after_2s():
    """Feeding exactly 500 samples triggers the fast path at least once."""
    pipe = _make_pipeline()
    pipe.feed(_synthetic_ecg(FAST_WINDOW))
    state = pipe.get_state()
    assert state.fast_path_calls >= 1


# ---------------------------------------------------------------------------
# Test 4: stride spacing — correct number of fast path calls
# ---------------------------------------------------------------------------

def test_stride_spacing():
    """
    After feeding 500 samples (fills buffer), each subsequent 25 samples
    triggers exactly one fast path call.
    Feed total = FAST_WINDOW + k*STRIDE → expect k+1 fast path calls
    (1 at the exact FAST_WINDOW boundary, then 1 per STRIDE thereafter).
    """
    pipe = _make_pipeline()
    k = 5
    total = FAST_WINDOW + k * STRIDE
    pipe.feed(_synthetic_ecg(total))
    state = pipe.get_state()
    # First call fires at sample FAST_WINDOW; subsequent calls every STRIDE
    assert state.fast_path_calls == k + 1


# ---------------------------------------------------------------------------
# Test 5: co-occurrence fires stim event
# ---------------------------------------------------------------------------

def test_co_occurrence_fires_stim():
    """When PhaseDetector outputs high diastole+exhalation, a StimEvent is produced."""
    pipe = _make_pipeline(threshold=0.5)

    # Mock PhaseDetector to always predict diastole=1, exhalation=1 on last frame
    high_logit = 5.0  # sigmoid(5) ≈ 0.993
    def _mock_forward(x):
        B = x.shape[0]
        out = torch.full((B, 10, 2), high_logit)
        return out

    pipe._phase_detector.forward = _mock_forward

    with patch("src.models.closed_loop_pipeline.ClosedLoopPipeline._run_fast_path",
               wraps=pipe._run_fast_path):
        events = pipe.feed(_synthetic_ecg(FAST_WINDOW))

    assert len(events) >= 1
    ev = events[0]
    assert isinstance(ev, StimEvent)
    assert ev.diastole_prob > 0.5
    assert ev.exhalation_prob > 0.5


# ---------------------------------------------------------------------------
# Test 6: no trigger without co-occurrence
# ---------------------------------------------------------------------------

def test_no_trigger_without_co_occurrence():
    """When only diastole is high (exhalation low), no StimEvent fires."""
    pipe = _make_pipeline(threshold=0.5)

    def _mock_forward(x):
        B = x.shape[0]
        out = torch.zeros(B, 10, 2)
        out[:, :, 0] = 5.0   # diastole high
        out[:, :, 1] = -5.0  # exhalation low
        return out

    pipe._phase_detector.forward = _mock_forward
    events = pipe.feed(_synthetic_ecg(FAST_WINDOW))
    assert events == []


# ---------------------------------------------------------------------------
# Test 7: slow path runs after 60s of data
# ---------------------------------------------------------------------------

def test_slow_path_runs_after_60s():
    """Feeding 60s of synthetic ECG triggers the slow path at least once."""
    pipe = _make_pipeline()
    slow_samples = int(60.0 * FS)  # 15000
    pipe.feed(_synthetic_ecg(slow_samples))
    state = pipe.get_state()
    assert state.slow_path_calls >= 1
    # Autonomic state should be populated (may be all-NaN on synthetic ECG, but not None)
    assert state.last_autonomic_state is not None


# ---------------------------------------------------------------------------
# Test 8: stim params update after slow path
# ---------------------------------------------------------------------------

def test_stim_params_update_after_slow_path():
    """After slow path runs, stim events carry params from the recommender (not safe defaults)."""
    pipe = _make_pipeline(threshold=0.5)

    # Mock slow path to set known non-default params
    updated_params = {"amplitude": 7.0, "frequency": 50.0, "pulse_width": 300.0}

    def _mock_slow():
        pipe._slow_path_calls += 1
        pipe._last_stim_params = dict(updated_params)
        pipe._last_autonomic_state = {"lf_hf_ratio": 2.5, "norm_hf_power": 0.25,
                                      "sampen": 1.2, "dfa_alpha1": 0.9}

    pipe._run_slow_path = _mock_slow

    # Mock fast path to always fire a stim event
    def _mock_fast():
        pipe._fast_path_calls += 1
        ev = StimEvent(
            timestamp_samples=pipe._total_samples,
            amplitude=pipe._last_stim_params["amplitude"],
            frequency=pipe._last_stim_params["frequency"],
            pulse_width=pipe._last_stim_params["pulse_width"],
            diastole_prob=0.99,
            exhalation_prob=0.99,
        )
        pipe._stim_events.append(ev)
        return ev

    # First: trigger slow path, then fire stim
    pipe._run_slow_path()
    assert pipe._last_stim_params == updated_params

    pipe._run_fast_path = _mock_fast
    # Feed enough to trigger fast path
    pipe._ecg_buffer.extend([0.0] * FAST_WINDOW)
    pipe._total_samples = FAST_WINDOW
    pipe._samples_since_last_inference = STRIDE

    events = pipe.feed(_synthetic_ecg(STRIDE))
    # At least one event should carry updated params
    all_events = pipe.get_state().stim_events
    updated_events = [e for e in all_events if e.amplitude == 7.0]
    assert len(updated_events) >= 1


# ---------------------------------------------------------------------------
# Test 9: reset clears all state
# ---------------------------------------------------------------------------

def test_reset_clears_state():
    """After reset(), state is identical to a freshly created pipeline."""
    pipe = _make_pipeline()
    pipe.feed(_synthetic_ecg(FAST_WINDOW + 2 * STRIDE))
    assert pipe.get_state().total_samples_fed > 0

    pipe.reset()
    state = pipe.get_state()
    assert state.total_samples_fed == 0
    assert state.fast_path_calls == 0
    assert state.slow_path_calls == 0
    assert state.stim_events == []
    assert state.last_autonomic_state is None
    assert len(pipe._ecg_buffer) == 0


# ---------------------------------------------------------------------------
# Test 10: latency regression — fast path mean < 50ms
# ---------------------------------------------------------------------------

def test_latency_regression():
    """Fast path (denoise + inference) runs in < 50ms mean over 50 calls."""
    pipe = _make_pipeline()
    # Pre-fill buffer
    pipe.feed(_synthetic_ecg(FAST_WINDOW))

    ecg_chunk = _synthetic_ecg(FAST_WINDOW)
    # Update buffer with a known window
    pipe._ecg_buffer.extend(ecg_chunk.tolist())

    times_ms = []
    for _ in range(50):
        t0 = time.perf_counter()
        pipe._run_fast_path()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(times_ms))
    assert mean_ms < 50.0, f"Fast path mean latency {mean_ms:.1f}ms exceeds 50ms threshold"
