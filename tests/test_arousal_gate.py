"""Tests for ArousalGate streaming EDA arousal gate.

Tests 1–8 per plan specification.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.arousal_gate import ArousalGate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "config_tinnitus.yaml")
FS = 4.0  # Hz — Empatica E4 EDA rate


def _flat_signal(value: float, duration_sec: float = 120.0, fs: float = FS) -> np.ndarray:
    """Constant EDA signal at given value."""
    return np.full(int(duration_sec * fs), value, dtype=np.float64)


def _make_gate() -> ArousalGate:
    return ArousalGate(config_path=CONFIG_PATH)


# ---------------------------------------------------------------------------
# 1. calibrate() sets thresholds
# ---------------------------------------------------------------------------

def test_calibrate_sets_thresholds():
    gate = _make_gate()
    assert not gate._calibrated

    signal = _flat_signal(1.0, duration_sec=120.0)
    gate.calibrate(signal, FS)

    assert gate._calibrated
    assert not math.isnan(gate._cal_mean)
    assert not math.isnan(gate._cal_std)
    assert gate._cal_std > 0
    assert gate._low_thresh < gate._cal_mean
    assert gate._high_thresh > gate._cal_mean


# ---------------------------------------------------------------------------
# 2. Flat signal at cal_mean → is_in_band() True
# ---------------------------------------------------------------------------

def test_baseline_in_band():
    gate = _make_gate()
    signal = _flat_signal(1.0, duration_sec=120.0)
    gate.calibrate(signal, FS)

    # Update with same-level signal — should be in band
    gate.update(_flat_signal(1.0, duration_sec=30.0), FS)
    assert gate.is_in_band() is True


# ---------------------------------------------------------------------------
# 3. Signal above high threshold → is_in_band() False
# ---------------------------------------------------------------------------

def test_elevated_out_of_band():
    gate = _make_gate()
    signal = _flat_signal(1.0, duration_sec=120.0)
    gate.calibrate(signal, FS)

    # Push far above high threshold
    high_value = gate._cal_mean + (gate._high_sigma + 5.0) * gate._cal_std + 10.0
    gate.update(_flat_signal(high_value, duration_sec=30.0), FS)
    assert gate.is_in_band() is False


# ---------------------------------------------------------------------------
# 4. Signal below low threshold → is_in_band() False
# ---------------------------------------------------------------------------

def test_low_out_of_band():
    gate = _make_gate()
    signal = _flat_signal(2.0, duration_sec=120.0)
    gate.calibrate(signal, FS)

    # Push far below low threshold; EDA can't go negative so use near-zero
    low_value = gate._cal_mean - (gate._low_sigma + 5.0) * gate._cal_std - 10.0
    # Clamp to 0 in case value goes negative (EDA is non-negative in practice,
    # but decompose_eda handles it; the gate logic is what matters)
    gate.update(_flat_signal(max(0.0, low_value), duration_sec=30.0), FS)
    assert gate.is_in_band() is False


# ---------------------------------------------------------------------------
# 5. get_state() has all expected keys
# ---------------------------------------------------------------------------

def test_get_state_keys():
    gate = _make_gate()
    gate.calibrate(_flat_signal(1.0), FS)
    gate.update(_flat_signal(1.0, duration_sec=10.0), FS)

    state = gate.get_state()
    expected_keys = {"in_band", "tonic_scl", "low_thresh", "high_thresh", "calibrated"}
    assert expected_keys.issubset(state.keys())
    assert state["calibrated"] is True
    assert isinstance(state["in_band"], bool)


# ---------------------------------------------------------------------------
# 6. is_in_band() before calibrate() raises RuntimeError
# ---------------------------------------------------------------------------

def test_uncalibrated_raises():
    gate = _make_gate()
    with pytest.raises(RuntimeError, match="calibrate"):
        gate.is_in_band()


# ---------------------------------------------------------------------------
# 7. Multiple update() calls maintain state
# ---------------------------------------------------------------------------

def test_update_streaming():
    gate = _make_gate()
    gate.calibrate(_flat_signal(1.0, duration_sec=120.0), FS)

    # Multiple small updates — gate should remain callable after each
    for _ in range(5):
        gate.update(_flat_signal(1.0, duration_sec=5.0), FS)

    assert gate._calibrated
    result = gate.is_in_band()
    assert isinstance(result, bool)
    assert not math.isnan(gate._tonic_scl)


# ---------------------------------------------------------------------------
# 8. Config values match YAML
# ---------------------------------------------------------------------------

def test_config_loaded():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    eda_cfg = cfg["eda"]

    gate = _make_gate()
    assert gate._low_sigma == float(eda_cfg["low_threshold_sigma"])
    assert gate._high_sigma == float(eda_cfg["high_threshold_sigma"])
    assert gate._frame_rate_hz == float(eda_cfg["frame_rate_hz"])
    assert gate._calibration_sec == float(eda_cfg["calibration_sec"])


# ---------------------------------------------------------------------------
# F19: skin temperature support
# ---------------------------------------------------------------------------

def test_update_with_temp_samples_produces_6_features():
    """update() with temp_samples appends skin_temp_mean as 6th feature."""
    gate = _make_gate()
    gate.calibrate(_flat_signal(1.0, duration_sec=120.0), FS)

    eda = _flat_signal(1.0, duration_sec=30.0)
    temp = np.full(len(eda), 33.5, dtype=np.float64)
    gate.update(eda, FS, temp_samples=temp)

    assert gate._current_features is not None
    assert len(gate._current_features) == 6
    assert abs(float(gate._current_features[5]) - 33.5) < 0.1


def test_update_without_temp_samples_produces_5_features():
    """update() without temp_samples returns 5-feature vector (backward compat)."""
    gate = _make_gate()
    gate.calibrate(_flat_signal(1.0, duration_sec=120.0), FS)

    gate.update(_flat_signal(1.0, duration_sec=30.0), FS)

    assert gate._current_features is not None
    assert len(gate._current_features) == 5


def test_temp_buffer_accumulates_and_trims():
    """Temperature ring buffer grows up to buffer capacity and does not exceed it."""
    gate = ArousalGate(config_path=CONFIG_PATH, buffer_sec=10.0)

    # Send more than buffer_sec worth of data
    eda = _flat_signal(1.0, duration_sec=15.0)
    temp = np.linspace(30.0, 35.0, len(eda))
    gate.update(eda, FS, temp_samples=temp)

    # Buffer should be capped at capacity
    expected_cap = int(10.0 * FS)
    assert len(gate._temp_buffer) <= expected_cap


def test_update_with_temp_none_keeps_5_features():
    """Passing temp_samples=None explicitly keeps 5-feature vector."""
    gate = _make_gate()
    gate.calibrate(_flat_signal(1.0, duration_sec=120.0), FS)

    gate.update(_flat_signal(1.0, duration_sec=30.0), FS, temp_samples=None)

    assert gate._current_features is not None
    assert len(gate._current_features) == 5


# ---------------------------------------------------------------------------
# F21: sliding recalibration
# ---------------------------------------------------------------------------

def _make_gate_recal(alpha: float = 0.3) -> ArousalGate:
    """Build ArousalGate with sliding recalibration config."""
    import yaml
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["eda"]["sliding_recalibration"] = True
    cfg["eda"]["recalibration_window_sec"] = 30   # short window for testing
    cfg["eda"]["recalibration_interval_sec"] = 10  # short interval for testing
    cfg["eda"]["recalibration_alpha"] = alpha

    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        tmp_path = f.name
    try:
        gate = ArousalGate(config_path=tmp_path)
    finally:
        os.unlink(tmp_path)
    return gate


def test_sliding_recal_shifts_baseline():
    """After enough samples, sliding recalibration shifts cal_mean toward new baseline."""
    gate = _make_gate_recal(alpha=0.3)

    baseline = _flat_signal(1.0, duration_sec=120.0)
    gate.calibrate(baseline, FS)
    original_mean = gate._cal_mean

    # Feed signal at a higher level for 2 minutes → recalibration should shift mean up
    high_signal = _flat_signal(3.0, duration_sec=120.0)
    gate.update(high_signal, FS)

    # cal_mean should have drifted toward 3.0
    assert gate._cal_mean > original_mean, (
        f"Expected cal_mean to shift up after recal, got {gate._cal_mean:.4f} vs original {original_mean:.4f}"
    )


def test_sliding_recal_disabled_keeps_baseline_fixed():
    """When sliding_recalibration=False, cal_mean stays fixed regardless of new data."""
    gate = _make_gate()  # uses config as-is (sliding_recalibration may be true in config)
    gate._sliding_recal = False  # force off

    baseline = _flat_signal(1.0, duration_sec=120.0)
    gate.calibrate(baseline, FS)
    original_mean = gate._cal_mean

    gate.update(_flat_signal(5.0, duration_sec=300.0), FS)

    assert gate._cal_mean == original_mean, (
        f"cal_mean should not change when recal disabled, got {gate._cal_mean:.4f}"
    )


def test_sliding_recal_ema_blend_math():
    """Verify EMA blend: new_mean = (1-α)*old + α*new."""
    gate = _make_gate_recal(alpha=0.5)

    # Calibrate at 2.0 µS (flat → std≈0, but enough to set cal_mean)
    gate.calibrate(_flat_signal(2.0, duration_sec=120.0), FS)
    original_mean = gate._cal_mean

    # Force a single recalibration step by directly calling the internals
    gate._recal_buffer_capacity = 1000
    gate._recal_buffer.extend(_flat_signal(4.0, duration_sec=250.0).tolist())
    gate._samples_since_recal = int(gate._recal_interval_sec * FS) + 1
    # Trigger recalibration via update() with a short chunk
    gate.update(_flat_signal(4.0, duration_sec=1.0), FS)

    # Mean should be between original (2.0) and 4.0
    assert gate._cal_mean > original_mean
    assert gate._cal_mean < 4.0, f"cal_mean {gate._cal_mean:.4f} should be blended, not fully replaced"


def test_sliding_recal_get_state_includes_cal_fields():
    """get_state() includes cal_mean, cal_std, and sliding_recalibration fields."""
    gate = _make_gate()
    gate.calibrate(_flat_signal(1.0, duration_sec=120.0), FS)

    state = gate.get_state()
    assert "sliding_recalibration" in state
    assert "cal_mean" in state
    assert "cal_std" in state
    assert isinstance(state["sliding_recalibration"], bool)
