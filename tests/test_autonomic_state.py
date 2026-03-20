"""Tests for autonomic_state.py — sliding-window HRV feature extractor (S-24)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rr_60s(hr_bpm: float = 72.0, n_beats: int = 72, seed: int = 42) -> np.ndarray:
    """Synthetic RR intervals (~60s window) with slight variability."""
    rng = np.random.default_rng(seed)
    mean_rr = 60.0 / hr_bpm
    return rng.normal(loc=mean_rr, scale=0.02, size=n_beats).clip(0.3, 2.0)


def _make_config(tmp_dir: Path) -> Path:
    cfg = """
autonomic_state:
  window_sec: 60
  features: [lf_hf_ratio, norm_hf_power, sampen, dfa_alpha1]
  fs_resample: 4.0

data:
  target_fs: 250

artifact:
  amplitude_mad_multiple: 30
  rr_deviation_percent: 60
  rr_fraction_threshold: 0.30

hrv:
  window_sec: [30, 300]
  subwindow_sec: 60

wavelet:
  family: "cmor"
  scale_range: [1, 64]
"""
    p = tmp_dir / "config_auto_test.yaml"
    p.write_text(cfg)
    return p


# ---------------------------------------------------------------------------
# Tests — norm_hf_power in hrv_freq
# ---------------------------------------------------------------------------

class TestNormHfPower:

    def test_norm_hf_power_added(self):
        """compute_hrv_freq() returns norm_hf_power key."""
        from src.features.hrv_freq import compute_hrv_freq
        result = compute_hrv_freq(_rr_60s())
        assert "norm_hf_power" in result

    def test_norm_hf_power_range(self):
        """nHF ∈ [0, 1] for valid RR."""
        from src.features.hrv_freq import compute_hrv_freq
        result = compute_hrv_freq(_rr_60s())
        nhf = result["norm_hf_power"]
        assert not np.isnan(nhf), "Expected finite nHF for 72-beat window"
        assert 0.0 <= nhf <= 1.0, f"nHF={nhf} out of [0,1]"

    def test_norm_hf_power_nan_short(self):
        """Returns NaN when insufficient RR."""
        from src.features.hrv_freq import compute_hrv_freq
        result = compute_hrv_freq(np.array([0.8, 0.9]))  # only 2 intervals
        assert np.isnan(result["norm_hf_power"])


# ---------------------------------------------------------------------------
# Tests — AutonomicState
# ---------------------------------------------------------------------------

class TestAutonomicState:

    def test_compute_from_rr(self):
        """compute() returns dict with all 4 features."""
        from src.models.autonomic_state import AutonomicState, AutonomicStateConfig
        auto = AutonomicState(AutonomicStateConfig())
        state = auto.compute(_rr_60s())
        assert set(state.keys()) == {"lf_hf_ratio", "norm_hf_power", "sampen", "dfa_alpha1"}

    def test_as_vector_shape(self):
        """as_vector() returns (4,) float32 array."""
        from src.models.autonomic_state import AutonomicState, AutonomicStateConfig
        auto = AutonomicState(AutonomicStateConfig())
        state = auto.compute(_rr_60s())
        vec = auto.as_vector(state)
        assert vec.shape == (4,)
        assert vec.dtype == np.float32

    def test_nan_handling_short_rr(self):
        """Short RR → all NaN state for freq features."""
        from src.models.autonomic_state import AutonomicState, AutonomicStateConfig
        auto = AutonomicState(AutonomicStateConfig())
        state = auto.compute(np.array([0.8]))  # 1 interval → insufficient
        assert np.isnan(state["lf_hf_ratio"])
        assert np.isnan(state["norm_hf_power"])

    def test_compute_from_ecg(self, tmp_path):
        """compute_from_ecg() with synthetic 60s ECG produces valid output."""
        import neurokit2 as nk
        from src.models.autonomic_state import AutonomicState, AutonomicStateConfig

        ecg = nk.ecg_simulate(duration=60, sampling_rate=250, heart_rate=72)
        config_path = _make_config(tmp_path)

        auto = AutonomicState(AutonomicStateConfig())
        state = auto.compute_from_ecg(np.array(ecg), fs=250.0, config_path=str(config_path))
        assert isinstance(state, dict)
        assert len(state) == 4
        # At least lf_hf_ratio and norm_hf_power should be finite for clean 60s ECG
        assert np.isfinite(state["lf_hf_ratio"]), f"lf_hf_ratio={state['lf_hf_ratio']}"

    def test_build_factory(self):
        """build_autonomic_state() loads from real config."""
        from src.models.autonomic_state import build_autonomic_state
        auto = build_autonomic_state("config_stroke.yaml")
        assert auto.cfg.window_sec == 60.0
        assert len(auto.cfg.features) == 4
