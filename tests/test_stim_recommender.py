"""Tests for stim_recommender.py — rule-based stim parameter mapping (S-24)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normal_state() -> dict[str, float]:
    """Healthy autonomic state — all features in normal range."""
    return {
        "lf_hf_ratio": 1.0,       # balanced ANS
        "norm_hf_power": 0.5,      # good vagal tone
        "sampen": 1.5,             # moderate complexity
        "dfa_alpha1": 1.0,         # normal fractal scaling
    }


def _sympathetic_state() -> dict[str, float]:
    """Sympathetic-dominant state."""
    return {
        "lf_hf_ratio": 4.0,       # high — sympathetic
        "norm_hf_power": 0.15,     # low — poor vagal tone
        "sampen": 1.5,
        "dfa_alpha1": 1.0,
    }


def _nan_state() -> dict[str, float]:
    """All features unavailable."""
    return {
        "lf_hf_ratio": float("nan"),
        "norm_hf_power": float("nan"),
        "sampen": float("nan"),
        "dfa_alpha1": float("nan"),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStimRecommender:

    def test_recommend_output_keys(self):
        """Returns amplitude, frequency, pulse_width."""
        from src.models.stim_recommender import StimRecommender, StimConfig
        sr = StimRecommender(StimConfig())
        params = sr.recommend(_normal_state())
        assert set(params.keys()) == {"amplitude", "frequency", "pulse_width"}

    def test_recommend_clipped(self):
        """All outputs within hardware limits."""
        from src.models.stim_recommender import StimRecommender, StimConfig
        cfg = StimConfig()
        sr = StimRecommender(cfg)

        for state in [_normal_state(), _sympathetic_state(), _nan_state()]:
            params = sr.recommend(state)
            assert cfg.amplitude_range[0] <= params["amplitude"] <= cfg.amplitude_range[1]
            assert cfg.frequency_range[0] <= params["frequency"] <= cfg.frequency_range[1]
            assert cfg.pulse_width_range[0] <= params["pulse_width"] <= cfg.pulse_width_range[1]

    def test_high_lf_hf_reduces_amp(self):
        """Sympathetic dominance → lower amplitude than normal."""
        from src.models.stim_recommender import StimRecommender, StimConfig
        sr = StimRecommender(StimConfig())
        normal = sr.recommend(_normal_state())
        sympathetic = sr.recommend(_sympathetic_state())
        assert sympathetic["amplitude"] < normal["amplitude"], (
            f"Sympathetic amp ({sympathetic['amplitude']}) should be < normal ({normal['amplitude']})"
        )

    def test_low_nhf_increases_freq(self):
        """Poor vagal tone → higher frequency than normal."""
        from src.models.stim_recommender import StimRecommender, StimConfig
        sr = StimRecommender(StimConfig())
        normal = sr.recommend(_normal_state())
        sympathetic = sr.recommend(_sympathetic_state())  # nHF=0.15 < 0.3 threshold
        assert sympathetic["frequency"] > normal["frequency"], (
            f"Low nHF freq ({sympathetic['frequency']}) should be > normal ({normal['frequency']})"
        )

    def test_all_nan_safe_defaults(self):
        """All NaN → conservative output."""
        from src.models.stim_recommender import StimRecommender, StimConfig
        sr = StimRecommender(StimConfig())
        params = sr.recommend(_nan_state())
        assert params["amplitude"] <= 2.0, f"NaN should give low amplitude, got {params['amplitude']}"
        assert params["frequency"] <= 15.0, f"NaN should give low frequency, got {params['frequency']}"

    def test_normal_state_midrange(self):
        """Healthy autonomic state → midpoint-ish parameters."""
        from src.models.stim_recommender import StimRecommender, StimConfig
        sr = StimRecommender(StimConfig())
        params = sr.recommend(_normal_state())
        # Normal state should produce values near the midpoint (5mA, 25Hz, 250µs)
        assert 3.0 <= params["amplitude"] <= 7.0
        assert 15.0 <= params["frequency"] <= 35.0
        assert 200.0 <= params["pulse_width"] <= 300.0

    def test_build_factory(self):
        """build_stim_recommender() loads from real config."""
        from src.models.stim_recommender import build_stim_recommender
        sr = build_stim_recommender("config_stroke.yaml")
        assert sr.cfg.amplitude_range == (0.0, 10.0)
        assert sr.cfg.lf_hf_high == 2.0

    def test_extreme_values(self):
        """Very large/small inputs → still clipped to hardware limits."""
        from src.models.stim_recommender import StimRecommender, StimConfig
        cfg = StimConfig()
        sr = StimRecommender(cfg)
        extreme = {
            "lf_hf_ratio": 100.0,
            "norm_hf_power": 0.0,
            "sampen": 50.0,
            "dfa_alpha1": 10.0,
        }
        params = sr.recommend(extreme)
        assert params["amplitude"] >= cfg.amplitude_range[0]
        assert params["frequency"] <= cfg.frequency_range[1]
        assert params["pulse_width"] <= cfg.pulse_width_range[1]
