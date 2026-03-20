"""
Stim Parameter Recommender — maps autonomic state to stimulation parameters (FR-2.4).

Input:  autonomic state dict {lf_hf_ratio, norm_hf_power, sampen, dfa_alpha1}
Output: stim parameters {amplitude (mA), frequency (Hz), pulse_width (µs)}

Rule-based initially for clinical auditability. ML version deferred to future phase.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StimConfig:
    amplitude_range: tuple[float, float] = (0.0, 10.0)     # mA
    frequency_range: tuple[float, float] = (1.0, 100.0)    # Hz
    pulse_width_range: tuple[int, int] = (50, 500)          # µs
    lf_hf_high: float = 2.0        # threshold for sympathetic dominance
    norm_hf_low: float = 0.3       # threshold for poor vagal tone
    sampen_high: float = 2.0       # threshold for high irregularity
    dfa_normal_range: tuple[float, float] = (0.5, 1.5)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class StimRecommender:
    """Maps autonomic state to stimulation parameters (rule-based).

    Rules (conservative — safety first):
    - Base: amplitude=5mA, frequency=25Hz, pulse_width=250µs (midpoint)
    - High LF/HF (sympathetic dominant) → reduce amplitude
    - Low nHF (poor vagal tone) → increase frequency
    - High SampEn (chaotic RR) → reduce amplitude + widen pulse
    - DFA-α1 outside normal range → reduce amplitude
    - All NaN → safe defaults (low amplitude, low frequency)
    """

    def __init__(self, cfg: StimConfig):
        self.cfg = cfg

    def recommend(self, autonomic_state: dict[str, float]) -> dict[str, float]:
        """Return stim parameters clipped to hardware limits.

        Args:
            autonomic_state: Dict with keys lf_hf_ratio, norm_hf_power, sampen, dfa_alpha1.
                             Values may be NaN (feature unavailable).

        Returns:
            {amplitude: float (mA), frequency: float (Hz), pulse_width: float (µs)}
        """
        lf_hf = autonomic_state.get("lf_hf_ratio", float("nan"))
        nhf = autonomic_state.get("norm_hf_power", float("nan"))
        sampen = autonomic_state.get("sampen", float("nan"))
        dfa = autonomic_state.get("dfa_alpha1", float("nan"))

        # Count how many features are available
        n_valid = sum(1 for v in [lf_hf, nhf, sampen, dfa] if not (math.isnan(v) if isinstance(v, float) else np.isnan(v)))

        # All NaN → conservative safe defaults
        if n_valid == 0:
            return {
                "amplitude": self.cfg.amplitude_range[0] + 1.0,  # 1 mA — minimal
                "frequency": 10.0,
                "pulse_width": float(self.cfg.pulse_width_range[0]),  # 50 µs — minimal
            }

        # Start at midpoint
        amp = 5.0
        freq = 25.0
        pw = 250.0

        # Rule 1: High LF/HF → sympathetic dominant → reduce amplitude
        if not (math.isnan(lf_hf) if isinstance(lf_hf, float) else np.isnan(lf_hf)):
            if lf_hf > self.cfg.lf_hf_high:
                amp -= min(2.0, (lf_hf - self.cfg.lf_hf_high) * 1.0)

        # Rule 2: Low nHF → poor parasympathetic tone → increase frequency
        if not (math.isnan(nhf) if isinstance(nhf, float) else np.isnan(nhf)):
            if nhf < self.cfg.norm_hf_low:
                freq += min(25.0, (self.cfg.norm_hf_low - nhf) * 50.0)

        # Rule 3: High SampEn → chaotic RR → reduce amplitude, widen pulse
        if not (math.isnan(sampen) if isinstance(sampen, float) else np.isnan(sampen)):
            if sampen > self.cfg.sampen_high:
                amp -= min(1.5, (sampen - self.cfg.sampen_high) * 0.5)
                pw += min(100.0, (sampen - self.cfg.sampen_high) * 30.0)

        # Rule 4: DFA-α1 outside normal → reduce amplitude (safety)
        if not (math.isnan(dfa) if isinstance(dfa, float) else np.isnan(dfa)):
            dfa_lo, dfa_hi = self.cfg.dfa_normal_range
            if dfa < dfa_lo or dfa > dfa_hi:
                amp -= 1.0

        # Clip to hardware limits
        amp_lo, amp_hi = self.cfg.amplitude_range
        freq_lo, freq_hi = self.cfg.frequency_range
        pw_lo, pw_hi = self.cfg.pulse_width_range

        return {
            "amplitude": _clamp(amp, amp_lo, amp_hi),
            "frequency": _clamp(freq, freq_lo, freq_hi),
            "pulse_width": _clamp(pw, float(pw_lo), float(pw_hi)),
        }


def build_stim_recommender(config_path: str = "config_stroke.yaml") -> StimRecommender:
    """Factory: load config → StimConfig → StimRecommender."""
    from ..training.build_model import load_config

    config = load_config(config_path)
    m = config.get("stim_recommender", {})

    cfg = StimConfig(
        amplitude_range=tuple(m.get("amplitude_range", [0.0, 10.0])),
        frequency_range=tuple(m.get("frequency_range", [1.0, 100.0])),
        pulse_width_range=tuple(m.get("pulse_width_range", [50, 500])),
        lf_hf_high=float(m.get("lf_hf_high", 2.0)),
        norm_hf_low=float(m.get("norm_hf_low", 0.3)),
        sampen_high=float(m.get("sampen_high", 2.0)),
        dfa_normal_range=tuple(m.get("dfa_normal_range", [0.5, 1.5])),
    )
    return StimRecommender(cfg)
