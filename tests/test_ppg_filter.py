"""Tests for PPG bandpass filter."""

import numpy as np
import pytest
from scipy.signal import welch

from src.features.ppg_filter import denoise_ppg


def _synthetic_ppg(fs: float = 125.0, duration_sec: float = 10.0, hr_bpm: float = 70.0) -> np.ndarray:
    """Generate synthetic PPG with cardiac fundamental + HF noise + baseline drift."""
    n = int(fs * duration_sec)
    t = np.arange(n) / fs
    hr_hz = hr_bpm / 60.0
    # Cardiac fundamental + 2nd harmonic
    ppg = np.sin(2 * np.pi * hr_hz * t) + 0.3 * np.sin(2 * np.pi * 2 * hr_hz * t)
    # Baseline drift (0.1 Hz)
    ppg += 0.5 * np.sin(2 * np.pi * 0.1 * t)
    # HF noise (50 Hz powerline)
    ppg += 0.2 * np.sin(2 * np.pi * 50 * t)
    return ppg


class TestDenoisePpg:
    def test_output_same_length(self):
        sig = _synthetic_ppg()
        out = denoise_ppg(sig, fs=125.0)
        assert len(out) == len(sig)

    def test_output_is_float64(self):
        sig = _synthetic_ppg()
        out = denoise_ppg(sig, fs=125.0)
        assert out.dtype == np.float64

    def test_removes_baseline_drift(self):
        """After filtering, DC and sub-0.5Hz power should be negligible."""
        sig = _synthetic_ppg(fs=125.0, duration_sec=20.0)
        out = denoise_ppg(sig, fs=125.0)
        # DC component
        assert abs(np.mean(out)) < 0.05

    def test_removes_high_freq_noise(self):
        """Power above 8 Hz should be suppressed after filtering."""
        fs = 125.0
        sig = _synthetic_ppg(fs=fs, duration_sec=20.0)
        out = denoise_ppg(sig, fs=fs)
        freqs, psd = welch(out, fs=fs, nperseg=256)
        hf_power = psd[freqs > 10].sum()
        passband_power = psd[(freqs >= 0.5) & (freqs <= 8)].sum()
        assert hf_power < passband_power * 0.01  # HF power <1% of passband

    def test_preserves_cardiac_frequency(self):
        """Cardiac fundamental (1.17 Hz for 70 bpm) should survive filtering."""
        fs = 125.0
        hr_bpm = 70.0
        hr_hz = hr_bpm / 60.0
        sig = _synthetic_ppg(fs=fs, duration_sec=20.0, hr_bpm=hr_bpm)
        out = denoise_ppg(sig, fs=fs)
        freqs, psd = welch(out, fs=fs, nperseg=512)
        peak_idx = np.argmax(psd)
        assert abs(freqs[peak_idx] - hr_hz) < 0.2  # peak within ±0.2 Hz of cardiac freq

    def test_too_short_signal_returns_copy(self):
        """Signal shorter than minimum returns unchanged copy with warning."""
        sig = np.random.randn(5)  # 5 samples — too short
        out = denoise_ppg(sig, fs=125.0, bandpass_order=4)
        np.testing.assert_array_equal(out, sig)

    def test_loads_params_from_config(self):
        sig = _synthetic_ppg(fs=125.0, duration_sec=10.0)
        out = denoise_ppg(sig, fs=125.0, config_path="config_tinnitus.yaml")
        assert len(out) == len(sig)
        assert out.dtype == np.float64

    def test_custom_cutoffs(self):
        sig = _synthetic_ppg(fs=125.0, duration_sec=10.0)
        out = denoise_ppg(sig, fs=125.0, bandpass_low=0.5, bandpass_high=5.0)
        assert len(out) == len(sig)

    def test_wesad_ppg_fs(self):
        """Should work at WESAD BVP rate (64 Hz)."""
        sig = _synthetic_ppg(fs=64.0, duration_sec=10.0)
        out = denoise_ppg(sig, fs=64.0)
        assert len(out) == len(sig)
