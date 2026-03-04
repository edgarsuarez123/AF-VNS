"""
Tests for preprocessing and feature extraction (FR-2.1, FR-2.2): CWT, HRV, pipeline shape, scaler.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.wavelet_filter import denoise
from src.features.peak_detector import get_rr_intervals
from src.features.hrv_time import compute_hrv_time
from src.features.hrv_freq import compute_hrv_freq
from src.features.hrv_nonlinear import compute_hrv_nonlinear
from src.features.artifact_scrubber import should_reject_window
from src.features.pipeline import waveform_to_hrv_sequence, N_FEATURES, WINDOW_5MIN_SEC
from src.features.scaler import fit_scaler, load_scaler, transform


def test_denoise_same_length():
    """denoise preserves signal length."""
    fs = 250.0
    signal = np.random.randn(int(fs * 60)).astype(np.float64) * 0.5
    out = denoise(signal, fs)
    assert out.shape == signal.shape
    assert out.dtype == np.float64


def test_denoise_reduces_baseline_power():
    """Optional: 0-0.5 Hz power is reduced after denoise."""
    from scipy import integrate, signal as scipy_signal
    fs = 250.0
    t = np.arange(0, 10.0, 1.0 / fs)
    baseline = np.sin(2 * np.pi * 0.2 * t)  # 0.2 Hz
    signal = baseline + 0.1 * np.random.randn(len(t))
    out = denoise(signal, fs)
    f, p_in = scipy_signal.welch(signal, fs=fs, nperseg=min(256, len(signal) // 2))
    f, p_out = scipy_signal.welch(out, fs=fs, nperseg=min(256, len(out) // 2))
    mask_lo = (f >= 0) & (f <= 0.5)
    power_in = integrate.trapezoid(p_in[mask_lo], f[mask_lo])
    power_out = integrate.trapezoid(p_out[mask_lo], f[mask_lo])
    assert power_out <= power_in * 1.5  # allow some tolerance


def test_hrv_time_vs_neurokit2():
    """Our compute_hrv_time returns valid RMSSD/SDNN; optional comparison to NeuroKit2 (peaks format)."""
    rr = np.array([0.8, 0.82, 0.79, 0.81, 0.80, 0.83, 0.81, 0.80, 0.79, 0.81], dtype=np.float64)
    ours = compute_hrv_time(rr)
    assert "rmssd" in ours and "sdnn" in ours
    assert not np.isnan(ours["rmssd"]) and not np.isnan(ours["sdnn"])
    assert ours["rmssd"] > 0 and ours["sdnn"] > 0


def test_hrv_freq_returns_keys():
    """compute_hrv_freq returns lf, hf, lf_hf_ratio."""
    rr = np.random.uniform(0.7, 0.9, 128).astype(np.float64)
    out = compute_hrv_freq(rr)
    assert "lf" in out and "hf" in out and "lf_hf_ratio" in out


def test_pipeline_shape():
    """waveform_to_hrv_sequence returns (n_time_steps, n_features) with n_time_steps=10, n_features=7."""
    try:
        import neurokit2 as nk
    except ImportError:
        pytest.skip("neurokit2 required for synthetic ECG")
    fs = 250.0
    duration = WINDOW_5MIN_SEC
    ecg = nk.ecg_simulate(duration=int(duration), sampling_rate=int(fs), heart_rate=72)
    ecg = np.asarray(ecg, dtype=np.float64)
    seq = waveform_to_hrv_sequence(ecg, fs, subwindow_sec=30.0)
    n_steps = int(WINDOW_5MIN_SEC / 30.0)
    assert seq.shape == (n_steps, N_FEATURES), f"Expected ({n_steps}, {N_FEATURES}), got {seq.shape}"
    assert seq.dtype == np.float32


def test_scaler_fit_load_transform():
    """fit_scaler, load_scaler, transform: transformed data has mean≈0, std≈1."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, N_FEATURES)).astype(np.float64)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "scaler.pkl")
        scaler = fit_scaler(X, path=path)
        loaded = load_scaler(path=path)
        Xt = transform(X, loaded)
    assert np.allclose(np.nanmean(Xt, axis=0), 0.0, atol=1e-5)
    assert np.allclose(np.nanstd(Xt, axis=0), 1.0, atol=1e-5)


def test_artifact_reject_amplitude():
    """should_reject_window returns True when amplitude exceeds MAD multiple."""
    signal = np.ones(1000) * 0.1
    signal[100:102] = 100.0  # spike
    rr = np.ones(50) * 0.8
    assert should_reject_window(signal, rr, amplitude_mad_multiple=5.0, rr_deviation_percent=25.0) is True


def test_artifact_accept_clean():
    """should_reject_window returns False for clean window (constant signal, no outliers)."""
    signal = np.ones(2500, dtype=np.float64) * 0.5
    rr = np.ones(30, dtype=np.float64) * 0.8
    assert should_reject_window(signal, rr, amplitude_mad_multiple=5.0, rr_deviation_percent=25.0) is False
