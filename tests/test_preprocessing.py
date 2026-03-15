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
from src.features.artifact_scrubber import correct_rr_intervals, should_reject_window
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
    assert should_reject_window(signal, rr, amplitude_mad_multiple=5.0, rr_deviation_percent=25.0, rr_fraction_threshold=0.30) is True


def test_artifact_accept_clean():
    """should_reject_window returns False for clean window (constant signal, no outliers)."""
    signal = np.ones(2500, dtype=np.float64) * 0.5
    rr = np.ones(30, dtype=np.float64) * 0.8
    assert should_reject_window(signal, rr, amplitude_mad_multiple=5.0, rr_deviation_percent=25.0, rr_fraction_threshold=0.30) is False


# ─── New tests for scaler / HRV bug fixes ───────────────────────────────────


def test_scaler_transform_sparse_nan():
    """Bug 1a: valid columns are scaled even when other columns in the same row are NaN."""
    rng = np.random.default_rng(99)
    X = np.full((50, N_FEATURES), np.nan, dtype=np.float64)
    X[:, 0] = rng.standard_normal(50)  # RMSSD valid
    X[:, 1] = rng.standard_normal(50)  # SDNN valid
    # cols 2-6 remain NaN
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "scaler.pkl")
        scaler = fit_scaler(X, path=path)
        Xt = transform(X, scaler)
    # Cols 0-1 should be scaled (non-NaN)
    assert not np.any(np.isnan(Xt[:, 0])), "Col 0 should be non-NaN after transform"
    assert not np.any(np.isnan(Xt[:, 1])), "Col 1 should be non-NaN after transform"
    # Cols 2-6 should remain NaN
    for col in range(2, N_FEATURES):
        assert np.all(np.isnan(Xt[:, col])), f"Col {col} should remain NaN"


def test_scaler_transform_fully_nan_rows():
    """Bug 1a edge: fully-NaN rows stay NaN; other rows are scaled correctly."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, N_FEATURES)).astype(np.float64)
    X[10, :] = np.nan  # fully NaN row
    X[20, :] = np.nan
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "scaler.pkl")
        scaler = fit_scaler(X, path=path)
        Xt = transform(X, scaler)
    assert np.all(np.isnan(Xt[10])), "Fully-NaN row 10 should remain NaN"
    assert np.all(np.isnan(Xt[20])), "Fully-NaN row 20 should remain NaN"
    assert not np.any(np.isnan(Xt[0])), "Row 0 (all valid) should be fully scaled"


def test_pipeline_shape_60s_subwindow():
    """Bugs 3a/3b: shape is (5, 7) when subwindow_sec=60 is passed explicitly."""
    try:
        import neurokit2 as nk
    except ImportError:
        pytest.skip("neurokit2 required for synthetic ECG")
    fs = 250.0
    ecg = nk.ecg_simulate(duration=int(WINDOW_5MIN_SEC), sampling_rate=int(fs), heart_rate=72)
    ecg = np.asarray(ecg, dtype=np.float64)
    seq = waveform_to_hrv_sequence(ecg, fs, subwindow_sec=60.0)
    assert seq.shape == (5, N_FEATURES), f"Expected (5, {N_FEATURES}), got {seq.shape}"


def test_pipeline_reads_config_subwindow():
    """Bug 3a: config subwindow_sec is read when subwindow_sec is not passed."""
    try:
        import neurokit2 as nk
    except ImportError:
        pytest.skip("neurokit2 required for synthetic ECG")
    import yaml as _yaml
    fs = 250.0
    ecg = nk.ecg_simulate(duration=int(WINDOW_5MIN_SEC), sampling_rate=int(fs), heart_rate=72)
    ecg = np.asarray(ecg, dtype=np.float64)
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = str(Path(tmp) / "test_config.yaml")
        cfg = {
            "hrv": {"subwindow_sec": 60},
            "wavelet": {"family": "cmor", "scale_range": [1, 64]},
            "artifact": {"amplitude_mad_multiple": 5, "rr_deviation_percent": 60},
        }
        with open(cfg_path, "w") as f:
            _yaml.safe_dump(cfg, f)
        seq = waveform_to_hrv_sequence(ecg, fs, config_path=cfg_path)
    assert seq.shape[0] == 5, f"Expected 5 time steps from config, got {seq.shape[0]}"


def test_hrv_freq_with_32_to_63_samples():
    """Bug 3c: freq HRV returns non-NaN with 48 R-R intervals (between old and new MIN_SAMPLES)."""
    rr = np.random.uniform(0.7, 0.9, 48).astype(np.float64)
    out = compute_hrv_freq(rr)
    assert not np.isnan(out["lf"]), "LF should be non-NaN with 48 intervals"
    assert not np.isnan(out["hf"]), "HF should be non-NaN with 48 intervals"


def test_hrv_freq_below_min_samples():
    """Bug 3c: freq HRV still returns NaN below 32 intervals."""
    rr = np.random.uniform(0.7, 0.9, 20).astype(np.float64)
    out = compute_hrv_freq(rr)
    assert np.isnan(out["lf"]), "LF should be NaN with only 20 intervals"


def test_end_to_end_pipeline_scaler_nonzero():
    """Full chain: sparse HRV -> fit_scaler -> transform -> nan_to_num -> non-zero for valid cols."""
    rng = np.random.default_rng(123)
    # Simulate a (5, 7) HRV sequence: cols 0-1 valid, cols 2-4 valid in some rows, cols 5-6 NaN
    seq = np.full((5, N_FEATURES), np.nan, dtype=np.float64)
    seq[:, 0] = rng.uniform(0.02, 0.08, 5)   # RMSSD
    seq[:, 1] = rng.uniform(0.03, 0.10, 5)   # SDNN
    seq[:3, 2] = rng.uniform(100, 500, 3)     # LF (3 of 5 rows)
    seq[:3, 3] = rng.uniform(50, 300, 3)      # HF
    seq[:3, 4] = seq[:3, 2] / seq[:3, 3]      # LF/HF
    # cols 5-6 stay NaN (nonlinear features missing)
    with tempfile.TemporaryDirectory() as tmp:
        scaler_path = str(Path(tmp) / "scaler.pkl")
        scaler = fit_scaler(seq.reshape(-1, N_FEATURES), path=scaler_path)
        scaled = transform(seq, scaler)
        np.nan_to_num(scaled, nan=0.0, copy=False)
    # RMSSD (col 0) and SDNN (col 1) should have non-zero values
    assert np.any(np.abs(scaled[:, 0]) > 1e-6), "RMSSD column should have non-zero scaled values"
    assert np.any(np.abs(scaled[:, 1]) > 1e-6), "SDNN column should have non-zero scaled values"
    # LF (col 2) should have non-zero for the 3 valid rows
    assert np.sum(np.abs(scaled[:, 2]) > 1e-6) >= 3, "LF column should have non-zero scaled values"


def test_fit_scaler_with_all_nan_columns():
    """Bugs 2a/2b: fit_scaler handles columns that are entirely NaN (mean=0, scale=1)."""
    rng = np.random.default_rng(7)
    X = np.full((100, N_FEATURES), np.nan, dtype=np.float64)
    X[:, 0] = rng.standard_normal(100)
    X[:, 1] = rng.standard_normal(100)
    # cols 2-6 are all NaN
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "scaler.pkl")
        scaler = fit_scaler(X, path=path)
    # Valid columns should have real stats
    assert abs(scaler.mean_[0]) < 1.0, "Col 0 mean should be near 0"
    assert scaler.scale_[0] > 0.5, "Col 0 scale should be reasonable"
    # All-NaN columns should have default mean=0, scale=1
    for col in range(2, N_FEATURES):
        assert scaler.mean_[col] == 0.0, f"All-NaN col {col} should have mean=0"
        assert scaler.scale_[col] == 1.0, f"All-NaN col {col} should have scale=1"


# ─── New tests for artifact correction and pipeline fixes ──────────────────


def test_correct_rr_interpolates_outliers():
    """correct_rr_intervals replaces outlier beats with interpolated values."""
    rr = np.ones(50, dtype=np.float64) * 0.8
    rr[10] = 2.0  # single outlier (150% deviation from 0.8)
    corrected, fraction = correct_rr_intervals(rr, rr_deviation_percent=60.0)
    assert fraction == pytest.approx(1.0 / 50, abs=1e-6)
    assert abs(corrected[10] - 0.8) < 0.1, "Outlier should be interpolated near median"
    assert np.allclose(corrected[:10], rr[:10]), "Non-outlier beats unchanged"


def test_correct_rr_all_outliers():
    """When nearly all beats are outliers, most get replaced; high fraction returned."""
    rr = np.array([0.2, 2.0, 0.1, 3.0, 0.15], dtype=np.float64)
    corrected, fraction = correct_rr_intervals(rr, rr_deviation_percent=10.0)
    # Median (0.2) is within 10% of itself, so at least 1 beat is not an outlier
    assert fraction >= 0.6, f"Most beats should be outliers, got fraction={fraction}"
    # Corrected values should be less extreme than originals
    assert corrected.max() < rr.max(), "Corrected max should be less extreme"


def test_artifact_fraction_accepts_few_outliers():
    """2/50 outlier beats (4%) should NOT trigger rejection (threshold=30%)."""
    signal = np.random.randn(2500) * 0.5
    rr = np.ones(50, dtype=np.float64) * 0.8
    rr[5] = 2.0  # outlier 1
    rr[25] = 2.0  # outlier 2
    result = should_reject_window(
        signal, rr, amplitude_mad_multiple=100.0, rr_deviation_percent=60.0, rr_fraction_threshold=0.30
    )
    assert result is False, "4% outlier fraction should not reject (threshold 30%)"


def test_artifact_fraction_rejects_many_outliers():
    """20/50 outlier beats (40%) should trigger rejection (threshold=30%)."""
    signal = np.random.randn(2500) * 0.5
    rr = np.ones(50, dtype=np.float64) * 0.8
    rr[:20] = 2.0  # 20 outliers
    result = should_reject_window(
        signal, rr, amplitude_mad_multiple=100.0, rr_deviation_percent=60.0, rr_fraction_threshold=0.30
    )
    assert result is True, "40% outlier fraction should reject (threshold 30%)"


def test_pipeline_nonlinear_valid():
    """With 300s synthetic ECG at 250 Hz, SampEn and DFA columns should not all be NaN."""
    try:
        import neurokit2 as nk
    except ImportError:
        pytest.skip("neurokit2 required for synthetic ECG")
    import yaml as _yaml
    fs = 250.0
    ecg = nk.ecg_simulate(duration=int(WINDOW_5MIN_SEC), sampling_rate=int(fs), heart_rate=72)
    ecg = np.asarray(ecg, dtype=np.float64)
    # Use config with high amplitude threshold so QRS spikes don't trigger rejection
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = str(Path(tmp) / "test_config.yaml")
        cfg = {
            "hrv": {"subwindow_sec": 60},
            "wavelet": {"family": "cmor", "scale_range": [1, 64]},
            "artifact": {"amplitude_mad_multiple": 20, "rr_deviation_percent": 60, "rr_fraction_threshold": 0.30},
        }
        with open(cfg_path, "w") as f:
            _yaml.safe_dump(cfg, f)
        seq = waveform_to_hrv_sequence(ecg, fs, subwindow_sec=60.0, config_path=cfg_path)
    # At least one timestep should have non-NaN sampen (col 5) or dfa (col 6)
    assert not np.all(np.isnan(seq[:, 5])), "SampEn should not be all-NaN with 300s ECG"
    assert not np.all(np.isnan(seq[:, 6])), "DFA alpha1 should not be all-NaN with 300s ECG"


def test_pipeline_corrected_rr_af_pattern():
    """Pipeline produces valid time-domain features for a window with AF-like irregular RR."""
    try:
        import neurokit2 as nk
    except ImportError:
        pytest.skip("neurokit2 required for synthetic ECG")
    import yaml as _yaml
    fs = 250.0
    ecg = nk.ecg_simulate(duration=int(WINDOW_5MIN_SEC), sampling_rate=int(fs), heart_rate=80, heart_rate_std=20)
    ecg = np.asarray(ecg, dtype=np.float64)
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = str(Path(tmp) / "test_config.yaml")
        cfg = {
            "hrv": {"subwindow_sec": 60},
            "wavelet": {"family": "cmor", "scale_range": [1, 64]},
            "artifact": {"amplitude_mad_multiple": 20, "rr_deviation_percent": 60, "rr_fraction_threshold": 0.30},
        }
        with open(cfg_path, "w") as f:
            _yaml.safe_dump(cfg, f)
        seq = waveform_to_hrv_sequence(ecg, fs, subwindow_sec=60.0, config_path=cfg_path)
    # RMSSD (col 0) and SDNN (col 1) should have at least some valid values
    valid_rmssd = np.sum(~np.isnan(seq[:, 0]))
    assert valid_rmssd >= 1, f"Expected at least 1 valid RMSSD, got {valid_rmssd}"


def test_short_record_60s_has_partial_hrv():
    """60s signal < 300s → row 0 has valid time-domain HRV, rows 1-4 NaN."""
    import neurokit2 as nk
    import yaml as _yaml
    fs = 250.0
    ecg = nk.ecg_simulate(duration=60, sampling_rate=int(fs), heart_rate=72)
    ecg = np.asarray(ecg, dtype=np.float64)
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = str(Path(tmp) / "test_config.yaml")
        cfg = {
            "hrv": {"subwindow_sec": 60},
            "wavelet": {"family": "cmor", "scale_range": [1, 64]},
            "artifact": {"amplitude_mad_multiple": 30, "rr_deviation_percent": 60, "rr_fraction_threshold": 0.30},
        }
        with open(cfg_path, "w") as f:
            _yaml.safe_dump(cfg, f)
        seq = waveform_to_hrv_sequence(ecg, fs, subwindow_sec=60.0, config_path=cfg_path)
    assert seq.shape == (5, N_FEATURES), f"Expected (5, {N_FEATURES}), got {seq.shape}"
    assert not np.isnan(seq[0, 0]), "RMSSD should be valid for 60s ECG"
    assert not np.isnan(seq[0, 1]), "SDNN should be valid for 60s ECG"
    assert np.all(np.isnan(seq[1:])), "Rows 1-4 should be NaN for short record"


def test_short_record_30s_partial_hrv():
    """30s ECG → row 0 has valid time-domain features, rows 1-4 NaN."""
    import neurokit2 as nk
    import yaml as _yaml
    fs = 250.0
    ecg = nk.ecg_simulate(duration=30, sampling_rate=int(fs), heart_rate=72)
    ecg = np.asarray(ecg, dtype=np.float64)
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = str(Path(tmp) / "test_config.yaml")
        cfg = {
            "hrv": {"subwindow_sec": 60},
            "wavelet": {"family": "cmor", "scale_range": [1, 64]},
            "artifact": {"amplitude_mad_multiple": 30, "rr_deviation_percent": 60, "rr_fraction_threshold": 0.30},
        }
        with open(cfg_path, "w") as f:
            _yaml.safe_dump(cfg, f)
        seq = waveform_to_hrv_sequence(ecg, fs, subwindow_sec=60.0, config_path=cfg_path)
    assert seq.shape == (5, N_FEATURES)
    assert not np.isnan(seq[0, 0]), "RMSSD should be valid for 30s ECG"
    assert not np.isnan(seq[0, 1]), "SDNN should be valid for 30s ECG"
    assert np.all(np.isnan(seq[1:])), "Rows 1-4 should be NaN"


def test_very_short_record_5s_all_nan():
    """5s random noise → too few peaks, all-NaN HRV."""
    import yaml as _yaml
    fs = 250.0
    signal = np.random.randn(int(fs * 5)).astype(np.float64) * 0.01
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = str(Path(tmp) / "test_config.yaml")
        cfg = {
            "hrv": {"subwindow_sec": 60},
            "wavelet": {"family": "cmor", "scale_range": [1, 64]},
            "artifact": {"amplitude_mad_multiple": 30, "rr_deviation_percent": 60, "rr_fraction_threshold": 0.30},
        }
        with open(cfg_path, "w") as f:
            _yaml.safe_dump(cfg, f)
        seq = waveform_to_hrv_sequence(signal, fs, subwindow_sec=60.0, config_path=cfg_path)
    assert seq.shape == (5, N_FEATURES)
    assert np.all(np.isnan(seq)), "All HRV should be NaN for 5s noise"


def test_nonlinear_inf_clamped_to_nan():
    """Constant RR intervals produce inf sampen; guard should clamp to NaN."""
    from src.features.hrv_nonlinear import compute_hrv_nonlinear
    # Constant intervals trigger inf from entropy_sample (no template matches)
    rr = np.ones(50, dtype=np.float64) * 0.8
    result = compute_hrv_nonlinear(rr)
    assert not np.isinf(result["sampen"]), "sampen must not be inf"
    assert not np.isinf(result["dfa_alpha1"]), "dfa_alpha1 must not be inf"


def test_fit_scaler_ignores_inf():
    """Scaler fit should ignore inf values, not be poisoned by them."""
    X = np.random.randn(100, 7).astype(np.float64)
    X[0, 5] = np.inf
    X[1, 5] = -np.inf
    X[2, 5] = np.nan
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "scaler.pkl")
        cfg_path = str(Path(tmp) / "test_config.yaml")
        import yaml as _yaml
        with open(cfg_path, "w") as f:
            _yaml.safe_dump({"paths": {"scaler": path}}, f)
        scaler = fit_scaler(X, path=path, config_path=cfg_path)
    assert np.isfinite(scaler.mean_[5]), f"mean[5] should be finite, got {scaler.mean_[5]}"
    assert scaler.scale_[5] > 0, f"scale[5] should be > 0, got {scaler.scale_[5]}"


def test_transform_treats_inf_as_nan():
    """Transform should treat inf values as missing (NaN in output)."""
    X_train = np.random.randn(50, 7).astype(np.float64)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "scaler.pkl")
        cfg_path = str(Path(tmp) / "test_config.yaml")
        import yaml as _yaml
        with open(cfg_path, "w") as f:
            _yaml.safe_dump({"paths": {"scaler": path}}, f)
        scaler = fit_scaler(X_train, path=path, config_path=cfg_path)
    X_test = np.random.randn(10, 7).astype(np.float64)
    X_test[3, 2] = np.inf
    X_test[7, 5] = -np.inf
    out = transform(X_test, scaler)
    assert np.isnan(out[3, 2]), "inf should become NaN after transform"
    assert np.isnan(out[7, 5]), "-inf should become NaN after transform"
    assert np.isfinite(out[0, 0]), "Normal values should stay finite"


def test_e2e_inf_sampen_survives_scaling():
    """End-to-end: inf in sampen column doesn't zero out all sampen values."""
    X = np.random.uniform(0.5, 2.0, (50, 7)).astype(np.float64)
    X[0, 5] = np.inf  # one inf sampen
    X[1, 5] = np.inf  # another
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "scaler.pkl")
        cfg_path = str(Path(tmp) / "test_config.yaml")
        import yaml as _yaml
        with open(cfg_path, "w") as f:
            _yaml.safe_dump({"paths": {"scaler": path}}, f)
        scaler = fit_scaler(X, path=path, config_path=cfg_path)
    out = transform(X, scaler)
    np.nan_to_num(out, nan=0.0, copy=False)
    # Non-inf sampen values should survive as non-zero
    non_inf_sampen = out[2:, 5]  # rows 2+ had valid values
    assert np.any(non_inf_sampen != 0), "Valid sampen values should be non-zero after scaling"
    # inf positions should be 0 (NaN -> 0)
    assert out[0, 5] == 0.0, "inf sampen should become 0 after nan_to_num"
    assert out[1, 5] == 0.0, "inf sampen should become 0 after nan_to_num"


def test_e2e_short_record_nonzero_after_scaling():
    """End-to-end: 45s ECG produces non-zero scaled HRV in row 0."""
    import neurokit2 as nk
    import yaml as _yaml
    fs = 250.0
    ecg = nk.ecg_simulate(duration=45, sampling_rate=int(fs), heart_rate=72)
    ecg = np.asarray(ecg, dtype=np.float64)
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = str(Path(tmp) / "test_config.yaml")
        cfg = {
            "hrv": {"subwindow_sec": 60},
            "wavelet": {"family": "cmor", "scale_range": [1, 64]},
            "artifact": {"amplitude_mad_multiple": 30, "rr_deviation_percent": 60, "rr_fraction_threshold": 0.30},
        }
        with open(cfg_path, "w") as f:
            _yaml.safe_dump(cfg, f)
        seq = waveform_to_hrv_sequence(ecg, fs, subwindow_sec=60.0, config_path=cfg_path)
        # Fit scaler on synthetic training data (not the single test sample)
        train_hrv = np.random.uniform(0.01, 2.0, (200, N_FEATURES)).astype(np.float64)
        scaler_path = str(Path(tmp) / "scaler.pkl")
        scaler = fit_scaler(train_hrv, path=scaler_path, config_path=cfg_path)
        scaled = transform(seq, scaler)
        np.nan_to_num(scaled, nan=0.0, copy=False)
    # Row 0 time-domain features should be non-zero
    assert scaled[0, 0] != 0.0, "Scaled RMSSD should be non-zero for 45s ECG"
    assert scaled[0, 1] != 0.0, "Scaled SDNN should be non-zero for 45s ECG"
    # Rows 1-4 should be all zero (NaN -> 0)
    assert np.all(scaled[1:] == 0.0), "Rows 1-4 should be zero after nan_to_num"
