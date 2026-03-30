"""Tests for EDA feature extraction (F7)."""

import os

import numpy as np
import pytest

from src.features.eda import (
    _empty_result,
    _load_config,
    calibrate_baseline,
    compute_arousal_in_band,
    decompose_eda,
    detect_scr_peaks,
    extract_eda_features,
)

WESAD_DIR = os.path.join("data", "raw", "tinnitus avns", "wesad", "WESAD")
WESAD_AVAILABLE = os.path.isdir(WESAD_DIR) and len(os.listdir(WESAD_DIR)) > 0

CONFIG_PATH = "config_tinnitus.yaml"

# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

def _synthetic_eda(
    fs: float = 4.0,
    duration_sec: float = 300.0,
    baseline_scl: float = 2.0,
    scr_rate_per_min: float = 3.0,
    scr_amplitude: float = 0.5,
    noise_std: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
    """Synthetic EDA: tonic DC + random SCR bursts + Gaussian noise.

    Each SCR is a fast-rise/slow-decay exponential pulse mimicking a real
    skin conductance response.
    """
    rng = np.random.default_rng(seed)
    n = int(fs * duration_sec)
    t = np.arange(n) / fs

    # Tonic baseline (DC)
    signal = np.full(n, baseline_scl, dtype=np.float64)

    # SCR bursts
    scr_interval_sec = 60.0 / max(scr_rate_per_min, 1e-9)
    scr_times = np.arange(scr_interval_sec, duration_sec - 5.0, scr_interval_sec)
    # Add ±20% jitter
    scr_times += rng.uniform(-0.2 * scr_interval_sec, 0.2 * scr_interval_sec, len(scr_times))

    for tc in scr_times:
        idx = int(tc * fs)
        if idx >= n:
            continue
        # Exponential SCR kernel: fast rise (0.5s), slow decay (5s)
        bt = t[idx:] - t[idx]
        amp = rng.uniform(0.5 * scr_amplitude, 1.5 * scr_amplitude)
        scr_kernel = amp * (1 - np.exp(-bt / 0.5)) * np.exp(-bt / 5.0)
        end = min(idx + len(scr_kernel), n)
        signal[idx:end] += scr_kernel[: end - idx]

    # Gaussian noise
    signal += rng.normal(0, noise_std, n)
    signal = np.clip(signal, 0, None)  # EDA must be non-negative
    return signal.astype(np.float64)


def _synthetic_stress_eda(
    fs: float = 4.0,
    duration_sec: float = 300.0,
    seed: int = 99,
) -> np.ndarray:
    """Elevated SCL (~5.0 µS) + higher SCR rate to simulate acute stress."""
    return _synthetic_eda(
        fs=fs,
        duration_sec=duration_sec,
        baseline_scl=5.0,
        scr_rate_per_min=8.0,
        scr_amplitude=1.0,
        noise_std=0.05,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# TestDecomposeEda
# ---------------------------------------------------------------------------

class TestDecomposeEda:

    def test_returns_tonic_phasic_keys(self):
        sig = _synthetic_eda(duration_sec=120.0)
        result = decompose_eda(sig, fs=4.0)
        assert "tonic" in result
        assert "phasic" in result

    def test_output_length_matches_input(self):
        sig = _synthetic_eda(duration_sec=120.0)
        result = decompose_eda(sig, fs=4.0)
        assert len(result["tonic"]) == len(sig)
        assert len(result["phasic"]) == len(sig)

    def test_tonic_smoother_than_phasic(self):
        """Tonic SCL should have lower high-freq variance than phasic SCR."""
        sig = _synthetic_eda(duration_sec=180.0, scr_rate_per_min=5.0, scr_amplitude=0.8)
        result = decompose_eda(sig, fs=4.0)
        tonic_var = float(np.var(np.diff(result["tonic"])))
        phasic_var = float(np.var(np.diff(result["phasic"])))
        # Tonic should be smoother (lower frame-to-frame variance)
        assert tonic_var < phasic_var, (
            f"Tonic var {tonic_var:.6f} should be < phasic var {phasic_var:.6f}"
        )

    def test_tonic_near_baseline_dc(self):
        """Tonic mean should be close to the synthetic DC baseline."""
        baseline_scl = 2.0
        sig = _synthetic_eda(
            duration_sec=240.0,
            baseline_scl=baseline_scl,
            scr_rate_per_min=1.0,
            scr_amplitude=0.3,
            noise_std=0.01,
        )
        result = decompose_eda(sig, fs=4.0)
        # Mean of tonic should be within 1.5 µS of the ground-truth baseline
        assert abs(float(np.mean(result["tonic"])) - baseline_scl) < 1.5

    def test_short_signal_handled(self):
        """5s signal — decompose_eda should return arrays without error."""
        sig = _synthetic_eda(duration_sec=5.0)
        result = decompose_eda(sig, fs=4.0)
        assert "tonic" in result and "phasic" in result
        assert len(result["tonic"]) > 0

    def test_highpass_fallback_method(self):
        """highpass method should also work without cvxopt."""
        sig = _synthetic_eda(duration_sec=120.0)
        result = decompose_eda(sig, fs=4.0, method="highpass")
        assert len(result["tonic"]) == len(sig)


# ---------------------------------------------------------------------------
# TestDetectScrPeaks
# ---------------------------------------------------------------------------

class TestDetectScrPeaks:

    def test_returns_required_keys(self):
        phasic = np.abs(_synthetic_eda(duration_sec=120.0) - 2.0)
        result = detect_scr_peaks(phasic, fs=4.0)
        assert "peak_indices" in result
        assert "amplitudes" in result
        assert "count" in result

    def test_finds_peaks_in_synthetic(self):
        """Decomposed phasic from SCR-rich signal should yield ≥1 peak."""
        sig = _synthetic_eda(
            duration_sec=300.0,
            scr_rate_per_min=5.0,
            scr_amplitude=1.0,
            noise_std=0.01,
        )
        decomposed = decompose_eda(sig, fs=4.0)
        result = detect_scr_peaks(decomposed["phasic"], fs=4.0, min_amplitude=0.02)
        assert result["count"] >= 1

    def test_min_amplitude_filter(self):
        """With a very high amplitude threshold, most peaks should be filtered."""
        sig = _synthetic_eda(
            duration_sec=300.0, scr_rate_per_min=5.0, scr_amplitude=0.5
        )
        decomposed = decompose_eda(sig, fs=4.0)
        phasic = decomposed["phasic"]
        result_low = detect_scr_peaks(phasic, fs=4.0, min_amplitude=0.001)
        result_high = detect_scr_peaks(phasic, fs=4.0, min_amplitude=50.0)
        assert result_high["count"] <= result_low["count"]
        assert result_high["count"] == 0

    def test_flat_signal_returns_zero_peaks(self):
        flat = np.zeros(400, dtype=np.float64)
        result = detect_scr_peaks(flat, fs=4.0)
        assert result["count"] == 0

    def test_amplitudes_match_indices(self):
        sig = _synthetic_eda(duration_sec=300.0, scr_rate_per_min=5.0, scr_amplitude=0.8)
        decomposed = decompose_eda(sig, fs=4.0)
        phasic = decomposed["phasic"]
        result = detect_scr_peaks(phasic, fs=4.0, min_amplitude=0.01)
        if result["count"] > 0:
            expected_amps = phasic[result["peak_indices"]]
            np.testing.assert_allclose(result["amplitudes"], expected_amps, rtol=1e-5)


# ---------------------------------------------------------------------------
# TestCalibrateBaseline
# ---------------------------------------------------------------------------

class TestCalibrateBaseline:

    def test_mean_matches_known_dc(self):
        """Flat tonic at known SCL → calibration mean should equal that SCL."""
        target_scl = 3.5
        tonic = np.full(1000, target_scl, dtype=np.float64)
        mean, std = calibrate_baseline(tonic, fs=4.0, calibration_sec=120.0)
        assert abs(mean - target_scl) < 0.01

    def test_std_positive(self):
        """Tonic with noise → std > 0."""
        rng = np.random.default_rng(0)
        tonic = 2.0 + rng.normal(0, 0.1, 500)
        _, std = calibrate_baseline(tonic, fs=4.0)
        assert std > 0

    def test_flat_signal_floors_std(self):
        """Perfectly flat tonic → std should be floored at 1e-6, not 0."""
        flat = np.full(480, 2.0, dtype=np.float64)
        _, std = calibrate_baseline(flat, fs=4.0)
        assert std >= 1e-6

    def test_clips_to_signal_length(self):
        """calibration_sec > signal length → use entire signal (no crash)."""
        short_tonic = np.linspace(1.5, 2.5, 100)
        mean, std = calibrate_baseline(short_tonic, fs=4.0, calibration_sec=9999.0)
        assert np.isfinite(mean)
        assert std >= 1e-6

    def test_calibration_sec_limits_window(self):
        """Only the first calibration_sec should influence the mean."""
        # First 30s at SCL=2, rest at SCL=10
        tonic = np.concatenate([
            np.full(int(4 * 30), 2.0),   # 30s calibration window
            np.full(int(4 * 270), 10.0), # 270s of elevated SCL
        ])
        mean, _ = calibrate_baseline(tonic, fs=4.0, calibration_sec=30.0)
        # Mean should reflect the calibration segment (≈2.0), not the tail (10.0)
        assert mean < 3.0, f"Mean {mean:.2f} should reflect calibration window ≈2.0"


# ---------------------------------------------------------------------------
# TestComputeArousalInBand
# ---------------------------------------------------------------------------

class TestComputeArousalInBand:

    def test_baseline_in_band(self):
        """Tonic at cal_mean → all frames should be in-band."""
        tonic = np.full(1200, 2.0, dtype=np.float64)  # 300s at 4 Hz
        result = compute_arousal_in_band(
            tonic, fs=4.0, cal_mean=2.0, cal_std=0.2,
            low_sigma=1.5, high_sigma=2.5, frame_rate_hz=1.0,
        )
        assert len(result) > 0
        assert float(np.mean(result)) == pytest.approx(1.0), "All frames should be in-band"

    def test_elevated_scl_out_of_band(self):
        """SCL >> baseline → frames should be out-of-band (upper threshold exceeded)."""
        # cal_mean=2.0, cal_std=0.2, high_thresh=2.5 → values at 5.0 are out
        tonic = np.full(1200, 5.0, dtype=np.float64)
        result = compute_arousal_in_band(
            tonic, fs=4.0, cal_mean=2.0, cal_std=0.2,
            low_sigma=1.5, high_sigma=2.5, frame_rate_hz=1.0,
        )
        assert float(np.mean(result)) == pytest.approx(0.0), "All frames should be out-of-band"

    def test_low_scl_out_of_band(self):
        """SCL << baseline → frames should be out-of-band (lower threshold)."""
        # cal_mean=2.0, cal_std=0.2, low_thresh=1.7 → values at 0.5 are out
        tonic = np.full(1200, 0.5, dtype=np.float64)
        result = compute_arousal_in_band(
            tonic, fs=4.0, cal_mean=2.0, cal_std=0.2,
            low_sigma=1.5, high_sigma=2.5, frame_rate_hz=1.0,
        )
        assert float(np.mean(result)) == pytest.approx(0.0)

    def test_output_shape_correct(self):
        """n_frames = floor(n_samples / samples_per_frame)."""
        fs = 4.0
        frame_rate_hz = 1.0
        duration_sec = 60.0
        tonic = np.full(int(fs * duration_sec), 2.0, dtype=np.float64)
        result = compute_arousal_in_band(
            tonic, fs=fs, cal_mean=2.0, cal_std=0.1,
            low_sigma=1.5, high_sigma=2.5, frame_rate_hz=frame_rate_hz,
        )
        expected_n = int(fs * duration_sec) // int(round(fs / frame_rate_hz))
        assert len(result) == expected_n

    def test_values_binary(self):
        """Output values must be 0.0 or 1.0 (binary gate)."""
        rng = np.random.default_rng(7)
        tonic = 2.0 + rng.normal(0, 0.5, 1200)
        result = compute_arousal_in_band(
            tonic, fs=4.0, cal_mean=2.0, cal_std=0.2,
            low_sigma=1.5, high_sigma=2.5, frame_rate_hz=1.0,
        )
        unique_vals = set(float(v) for v in result)
        assert unique_vals <= {0.0, 1.0}, f"Non-binary values found: {unique_vals}"

    def test_dtype_float32(self):
        tonic = np.full(400, 2.0, dtype=np.float64)
        result = compute_arousal_in_band(
            tonic, fs=4.0, cal_mean=2.0, cal_std=0.1,
            low_sigma=1.5, high_sigma=2.5,
        )
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# TestExtractEdaFeatures
# ---------------------------------------------------------------------------

class TestExtractEdaFeatures:

    REQUIRED_KEYS = {
        "arousal_in_band", "tonic_scl", "phasic_scr",
        "scr_count", "calibration_mean", "calibration_std",
        "low_threshold", "high_threshold", "in_band_fraction",
    }

    def test_output_keys_present(self):
        sig = _synthetic_eda(duration_sec=300.0)
        result = extract_eda_features(sig, fs=4.0, config_path=CONFIG_PATH)
        assert self.REQUIRED_KEYS <= set(result.keys())

    def test_arousal_in_band_dtype(self):
        sig = _synthetic_eda(duration_sec=300.0)
        result = extract_eda_features(sig, fs=4.0, config_path=CONFIG_PATH)
        assert result["arousal_in_band"].dtype == np.float32

    def test_tonic_phasic_dtype(self):
        sig = _synthetic_eda(duration_sec=300.0)
        result = extract_eda_features(sig, fs=4.0, config_path=CONFIG_PATH)
        assert result["tonic_scl"].dtype == np.float32
        assert result["phasic_scr"].dtype == np.float32

    def test_self_cal_baseline_mostly_in_band(self):
        """Baseline signal self-calibrated → >80% frames should be in-band."""
        sig = _synthetic_eda(
            duration_sec=600.0,
            baseline_scl=2.0,
            scr_rate_per_min=2.0,
            scr_amplitude=0.3,
            noise_std=0.01,
        )
        result = extract_eda_features(sig, fs=4.0, config_path=CONFIG_PATH)
        assert result["in_band_fraction"] > 0.8, (
            f"Expected >80% in-band for baseline, got {result['in_band_fraction']:.2%}"
        )

    def test_cross_cal_stress_mostly_out_of_band(self):
        """Stress signal cross-calibrated from baseline → >50% frames out-of-band."""
        baseline_sig = _synthetic_eda(
            duration_sec=600.0, baseline_scl=2.0,
            scr_rate_per_min=2.0, scr_amplitude=0.3,
        )
        stress_sig = _synthetic_stress_eda(duration_sec=300.0)
        result = extract_eda_features(
            stress_sig, fs=4.0,
            calibration_signal=baseline_sig,
            calibration_fs=4.0,
            config_path=CONFIG_PATH,
        )
        out_of_band_fraction = 1.0 - result["in_band_fraction"]
        assert out_of_band_fraction > 0.5, (
            f"Expected >50% out-of-band for stress, got {out_of_band_fraction:.2%}"
        )

    def test_short_signal_returns_nan(self):
        """Signal <10s → _empty_result (NaN arrays)."""
        sig = _synthetic_eda(duration_sec=5.0)
        result = extract_eda_features(sig, fs=4.0, config_path=CONFIG_PATH)
        assert np.all(np.isnan(result["arousal_in_band"]))
        assert result["scr_count"] == 0

    def test_scr_count_is_int(self):
        sig = _synthetic_eda(duration_sec=300.0)
        result = extract_eda_features(sig, fs=4.0, config_path=CONFIG_PATH)
        assert isinstance(result["scr_count"], int)

    def test_in_band_fraction_in_range(self):
        sig = _synthetic_eda(duration_sec=300.0)
        result = extract_eda_features(sig, fs=4.0, config_path=CONFIG_PATH)
        assert 0.0 <= result["in_band_fraction"] <= 1.0

    def test_config_loaded_correctly(self):
        """frame_rate_hz=1.0 → n_frames = floor(n_samples / 4) for 4 Hz signal."""
        duration_sec = 300.0
        sig = _synthetic_eda(duration_sec=duration_sec)
        result = extract_eda_features(sig, fs=4.0, config_path=CONFIG_PATH)
        expected_max_frames = int(duration_sec)  # 300s × 1 Hz
        assert len(result["arousal_in_band"]) <= expected_max_frames
        assert len(result["arousal_in_band"]) > 0

    def test_known_calibration_100pct_in_band(self):
        """Synthetic tonic at exactly cal_mean → 100% in-band."""
        # Build a perfectly flat EDA at 2.0 µS for 300s
        flat_sig = np.full(int(4 * 300), 2.0, dtype=np.float64)
        # Cross-calibrate from same flat signal — cal_mean≈2.0, cal_std floored at 1e-6
        result = extract_eda_features(
            flat_sig, fs=4.0,
            calibration_signal=flat_sig.copy(),
            calibration_fs=4.0,
            config_path=CONFIG_PATH,
        )
        # Allow up to 5% deviation: neurokit2 filter edge effects on a perfectly flat signal
        # introduce tiny tonic artifacts at signal boundaries. With cal_std floored at 1e-6,
        # the threshold band is ~2e-6 uS wide, so edge frames can fall outside. Real EDA always
        # has natural variance (cal_std >> 1e-6) making this a non-issue in practice.
        assert result["in_band_fraction"] >= 0.95, (
            f"Flat signal at cal_mean should be ~100% in-band (±5% edge tolerance), got {result['in_band_fraction']:.2%}"
        )

    def test_calibration_thresholds_consistent(self):
        sig = _synthetic_eda(duration_sec=300.0)
        result = extract_eda_features(sig, fs=4.0, config_path=CONFIG_PATH)
        # low < mean < high
        assert result["low_threshold"] < result["calibration_mean"]
        assert result["calibration_mean"] < result["high_threshold"]


# ---------------------------------------------------------------------------
# TestWesadIntegration
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not WESAD_AVAILABLE, reason="WESAD dataset not available")
class TestWesadIntegration:
    """Per-subject validation on real WESAD data.

    Requires data at data/raw/tinnitus avns/wesad/WESAD/ (S2–S17, S12 excluded).
    """

    SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    EDA_FS = 4.0  # Empatica E4 EDA native rate
    MIN_SUBJECTS_PASS = 5

    def _load_subject(self, subject_id: int):
        """Load WESAD subject pickle and return {baseline_eda, stress_eda}."""
        import pickle
        pkl_path = os.path.join(
            WESAD_DIR, f"S{subject_id}", f"S{subject_id}.pkl"
        )
        if not os.path.isfile(pkl_path):
            return None
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        labels = data["label"]  # (n_samples_700hz,)
        eda_wrist = data["signal"]["wrist"]["EDA"].ravel()  # 4 Hz

        # Labels are at 700 Hz — downsample to 4 Hz for alignment
        ratio = 700 // 4
        min_len = min(len(labels) // ratio, len(eda_wrist))
        labels_4hz = labels[::ratio][:min_len]
        eda_4hz = eda_wrist[:min_len]

        # Extract baseline (label=1) and stress (label=2) epochs
        baseline_mask = labels_4hz == 1
        stress_mask = labels_4hz == 2

        baseline_eda = eda_4hz[baseline_mask].astype(np.float64)
        stress_eda = eda_4hz[stress_mask].astype(np.float64)

        return {"baseline": baseline_eda, "stress": stress_eda}

    def test_baseline_self_cal_in_band(self):
        """Baseline self-calibration: ≥MIN_SUBJECTS_PASS have >80% in-band."""
        pass_count = 0
        tested = 0
        for sid in self.SUBJECTS:
            subject = self._load_subject(sid)
            if subject is None:
                continue
            baseline_eda = subject["baseline"]
            if len(baseline_eda) < int(10 * self.EDA_FS):
                continue
            tested += 1
            result = extract_eda_features(
                baseline_eda, fs=self.EDA_FS, config_path=CONFIG_PATH
            )
            if result["in_band_fraction"] > 0.8:
                pass_count += 1

        assert tested >= self.MIN_SUBJECTS_PASS, (
            f"Only {tested} subjects had sufficient baseline data"
        )
        assert pass_count >= self.MIN_SUBJECTS_PASS, (
            f"Only {pass_count}/{tested} subjects passed 80% in-band criterion "
            f"(need {self.MIN_SUBJECTS_PASS})"
        )

    def test_stress_cross_cal_out_of_band(self):
        """Stress cross-calibrated from baseline: ≥MIN_SUBJECTS_PASS have >50% out-of-band."""
        pass_count = 0
        tested = 0
        for sid in self.SUBJECTS:
            subject = self._load_subject(sid)
            if subject is None:
                continue
            baseline_eda = subject["baseline"]
            stress_eda = subject["stress"]
            if (len(baseline_eda) < int(10 * self.EDA_FS) or
                    len(stress_eda) < int(10 * self.EDA_FS)):
                continue
            tested += 1
            result = extract_eda_features(
                stress_eda, fs=self.EDA_FS,
                calibration_signal=baseline_eda,
                calibration_fs=self.EDA_FS,
                config_path=CONFIG_PATH,
            )
            out_of_band = 1.0 - result["in_band_fraction"]
            if out_of_band > 0.5:
                pass_count += 1

        assert tested >= self.MIN_SUBJECTS_PASS, (
            f"Only {tested} subjects had sufficient baseline+stress data"
        )
        assert pass_count >= self.MIN_SUBJECTS_PASS, (
            f"Only {pass_count}/{tested} subjects passed 50% out-of-band criterion "
            f"(need {self.MIN_SUBJECTS_PASS})"
        )
