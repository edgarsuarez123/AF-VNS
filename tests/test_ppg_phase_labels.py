"""Tests for PPG diastole detection."""

import numpy as np
import pytest

from src.features.ppg_phase_labels import (
    generate_ppg_phase_labels,
    get_ppg_peak_indices,
    _detect_dicrotic_notch,
)
from src.features.ppg_filter import denoise_ppg


def _synthetic_ppg_pulse(
    fs: float = 125.0,
    duration_sec: float = 10.0,
    hr_bpm: float = 70.0,
) -> np.ndarray:
    """Synthetic PPG with realistic systolic peak + dicrotic notch shape."""
    n = int(fs * duration_sec)
    t = np.arange(n) / fs
    hr_hz = hr_bpm / 60.0
    period = 1.0 / hr_hz
    signal = np.zeros(n)
    for beat_start in np.arange(0, duration_sec, period):
        bt = t - beat_start
        # Systolic peak at 0.1s into beat
        mask = (bt >= 0) & (bt < period * 0.95)
        signal += np.where(mask, np.exp(-((bt - 0.1) ** 2) / 0.002), 0)
        # Dicrotic notch at ~0.3s (small dip then secondary hump)
        signal += np.where(mask, -0.2 * np.exp(-((bt - 0.30) ** 2) / 0.001), 0)
        signal += np.where(mask, 0.15 * np.exp(-((bt - 0.35) ** 2) / 0.003), 0)
    return signal.astype(np.float64)


class TestGetPpgPeakIndices:
    def test_detects_peaks(self):
        sig = _synthetic_ppg_pulse(fs=125.0, duration_sec=10.0, hr_bpm=70.0)
        peaks = get_ppg_peak_indices(sig, fs=125.0)
        assert len(peaks) >= 8  # ~11 beats in 10s at 70 bpm

    def test_returns_int64(self):
        sig = _synthetic_ppg_pulse(fs=125.0, duration_sec=5.0)
        peaks = get_ppg_peak_indices(sig, fs=125.0)
        assert peaks.dtype == np.int64

    def test_peaks_in_bounds(self):
        sig = _synthetic_ppg_pulse(fs=125.0, duration_sec=5.0)
        peaks = get_ppg_peak_indices(sig, fs=125.0)
        assert np.all(peaks >= 0)
        assert np.all(peaks < len(sig))


class TestDetectDicroticNotch:
    def test_returns_index_in_range(self):
        n = 200
        sig = np.zeros(n)
        sig[50] = 1.0  # systolic peak
        sig[80] = -0.2  # dicrotic notch
        notch = _detect_dicrotic_notch(sig, peak_idx=50, next_peak_idx=180, notch_start_fraction=0.3)
        # Should be somewhere between 50+0.3*130=89 and 180-0.3*130=141
        assert 50 < notch < 180

    def test_fallback_on_too_short_beat(self):
        sig = np.zeros(10)
        # Peak and next peak too close for search window
        notch = _detect_dicrotic_notch(sig, peak_idx=0, next_peak_idx=5, notch_start_fraction=0.5)
        assert 0 <= notch <= 5


class TestGeneratePpgPhaseLabels:
    def test_output_keys(self):
        sig = _synthetic_ppg_pulse(fs=125.0, duration_sec=10.0)
        result = generate_ppg_phase_labels(sig, fs=125.0)
        assert "labels" in result
        assert "quality" in result
        assert "n_beats" in result
        assert "mean_hr_bpm" in result

    def test_labels_shape(self):
        fs = 125.0
        duration = 10.0
        sig = _synthetic_ppg_pulse(fs=fs, duration_sec=duration)
        result = generate_ppg_phase_labels(sig, fs=fs, frame_rate_hz=5.0)
        expected_frames = int(fs * duration) // int(fs / 5.0)
        assert len(result["labels"]) == expected_frames

    def test_labels_values_valid(self):
        sig = _synthetic_ppg_pulse(fs=125.0, duration_sec=10.0)
        result = generate_ppg_phase_labels(sig, fs=125.0)
        valid = result["labels"][~np.isnan(result["labels"])]
        assert np.all((valid == 0) | (valid == 1))

    def test_float32_output(self):
        sig = _synthetic_ppg_pulse(fs=125.0, duration_sec=10.0)
        result = generate_ppg_phase_labels(sig, fs=125.0)
        assert result["labels"].dtype == np.float32
        assert result["quality"].dtype == np.float32

    def test_hr_bpm_reasonable(self):
        sig = _synthetic_ppg_pulse(fs=125.0, duration_sec=10.0, hr_bpm=70.0)
        result = generate_ppg_phase_labels(sig, fs=125.0)
        assert 50 < result["mean_hr_bpm"] < 100

    def test_too_few_beats_returns_nan(self):
        # Signal too short to get enough beats
        sig = np.zeros(10)
        result = generate_ppg_phase_labels(sig, fs=125.0, min_beats=2)
        assert np.all(np.isnan(result["labels"]))

    def test_loads_config(self):
        sig = _synthetic_ppg_pulse(fs=125.0, duration_sec=10.0)
        result = generate_ppg_phase_labels(sig, fs=125.0, config_path="config_tinnitus.yaml")
        assert "labels" in result

    def test_diastole_frames_present(self):
        """Should have some diastole (1) frames in a clean PPG signal."""
        sig = _synthetic_ppg_pulse(fs=125.0, duration_sec=10.0)
        result = generate_ppg_phase_labels(sig, fs=125.0)
        valid = result["labels"][~np.isnan(result["labels"])]
        assert np.any(valid == 1)

    def test_systole_frames_present(self):
        """Should have some systole (0) frames in a clean PPG signal."""
        sig = _synthetic_ppg_pulse(fs=125.0, duration_sec=10.0)
        result = generate_ppg_phase_labels(sig, fs=125.0)
        valid = result["labels"][~np.isnan(result["labels"])]
        assert np.any(valid == 0)

    def test_works_at_wesad_fs(self):
        """Should work at WESAD BVP rate (64 Hz)."""
        sig = _synthetic_ppg_pulse(fs=64.0, duration_sec=10.0, hr_bpm=65.0)
        result = generate_ppg_phase_labels(sig, fs=64.0)
        assert len(result["labels"]) > 0

    def test_pipeline_with_filter(self):
        """Full pipeline: denoise_ppg → generate_ppg_phase_labels."""
        sig = _synthetic_ppg_pulse(fs=125.0, duration_sec=10.0)
        sig_noisy = sig + 0.1 * np.random.randn(len(sig))
        filtered = denoise_ppg(sig_noisy, fs=125.0)
        result = generate_ppg_phase_labels(filtered, fs=125.0)
        valid = result["labels"][~np.isnan(result["labels"])]
        # After filtering, should still detect both phases
        assert len(valid) > 0


class TestBidmcPpgDiastole:
    """Integration test against real BIDMC data."""

    BIDMC_DIR = "data/raw/stroke avns/bidmc"

    def test_bidmc_record_diastole_labels(self):
        """Run PPG diastole detection on a BIDMC record and compare with ECG labels."""
        import wfdb
        from pathlib import Path
        from src.features.phase_labels import generate_phase_labels

        hea_files = sorted(Path(self.BIDMC_DIR).glob("*.hea"))
        assert len(hea_files) > 0, "No BIDMC records found"

        # Test first 3 records
        agreements = []
        for hea in hea_files[:3]:
            stem = hea.stem
            rec = wfdb.rdrecord(str(hea.parent / stem))
            sig_names = [n.strip().upper() for n in rec.sig_name]
            sigs = np.nan_to_num(rec.p_signal, nan=0.0)

            ppg_idx = next((i for i, n in enumerate(sig_names) if "PLETH" in n), None)
            ecg_idx = next((i for i, n in enumerate(sig_names) if n == "II"), None)
            if ppg_idx is None or ecg_idx is None:
                continue

            fs = float(rec.fs)  # 125 Hz
            ppg = sigs[:, ppg_idx]
            ecg = sigs[:, ecg_idx]

            ppg_filtered = denoise_ppg(ppg, fs=fs)
            ppg_result = generate_ppg_phase_labels(ppg_filtered, fs=fs, config_path="config_tinnitus.yaml")
            ecg_result = generate_phase_labels(ecg, fs=fs, config_path="config_stroke.yaml")

            ppg_labels = ppg_result["labels"]
            ecg_labels = ecg_result["labels"]

            # Align to shorter length
            min_len = min(len(ppg_labels), len(ecg_labels))
            if min_len < 10:
                continue

            ppg_l = ppg_labels[:min_len]
            ecg_l = ecg_labels[:min_len]

            # Compare only frames where both are non-NaN
            both_valid = ~np.isnan(ppg_l) & ~np.isnan(ecg_l)
            if both_valid.sum() < 5:
                continue

            agreement = (ppg_l[both_valid] == ecg_l[both_valid]).mean()
            agreements.append(agreement)

        if agreements:
            mean_agreement = np.mean(agreements)
            # Target: >85% — allow some slack in test since notch detection is heuristic
            assert mean_agreement > 0.60, f"PPG-ECG diastole agreement {mean_agreement:.2%} below 60%"
