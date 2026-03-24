"""
Tests for phase_labels.py (diastolic phase) and edr.py (exhalation phase).

Unit tests use synthetic ECG via neurokit2. Integration tests hit real CVES data
and are excluded from CI by default (@pytest.mark.integration).

Step S-15 of the stroke AVNS pipeline.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.phase_labels import (
    generate_phase_labels,
    get_rpeak_indices,
    get_twave_offsets,
)
from src.features.edr import (
    extract_edr,
    generate_exhalation_labels,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_phase_config(tmp_dir: str, **overrides) -> str:
    """Write a minimal config YAML with phase_detection + edr sections.

    Any key in overrides is patched into the appropriate section.
    Keys prefixed with 'edr_' go into the edr section (stripped);
    all others go into phase_detection.
    """
    phase_cfg = {
        "frame_rate_hz": 5.0,
        "fallback_fraction": 0.40,
        "min_beats": 2,
        "min_hr_bpm": 40.0,
        "max_hr_bpm": 200.0,
        "delineate_method": "dwt",
        "denoise_before_delineate": False,  # off by default in tests for speed
    }
    edr_cfg = {
        "method": "vangent2019",
        "min_resp_rate_bpm": 6.0,
        "max_resp_rate_bpm": 30.0,
        "min_resp_cycles": 2,
        "denoise_before_edr": False,  # off by default in tests for speed
    }
    for k, v in overrides.items():
        if k.startswith("edr_"):
            edr_cfg[k[4:]] = v
        else:
            phase_cfg[k] = v

    cfg = {"phase_detection": phase_cfg, "edr": edr_cfg}
    path = str(Path(tmp_dir) / "test_config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def _synth_ecg(duration: int = 30, fs: int = 250, heart_rate: int = 72) -> np.ndarray:
    """Generate synthetic ECG via neurokit2."""
    import neurokit2 as nk
    ecg = nk.ecg_simulate(duration=duration, sampling_rate=fs, heart_rate=heart_rate)
    return np.asarray(ecg, dtype=np.float64)


FS = 250.0


# ===========================================================================
# get_rpeak_indices
# ===========================================================================

class TestGetRpeakIndices:

    def test_dtype_int64(self):
        ecg = _synth_ecg()
        peaks = get_rpeak_indices(ecg, FS)
        assert peaks.dtype == np.int64

    def test_count_matches_hr(self):
        duration = 30
        hr = 72
        ecg = _synth_ecg(duration=duration, heart_rate=hr)
        peaks = get_rpeak_indices(ecg, FS)
        expected = hr / 60 * duration  # 36
        assert abs(len(peaks) - expected) <= 3, f"Expected ~{expected}, got {len(peaks)}"

    def test_sorted_within_bounds(self):
        ecg = _synth_ecg()
        peaks = get_rpeak_indices(ecg, FS)
        assert len(peaks) > 0
        assert np.all(np.diff(peaks) > 0), "Peaks not monotonically increasing"
        assert peaks[0] >= 0
        assert peaks[-1] < len(ecg)

    def test_empty_for_dc_signal(self):
        dc = np.zeros(int(FS * 30), dtype=np.float64)
        peaks = get_rpeak_indices(dc, FS)
        assert len(peaks) == 0


# ===========================================================================
# get_twave_offsets
# ===========================================================================

class TestGetTwaveOffsets:

    @pytest.fixture()
    def ecg_and_peaks(self):
        ecg = _synth_ecg()
        peaks = get_rpeak_indices(ecg, FS)
        return ecg, peaks

    def test_shape_matches_rpeaks(self, ecg_and_peaks):
        ecg, peaks = ecg_and_peaks
        t_off, fb = get_twave_offsets(ecg, peaks, FS)
        assert len(t_off) == len(peaks)
        assert len(fb) == len(peaks)

    def test_dtypes(self, ecg_and_peaks):
        ecg, peaks = ecg_and_peaks
        t_off, fb = get_twave_offsets(ecg, peaks, FS)
        assert t_off.dtype == np.int64
        assert fb.dtype == bool

    def test_after_rpeaks(self, ecg_and_peaks):
        ecg, peaks = ecg_and_peaks
        t_off, _ = get_twave_offsets(ecg, peaks, FS)
        assert np.all(t_off > peaks), "T-wave offsets must be after R-peaks"

    def test_before_next_rpeak(self, ecg_and_peaks):
        ecg, peaks = ecg_and_peaks
        t_off, _ = get_twave_offsets(ecg, peaks, FS)
        for i in range(len(peaks) - 1):
            assert t_off[i] < peaks[i + 1], f"Beat {i}: t_off {t_off[i]} >= next R {peaks[i+1]}"

    def test_within_signal_bounds(self, ecg_and_peaks):
        ecg, peaks = ecg_and_peaks
        t_off, _ = get_twave_offsets(ecg, peaks, FS)
        assert np.all(t_off < len(ecg))

    def test_valid_offsets_on_noise(self):
        """Random noise with manually placed rpeaks → offsets still valid."""
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(int(FS * 10)).astype(np.float64)
        # Place rpeaks at regular 250-sample intervals (HR~60)
        rpeaks = np.arange(125, len(signal) - 125, 250, dtype=np.int64)
        t_off, fb = get_twave_offsets(signal, rpeaks, FS, fallback_fraction=0.40)
        # Regardless of fallback status, offsets must be structurally valid
        assert np.all(t_off > rpeaks), "T-offsets must be after R-peaks"
        assert np.all(t_off < len(signal)), "T-offsets must be within signal"
        for i in range(len(rpeaks) - 1):
            assert t_off[i] < rpeaks[i + 1], f"Beat {i}: overlap into next cycle"

    def test_single_beat(self):
        """Single rpeak uses 1-second default RR."""
        ecg = _synth_ecg(duration=5)
        rpeak = np.array([int(FS * 2)], dtype=np.int64)  # one peak at 2s
        t_off, fb = get_twave_offsets(ecg, rpeak, FS, fallback_fraction=0.40)
        assert len(t_off) == 1
        # Fallback: rpeak + 0.40 * fs (1-sec default RR)
        expected = rpeak[0] + int(0.40 * FS)
        assert t_off[0] == expected or fb[0]  # should be fallback


# ===========================================================================
# generate_phase_labels
# ===========================================================================

class TestGeneratePhaseLabels:

    @pytest.fixture()
    def result_normal(self):
        ecg = _synth_ecg(duration=30)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_phase_config(tmp)
            return generate_phase_labels(ecg, FS, config_path=cfg)

    def test_keys(self, result_normal):
        expected_keys = {"labels", "quality", "n_beats", "fallback_fraction", "mean_hr_bpm"}
        assert set(result_normal.keys()) == expected_keys

    def test_shapes(self, result_normal):
        r = result_normal
        assert r["labels"].dtype == np.float32
        assert r["quality"].dtype == np.float32
        assert len(r["labels"]) == len(r["quality"])
        # 30s * 250Hz / 50 samples-per-frame = 150 frames
        assert len(r["labels"]) == 150

    def test_values_binary_or_nan(self, result_normal):
        labels = result_normal["labels"]
        valid = labels[~np.isnan(labels)]
        assert set(valid.tolist()).issubset({0.0, 1.0})

    def test_has_both_phases(self, result_normal):
        labels = result_normal["labels"]
        valid = labels[~np.isnan(labels)]
        assert 0.0 in valid, "No systole frames found"
        assert 1.0 in valid, "No diastole frames found"

    def test_diastole_dominates(self, result_normal):
        labels = result_normal["labels"]
        valid = labels[~np.isnan(labels)]
        diastole_frac = (valid == 1.0).sum() / len(valid)
        assert diastole_frac > 0.4, f"Diastole fraction too low: {diastole_frac:.2f}"

    def test_hr_in_range(self, result_normal):
        hr = result_normal["mean_hr_bpm"]
        assert 60 <= hr <= 85, f"HR {hr:.1f} outside [60, 85]"

    def test_quality_nonzero(self, result_normal):
        assert result_normal["quality"].mean() > 0

    def test_too_short_nan(self):
        short = np.zeros(int(FS * 1.5), dtype=np.float64)  # 1.5s < 2s
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_phase_config(tmp)
            r = generate_phase_labels(short, FS, config_path=cfg)
        assert r["n_beats"] == 0
        assert np.isnan(r["mean_hr_bpm"])
        if len(r["labels"]) > 0:
            assert np.all(np.isnan(r["labels"]))

    def test_no_rpeaks_nan(self):
        dc = np.zeros(int(FS * 30), dtype=np.float64)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_phase_config(tmp)
            r = generate_phase_labels(dc, FS, config_path=cfg)
        assert r["n_beats"] == 0
        assert np.all(np.isnan(r["labels"]))

    def test_low_hr_nan(self):
        """Config max_hr=50 but ECG has HR=72 → exceeds max → NaN."""
        ecg = _synth_ecg(duration=30, heart_rate=72)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_phase_config(tmp, max_hr_bpm=50.0)
            r = generate_phase_labels(ecg, FS, config_path=cfg)
        assert np.all(np.isnan(r["labels"]))

    def test_high_hr_nan(self):
        """Config min_hr=80 but ECG has HR=72 → below min → NaN."""
        ecg = _synth_ecg(duration=30, heart_rate=72)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_phase_config(tmp, min_hr_bpm=80.0)
            r = generate_phase_labels(ecg, FS, config_path=cfg)
        assert np.all(np.isnan(r["labels"]))

    def test_min_beats_nan(self):
        """Config min_beats=100 on short signal → not enough beats → NaN."""
        ecg = _synth_ecg(duration=5, heart_rate=72)  # ~6 beats
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_phase_config(tmp, min_beats=100)
            r = generate_phase_labels(ecg, FS, config_path=cfg)
        assert np.all(np.isnan(r["labels"]))

    def test_denoise_off(self):
        """denoise_before_delineate=false still produces valid labels."""
        ecg = _synth_ecg(duration=30)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_phase_config(tmp, denoise_before_delineate=False)
            r = generate_phase_labels(ecg, FS, config_path=cfg)
        valid = r["labels"][~np.isnan(r["labels"])]
        assert len(valid) > 0
        assert 0.0 in valid and 1.0 in valid

    def test_reproducible(self):
        ecg = _synth_ecg(duration=30)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_phase_config(tmp)
            r1 = generate_phase_labels(ecg, FS, config_path=cfg)
            r2 = generate_phase_labels(ecg, FS, config_path=cfg)
        np.testing.assert_array_equal(r1["labels"], r2["labels"])
        np.testing.assert_array_equal(r1["quality"], r2["quality"])


# ===========================================================================
# extract_edr
# ===========================================================================

class TestExtractEdr:

    @pytest.fixture()
    def edr_result(self):
        ecg = _synth_ecg(duration=60)  # need longer signal for EDR
        return extract_edr(ecg, FS)

    def test_signal_length(self, edr_result):
        edr_sig, _, _ = edr_result
        expected = int(FS * 60)
        assert len(edr_sig) == expected

    def test_signal_dtype_float64(self, edr_result):
        edr_sig, _, _ = edr_result
        assert edr_sig.dtype == np.float64

    def test_peaks_troughs_int64(self, edr_result):
        _, peaks, troughs = edr_result
        assert peaks.dtype == np.int64
        assert troughs.dtype == np.int64

    def test_peaks_within_bounds(self, edr_result):
        edr_sig, peaks, troughs = edr_result
        n = len(edr_sig)
        if len(peaks) > 0:
            assert peaks[0] >= 0
            assert peaks[-1] < n
        if len(troughs) > 0:
            assert troughs[0] >= 0
            assert troughs[-1] < n

    def test_peaks_sorted(self, edr_result):
        _, peaks, troughs = edr_result
        if len(peaks) > 1:
            assert np.all(np.diff(peaks) > 0)
        if len(troughs) > 1:
            assert np.all(np.diff(troughs) > 0)

    def test_short_signal_zeros(self):
        """Signal with < 2 R-peaks → zeros EDR, empty peak/trough arrays."""
        # Use a flat DC signal long enough for neurokit2 internals but with no R-peaks
        short = np.zeros(int(FS * 5), dtype=np.float64)
        edr_sig, peaks, troughs = extract_edr(short, FS)
        assert np.all(edr_sig == 0.0)
        assert len(peaks) == 0
        assert len(troughs) == 0


# ===========================================================================
# QRS amplitude modulation EDR (S-62)
# ===========================================================================

class TestQrsAmplitudeEdr:
    """Tests for method='amplitude' and method='fusion' in extract_edr."""

    @pytest.fixture()
    def long_ecg(self):
        return _synth_ecg(duration=60)

    def test_amplitude_method_length(self, long_ecg):
        edr_sig, _, _ = extract_edr(long_ecg, FS, method="amplitude")
        assert len(edr_sig) == len(long_ecg)

    def test_amplitude_method_dtype(self, long_ecg):
        edr_sig, _, _ = extract_edr(long_ecg, FS, method="amplitude")
        assert edr_sig.dtype == np.float64

    def test_amplitude_method_peaks_int64(self, long_ecg):
        _, peaks, troughs = extract_edr(long_ecg, FS, method="amplitude")
        assert peaks.dtype == np.int64
        assert troughs.dtype == np.int64

    def test_amplitude_method_peaks_within_bounds(self, long_ecg):
        edr_sig, peaks, troughs = extract_edr(long_ecg, FS, method="amplitude")
        n = len(edr_sig)
        if len(peaks) > 0:
            assert peaks.min() >= 0 and peaks.max() < n
        if len(troughs) > 0:
            assert troughs.min() >= 0 and troughs.max() < n

    def test_amplitude_method_not_constant(self, long_ecg):
        """QRS-AM signal should show variation — not flat."""
        edr_sig, _, _ = extract_edr(long_ecg, FS, method="amplitude")
        assert edr_sig.std() > 0.0

    def test_fusion_method_length(self, long_ecg):
        edr_sig, _, _ = extract_edr(long_ecg, FS, method="fusion")
        assert len(edr_sig) == len(long_ecg)

    def test_fusion_method_dtype(self, long_ecg):
        edr_sig, _, _ = extract_edr(long_ecg, FS, method="fusion")
        assert edr_sig.dtype == np.float64

    def test_fusion_method_not_constant(self, long_ecg):
        edr_sig, _, _ = extract_edr(long_ecg, FS, method="fusion")
        assert edr_sig.std() > 0.0

    def test_short_signal_amplitude_zeros(self):
        """Signal with < 2 R-peaks → zero EDR, empty arrays (amplitude method)."""
        short = np.zeros(int(FS * 5), dtype=np.float64)
        edr_sig, peaks, troughs = extract_edr(short, FS, method="amplitude")
        assert np.all(edr_sig == 0.0)
        assert len(peaks) == 0
        assert len(troughs) == 0

    def test_known_amplitude_modulation(self):
        """Synthetic ECG with sinusoidal R-peak amplitude at 0.2 Hz.
        QRS-AM should produce a signal correlated with the modulation."""
        import neurokit2 as nk
        from src.features.phase_labels import get_rpeak_indices

        fs = 250.0
        duration = 60
        t = np.arange(int(fs * duration)) / fs
        # Clean synthetic ECG
        ecg = nk.ecg_simulate(duration=duration, sampling_rate=int(fs), heart_rate=70, noise=0.01)
        ecg = np.asarray(ecg, dtype=np.float64)

        # Get R-peaks and scale amplitudes with a 0.2 Hz sine
        rpeaks = get_rpeak_indices(ecg, fs)
        if len(rpeaks) < 5:
            pytest.skip("Not enough R-peaks in synthetic ECG")
        resp_freq = 0.2  # Hz
        modulation = 0.3 * np.sin(2 * np.pi * resp_freq * rpeaks / fs)
        ecg_mod = ecg.copy()
        for i, rp in enumerate(rpeaks):
            ecg_mod[max(0, rp-5):min(len(ecg), rp+5)] *= (1.0 + modulation[i])

        edr_sig, _, _ = extract_edr(ecg_mod, fs, method="amplitude")

        # Sample the EDR at R-peak positions and check correlation with modulation
        if len(edr_sig) > 0 and len(rpeaks) > 5:
            edr_at_peaks = edr_sig[rpeaks]
            correlation = np.corrcoef(edr_at_peaks, modulation)[0, 1]
            # QRS-AM should show non-trivial correlation (unit test — real validation in S-63 smoke test)
            assert abs(correlation) > 0.05, f"Expected |corr|>0.05, got {correlation:.3f}"


# ===========================================================================
# generate_exhalation_labels
# ===========================================================================

class TestGenerateExhalationLabels:

    @pytest.fixture()
    def result_normal(self):
        ecg = _synth_ecg(duration=60)  # 60s for reliable respiratory detection
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_phase_config(tmp)
            return generate_exhalation_labels(ecg, FS, config_path=cfg)

    def test_keys(self, result_normal):
        expected_keys = {"labels", "quality", "n_resp_cycles", "resp_rate_bpm"}
        assert set(result_normal.keys()) == expected_keys

    def test_shapes(self, result_normal):
        r = result_normal
        assert r["labels"].dtype == np.float32
        assert r["quality"].dtype == np.float32
        assert len(r["labels"]) == len(r["quality"])
        # 60s * 250Hz / 50 samples-per-frame = 300 frames
        assert len(r["labels"]) == 300

    def test_values_binary_or_nan(self, result_normal):
        labels = result_normal["labels"]
        valid = labels[~np.isnan(labels)]
        assert set(valid.tolist()).issubset({0.0, 1.0})

    def test_has_both_phases(self, result_normal):
        labels = result_normal["labels"]
        valid = labels[~np.isnan(labels)]
        assert 0.0 in valid, "No inhale frames found"
        assert 1.0 in valid, "No exhale frames found"

    def test_resp_rate_in_range(self, result_normal):
        rate = result_normal["resp_rate_bpm"]
        if not np.isnan(rate):
            assert 6 <= rate <= 30, f"Resp rate {rate:.1f} outside [6, 30]"

    def test_n_resp_cycles_positive(self, result_normal):
        assert result_normal["n_resp_cycles"] >= 2

    def test_too_short_nan(self):
        short = np.zeros(int(FS * 5), dtype=np.float64)  # 5s < 10s minimum
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_phase_config(tmp)
            r = generate_exhalation_labels(short, FS, config_path=cfg)
        assert r["n_resp_cycles"] == 0
        assert np.isnan(r["resp_rate_bpm"])
        if len(r["labels"]) > 0:
            assert np.all(np.isnan(r["labels"]))

    def test_few_cycles_nan(self):
        """Config min_resp_cycles=100 → not enough cycles → NaN."""
        ecg = _synth_ecg(duration=60)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_phase_config(tmp, edr_min_resp_cycles=100)
            r = generate_exhalation_labels(ecg, FS, config_path=cfg)
        assert np.all(np.isnan(r["labels"]))

    def test_low_resp_rate_nan(self):
        """Config min_resp_rate=20 with typical ~12 bpm EDR → below min → NaN."""
        ecg = _synth_ecg(duration=60)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_phase_config(tmp, edr_min_resp_rate_bpm=25.0)
            r = generate_exhalation_labels(ecg, FS, config_path=cfg)
        # If the resp rate is below 25, should be NaN. If above, test is still valid.
        rate = r["resp_rate_bpm"]
        if not np.isnan(rate) and rate < 25.0:
            assert np.all(np.isnan(r["labels"]))

    def test_denoise_off(self):
        ecg = _synth_ecg(duration=60)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_phase_config(tmp, edr_denoise_before_edr=False)
            r = generate_exhalation_labels(ecg, FS, config_path=cfg)
        # Should produce valid output (may or may not have both phases)
        assert r["labels"].dtype == np.float32

    def test_exhale_convention(self):
        """Verify exhale=1 convention (inverted from neurokit2)."""
        ecg = _synth_ecg(duration=60)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_phase_config(tmp)
            r = generate_exhalation_labels(ecg, FS, config_path=cfg)
        labels = r["labels"]
        valid = labels[~np.isnan(labels)]
        if len(valid) > 0:
            assert 1.0 in valid, "No exhale (1.0) frames — convention may be wrong"

    def test_reproducible(self):
        ecg = _synth_ecg(duration=60)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_phase_config(tmp)
            r1 = generate_exhalation_labels(ecg, FS, config_path=cfg)
            r2 = generate_exhalation_labels(ecg, FS, config_path=cfg)
        np.testing.assert_array_equal(r1["labels"], r2["labels"])
        np.testing.assert_array_equal(r1["quality"], r2["quality"])


# ===========================================================================
# Integration tests (real CVES data)
# ===========================================================================

@pytest.mark.integration
class TestCvesIntegration:
    """Run against first real CVES record. Skipped if data not on disk."""

    CVES_DIR = Path("data/raw/stroke avns/cves/data")

    @pytest.fixture(scope="class")
    def cves_ecg(self):
        """Load first CVES record that yields valid R-peaks."""
        import wfdb
        if not self.CVES_DIR.is_dir():
            pytest.skip("CVES data not found — skipping integration test")
        hea_files = sorted(self.CVES_DIR.rglob("*.hea"))
        if len(hea_files) == 0:
            pytest.skip("No .hea files in CVES directory")
        # Try up to 10 records to find one with detectable ECG
        for hea in hea_files[:10]:
            rec_name = str(hea.with_suffix(""))
            record = wfdb.rdrecord(rec_name)
            # Find ECG channel by name; fall back to channel 0
            ecg_idx = 0
            for i, name in enumerate(record.sig_name):
                if name.lower() == "ecg":
                    ecg_idx = i
                    break
            signal = record.p_signal[:, ecg_idx].astype(np.float64)
            fs = float(record.fs)
            peaks = get_rpeak_indices(signal, fs)
            if len(peaks) >= 10:
                rr_sec = np.diff(peaks) / fs
                hr = 60.0 / np.mean(rr_sec)
                if 40 <= hr <= 200:
                    return signal, fs
        pytest.skip("No usable CVES records found (need >=10 R-peaks with HR 40-200)")

    def test_phase_labels_cves(self, cves_ecg):
        signal, fs = cves_ecg
        r = generate_phase_labels(signal, fs, config_path="config_stroke.yaml")
        assert r["n_beats"] > 5
        assert 40 <= r["mean_hr_bpm"] <= 200
        labels = r["labels"][~np.isnan(r["labels"])]
        assert len(labels) > 0
        assert 0.0 in labels and 1.0 in labels

    def test_exhalation_labels_cves(self, cves_ecg):
        signal, fs = cves_ecg
        r = generate_exhalation_labels(signal, fs, config_path="config_stroke.yaml")
        if r["n_resp_cycles"] >= 2:
            assert 6 <= r["resp_rate_bpm"] <= 30
            labels = r["labels"][~np.isnan(r["labels"])]
            assert 0.0 in labels and 1.0 in labels
        else:
            # Short record — just verify structure
            assert "labels" in r and "quality" in r
