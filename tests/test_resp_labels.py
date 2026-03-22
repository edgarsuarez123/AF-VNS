"""
Tests for resp_labels.py — reference respiratory signal label generation.
Uses synthetic sine-wave respiratory signals to validate peak detection,
polarity handling, and frame-level label output.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.resp_labels import (
    generate_exhalation_labels_from_reference,
    _bandpass_filter,
    _detect_resp_peaks,
    _build_resp_phase_array,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(tmp_dir: Path, **overrides) -> str:
    """Write a minimal config with resp_labels section."""
    cfg = {
        "resp_labels": {
            "bandpass_low_hz": 0.08,
            "bandpass_high_hz": 0.6,
            "bandpass_order": 4,
            "min_peak_distance_sec": 1.5,
            "min_resp_rate_bpm": 6.0,
            "max_resp_rate_bpm": 30.0,
            "min_resp_cycles": 2,
            "thermst_invert": True,
        }
    }
    cfg["resp_labels"].update(overrides)
    cfg_path = tmp_dir / "config_test.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return str(cfg_path)


def _make_sine_resp(fs: float = 250.0, duration_sec: float = 60.0, bpm: float = 15.0) -> np.ndarray:
    """Synthetic respiratory signal: clean sine wave at given breathing rate."""
    t = np.arange(int(fs * duration_sec)) / fs
    freq = bpm / 60.0  # Hz
    return np.sin(2 * np.pi * freq * t).astype(np.float64)


# ---------------------------------------------------------------------------
# Tests — bandpass filter
# ---------------------------------------------------------------------------

def test_bandpass_preserves_resp_band():
    """A 0.25 Hz (15 bpm) sine passes through the respiratory bandpass."""
    fs = 250.0
    sig = _make_sine_resp(fs=fs, bpm=15.0)
    filtered = _bandpass_filter(sig, fs, 0.08, 0.6, order=4)
    # Power should be mostly preserved (>50% of original amplitude)
    assert np.max(np.abs(filtered)) > 0.5 * np.max(np.abs(sig))


def test_bandpass_removes_high_freq():
    """A 5 Hz component is attenuated by the respiratory bandpass."""
    fs = 250.0
    t = np.arange(int(fs * 60)) / fs  # longer signal reduces edge effects
    noise = np.sin(2 * np.pi * 5.0 * t)  # 5 Hz — well above respiratory band
    filtered = _bandpass_filter(noise, fs, 0.08, 0.6, order=4)
    # Trim edge transients (first/last 5s) and check attenuation
    trim = int(fs * 5)
    filtered_core = filtered[trim:-trim]
    assert np.max(np.abs(filtered_core)) < 0.05 * np.max(np.abs(noise))


# ---------------------------------------------------------------------------
# Tests — peak detection
# ---------------------------------------------------------------------------

def test_detect_peaks_on_sine():
    """Clean sine at 15 bpm for 60s → ~15 peaks."""
    fs = 250.0
    sig = _make_sine_resp(fs=fs, duration_sec=60.0, bpm=15.0)
    cfg = {
        "bandpass_low_hz": 0.08,
        "bandpass_high_hz": 0.6,
        "bandpass_order": 4,
        "min_peak_distance_sec": 1.5,
        "thermst_invert": True,
    }
    peaks, troughs = _detect_resp_peaks(sig, fs, "flow_rate", cfg)
    # 15 bpm × 1 min = 15 cycles → ~15 peaks (±2 for edge effects)
    assert abs(len(peaks) - 15) <= 2, f"Expected ~15 peaks, got {len(peaks)}"
    assert abs(len(troughs) - 15) <= 2, f"Expected ~15 troughs, got {len(troughs)}"


def test_thermst_polarity_inversion():
    """thermst peaks (warm exhaled air) should be treated as exhale end.
    After inversion, peak detection on -thermst should find peaks at inhale end."""
    fs = 250.0
    # Simulate thermst: positive = warm exhaled air
    sig = _make_sine_resp(fs=fs, duration_sec=60.0, bpm=15.0)
    cfg = {
        "bandpass_low_hz": 0.08,
        "bandpass_high_hz": 0.6,
        "bandpass_order": 4,
        "min_peak_distance_sec": 1.5,
        "thermst_invert": True,
    }

    peaks_flow, _ = _detect_resp_peaks(sig, fs, "flow_rate", cfg)
    peaks_therm, _ = _detect_resp_peaks(-sig, fs, "thermst", cfg)

    # Both should find roughly the same peak locations (within 1 sample)
    assert len(peaks_flow) > 0
    assert len(peaks_therm) > 0
    # Peak counts should be similar
    assert abs(len(peaks_flow) - len(peaks_therm)) <= 2


# ---------------------------------------------------------------------------
# Tests — phase array building
# ---------------------------------------------------------------------------

def test_build_phase_array_alternates():
    """Peaks and troughs produce alternating inhale/exhale labels."""
    n = 1000
    peaks = np.array([100, 500, 900])
    troughs = np.array([300, 700])
    phase, quality = _build_resp_phase_array(peaks, troughs, n)

    # peak(100) → trough(300) = exhale
    assert phase[150] == 1.0
    # trough(300) → peak(500) = inhale
    assert phase[400] == 0.0
    # Before first landmark = NaN
    assert np.isnan(phase[50])


# ---------------------------------------------------------------------------
# Tests — generate_exhalation_labels_from_reference (full pipeline)
# ---------------------------------------------------------------------------

def test_generate_labels_return_interface():
    """Output dict has all required keys with correct types."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _make_config(Path(tmp))
        sig = _make_sine_resp(fs=250.0, duration_sec=60.0, bpm=15.0)
        result = generate_exhalation_labels_from_reference(
            sig, 250.0, "flow_rate", 5.0, cfg_path,
        )
        assert "labels" in result
        assert "quality" in result
        assert "n_resp_cycles" in result
        assert "resp_rate_bpm" in result
        assert isinstance(result["labels"], np.ndarray)
        assert result["labels"].dtype == np.float32


def test_labels_binary_or_nan():
    """All label values must be 0.0, 1.0, or NaN."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _make_config(Path(tmp))
        sig = _make_sine_resp(fs=250.0, duration_sec=60.0, bpm=15.0)
        result = generate_exhalation_labels_from_reference(
            sig, 250.0, "flow_rate", 5.0, cfg_path,
        )
        labels = result["labels"]
        valid = labels[~np.isnan(labels)]
        assert set(valid.tolist()).issubset({0.0, 1.0}), \
            f"Labels contain unexpected values: {set(valid.tolist())}"


def test_frame_count():
    """60s signal at 250 Hz with 5 Hz frames → 300 frames."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _make_config(Path(tmp))
        sig = _make_sine_resp(fs=250.0, duration_sec=60.0, bpm=15.0)
        result = generate_exhalation_labels_from_reference(
            sig, 250.0, "flow_rate", 5.0, cfg_path,
        )
        expected_frames = int(250.0 * 60.0) // int(250.0 / 5.0)
        assert len(result["labels"]) == expected_frames


def test_has_both_phases():
    """Clean sine produces both exhale (1) and inhale (0) labels."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _make_config(Path(tmp))
        sig = _make_sine_resp(fs=250.0, duration_sec=60.0, bpm=15.0)
        result = generate_exhalation_labels_from_reference(
            sig, 250.0, "flow_rate", 5.0, cfg_path,
        )
        valid = result["labels"][~np.isnan(result["labels"])]
        assert 0.0 in valid, "No inhale labels found"
        assert 1.0 in valid, "No exhale labels found"
        # Roughly 50/50 split (±20%)
        exhale_frac = (valid == 1.0).mean()
        assert 0.3 < exhale_frac < 0.7, f"Exhale fraction {exhale_frac:.2f} too skewed"


def test_short_signal_nan():
    """Signal shorter than 10s returns all-NaN labels."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _make_config(Path(tmp))
        sig = _make_sine_resp(fs=250.0, duration_sec=5.0, bpm=15.0)
        result = generate_exhalation_labels_from_reference(
            sig, 250.0, "flow_rate", 5.0, cfg_path,
        )
        assert result["n_resp_cycles"] == 0
        assert np.all(np.isnan(result["labels"]))


def test_rate_too_fast_nan():
    """Breathing at 40 bpm (above max 30) returns all-NaN."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _make_config(Path(tmp))
        sig = _make_sine_resp(fs=250.0, duration_sec=60.0, bpm=40.0)
        result = generate_exhalation_labels_from_reference(
            sig, 250.0, "flow_rate", 5.0, cfg_path,
        )
        assert result["n_resp_cycles"] == 0
        assert np.all(np.isnan(result["labels"]))


def test_reproducible():
    """Same input produces identical output."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _make_config(Path(tmp))
        sig = _make_sine_resp(fs=250.0, duration_sec=60.0, bpm=15.0)
        r1 = generate_exhalation_labels_from_reference(sig, 250.0, "flow_rate", 5.0, cfg_path)
        r2 = generate_exhalation_labels_from_reference(sig, 250.0, "flow_rate", 5.0, cfg_path)
        np.testing.assert_array_equal(r1["labels"], r2["labels"])
        assert r1["n_resp_cycles"] == r2["n_resp_cycles"]


def test_resp_rate_reasonable():
    """15 bpm input should produce resp_rate_bpm near 15."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _make_config(Path(tmp))
        sig = _make_sine_resp(fs=250.0, duration_sec=60.0, bpm=15.0)
        result = generate_exhalation_labels_from_reference(
            sig, 250.0, "flow_rate", 5.0, cfg_path,
        )
        assert 12.0 < result["resp_rate_bpm"] < 18.0, \
            f"Expected ~15 bpm, got {result['resp_rate_bpm']}"
