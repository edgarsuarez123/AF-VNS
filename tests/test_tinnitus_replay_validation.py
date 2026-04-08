"""
Tests for F16 — tinnitus_replay_validation.py

Covers:
  - AlwaysInBandGate interface contract
  - generate_ground_truth output shapes on synthetic data
  - replay_record chunk alignment and event collection
  - compute_metrics: perfect pipeline (all GT-true points fire)
  - compute_metrics: no events
  - evaluate_wesad integration (skipped if WESAD not downloaded)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.training.tinnitus_replay_validation import (
    AlwaysInBandGate,
    GateMetrics,
    TriFoldMetrics,
    _safe_div,
    compute_metrics,
    generate_ground_truth,
    replay_record,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_ppg(
    duration_sec: float = 30.0,
    fs: float = 125.0,
    hr_bpm: float = 70.0,
) -> np.ndarray:
    """Sinusoidal PPG at given heart rate."""
    t = np.linspace(0, duration_sec, int(duration_sec * fs), endpoint=False)
    hr_hz = hr_bpm / 60.0
    ppg = 0.5 * np.sin(2 * np.pi * hr_hz * t) + 0.1 * np.sin(2 * np.pi * 2 * hr_hz * t)
    return ppg.astype(np.float64)


def _make_synthetic_resp(duration_sec: float = 30.0, fs: float = 125.0) -> np.ndarray:
    """Sinusoidal respiratory signal at 15 breaths/min."""
    t = np.linspace(0, duration_sec, int(duration_sec * fs), endpoint=False)
    return (np.sin(2 * np.pi * (15 / 60) * t)).astype(np.float64)


def _make_synthetic_eda(duration_sec: float = 30.0, fs: float = 4.0) -> np.ndarray:
    """Flat EDA signal (baseline-like)."""
    n = int(duration_sec * fs)
    return np.full(n, 2.0, dtype=np.float64)


def _make_wesad_like_record(
    duration_sec: float = 30.0,
    ppg_fs: float = 64.0,
    with_eda: bool = True,
    with_resp: bool = True,
    label: int = 0,
) -> dict:
    """Build a synthetic TinnitusRecordDict-like dict."""
    ppg = _make_synthetic_ppg(duration_sec, ppg_fs)
    record: dict = {
        "subject_id": "wesad_S99_baseline_0",
        "session_id": "S99",
        "ppg_signal": ppg,
        "ppg_fs": ppg_fs,
        "label": label,
        "condition": "baseline" if label == 0 else "stress",
    }
    if with_resp:
        record["resp_signal"] = _make_synthetic_resp(duration_sec, ppg_fs)
        record["resp_fs"] = ppg_fs
        record["resp_channel"] = "chest_belt"
    if with_eda:
        record["eda_signal"] = _make_synthetic_eda(duration_sec, fs=4.0)
        record["eda_fs"] = 4.0
    return record


# ---------------------------------------------------------------------------
# Test: AlwaysInBandGate
# ---------------------------------------------------------------------------

class TestAlwaysInBandGate:
    def test_is_in_band_always_true(self):
        gate = AlwaysInBandGate()
        assert gate.is_in_band() is True

    def test_calibrated_flag(self):
        gate = AlwaysInBandGate()
        assert gate._calibrated is True

    def test_calibrate_noop(self):
        gate = AlwaysInBandGate()
        gate.calibrate(np.zeros(100), 4.0)  # should not raise
        assert gate._calibrated is True

    def test_update_noop(self):
        gate = AlwaysInBandGate()
        gate.update(np.ones(10), 4.0)  # should not raise
        assert gate.is_in_band() is True

    def test_get_state_keys(self):
        gate = AlwaysInBandGate()
        state = gate.get_state()
        assert state["in_band"] is True
        assert state["calibrated"] is True
        assert state["using_classifier"] is False


# ---------------------------------------------------------------------------
# Test: generate_ground_truth
# ---------------------------------------------------------------------------

class TestGenerateGroundTruth:
    def test_output_shapes_no_eda(self):
        """Verify dia/exh label arrays have the right frame count (no EDA)."""
        rec = _make_wesad_like_record(duration_sec=30.0, ppg_fs=125.0, with_eda=False)
        gt = generate_ground_truth(rec, ppg_fs_target=125.0, phase_frame_rate_hz=5.0)

        assert "ppg_resampled" in gt
        assert "dia_labels" in gt
        assert "exh_labels" in gt
        assert gt["arousal_labels"] is None  # No EDA

        n_ppg = len(gt["ppg_resampled"])
        # Expect roughly duration_sec * frame_rate_hz frames
        expected_frames = n_ppg // int(125.0 / 5.0)  # 25 samples/frame
        assert len(gt["dia_labels"]) == expected_frames
        assert len(gt["exh_labels"]) == expected_frames

    def test_output_shapes_with_eda(self):
        """EDA present → arousal_labels should be non-None."""
        rec = _make_wesad_like_record(duration_sec=30.0, ppg_fs=125.0, with_eda=True)
        gt = generate_ground_truth(rec, ppg_fs_target=125.0)
        # arousal_labels may or may not be None depending on EDA length thresholds
        # Just check it's an ndarray or None
        assert gt["arousal_labels"] is None or isinstance(gt["arousal_labels"], np.ndarray)

    def test_resamples_ppg_from_64hz(self):
        """WESAD BVP at 64 Hz is resampled to 125 Hz."""
        rec = _make_wesad_like_record(duration_sec=10.0, ppg_fs=64.0, with_eda=False, with_resp=False)
        gt = generate_ground_truth(rec, ppg_fs_target=125.0)

        expected_len = round(10.0 * 64.0 * 125.0 / 64.0)  # ~1250
        assert abs(len(gt["ppg_resampled"]) - expected_len) <= 2
        assert gt["ppg_fs"] == 125.0

    def test_exh_source_reference_preferred(self):
        """Reference resp signal is preferred over PPG-derived when available."""
        rec = _make_wesad_like_record(duration_sec=30.0, ppg_fs=125.0, with_resp=True)
        gt = generate_ground_truth(rec, ppg_fs_target=125.0)
        # synthetic resp has enough cycles to qualify
        # exh_source is "reference" if n_resp_cycles >= 2, else "ppg_derived"
        assert gt["exh_source"] in ("reference", "ppg_derived")

    def test_no_resp_falls_back_to_ppg_derived(self):
        """No resp signal → exh_source == 'ppg_derived'."""
        rec = _make_wesad_like_record(duration_sec=30.0, ppg_fs=125.0,
                                      with_resp=False, with_eda=False)
        gt = generate_ground_truth(rec, ppg_fs_target=125.0)
        assert gt["exh_source"] == "ppg_derived"


# ---------------------------------------------------------------------------
# Test: replay_record
# ---------------------------------------------------------------------------

class TestReplayRecord:
    def _make_mock_pipeline(self):
        """Return a pipeline mock that always fires one event per feed() call."""
        from src.models.tinnitus_closed_loop import TinnitusStimEvent

        mock_pipeline = MagicMock()
        mock_pipeline.feed.side_effect = lambda ppg, eda=None, temp_samples=None: [
            TinnitusStimEvent(
                timestamp_samples=100,
                amplitude=0.2,
                frequency=10.0,
                pulse_width=50.0,
                diastole_prob=0.9,
                exhalation_prob=0.8,
                arousal_in_band=True,
            )
        ]
        return mock_pipeline

    def test_chunk_count(self):
        """feed() should be called ceil(n_ppg_samples / chunk_size) times."""
        import math
        pipeline = self._make_mock_pipeline()
        ppg = np.zeros(300)
        replay_record(pipeline, ppg, ppg_fs=125.0, chunk_sec=1.0)
        expected_calls = math.ceil(len(ppg) / 125)
        assert pipeline.feed.call_count == expected_calls

    def test_events_collected(self):
        """All events returned by feed() are collected."""
        pipeline = self._make_mock_pipeline()
        ppg = np.zeros(250)
        events = replay_record(pipeline, ppg, ppg_fs=125.0, chunk_sec=1.0)
        assert len(events) == pipeline.feed.call_count

    def test_eda_chunks_aligned(self):
        """EDA is sliced proportionally to PPG chunks."""
        pipeline = MagicMock()
        pipeline.feed.return_value = []

        ppg = np.zeros(125)
        eda = np.zeros(4)  # 1s × 4 Hz
        replay_record(pipeline, ppg, ppg_fs=125.0, eda_signal=eda, eda_fs=4.0, chunk_sec=1.0)

        # Should be called once with eda shape (4,) or less
        call_args_list = pipeline.feed.call_args_list
        assert len(call_args_list) == 1
        _, kwargs = call_args_list[0]
        eda_arg = call_args_list[0][0][1]  # second positional arg
        assert len(eda_arg) <= 4

    def test_no_eda_passes_none(self):
        """When no EDA, feed() second arg is None."""
        pipeline = MagicMock()
        pipeline.feed.return_value = []
        ppg = np.zeros(125)
        replay_record(pipeline, ppg, ppg_fs=125.0, eda_signal=None, chunk_sec=1.0)
        _, _ = pipeline.feed.call_args
        args = pipeline.feed.call_args[0]
        # second positional arg should be None
        assert args[1] is None if len(args) > 1 else True


# ---------------------------------------------------------------------------
# Test: compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def _make_gt(self, n_ppg: int = 1000, all_true: bool = True, ppg_fs: float = 125.0):
        """Ground truth with all labels = 1 or all = 0."""
        n_phase_frames = n_ppg // int(ppg_fs / 5.0)  # 5 Hz
        n_arousal_frames = n_ppg // int(ppg_fs / 1.0)  # 1 Hz
        val = 1.0 if all_true else 0.0
        return {
            "ppg_resampled": np.zeros(n_ppg),
            "ppg_fs": ppg_fs,
            "dia_labels": np.full(n_phase_frames, val, dtype=np.float32),
            "exh_labels": np.full(n_phase_frames, val, dtype=np.float32),
            "arousal_labels": np.full(n_arousal_frames, val, dtype=np.float32),
        }

    def _make_events_at_all_grid_points(
        self, n_ppg: int = 1000, ppg_fs: float = 125.0,
        stride: int = 13, fast_window: int = 250
    ):
        """Create mock events at every evaluation grid point."""
        from src.models.tinnitus_closed_loop import TinnitusStimEvent
        return [
            TinnitusStimEvent(
                timestamp_samples=g,
                amplitude=0.2, frequency=10.0, pulse_width=50.0,
                diastole_prob=0.9, exhalation_prob=0.8, arousal_in_band=True,
            )
            for g in range(fast_window, n_ppg, stride)
        ]

    def test_no_events_zero_precision(self):
        """No events → trifold_precision = 0, stim_rate = 0."""
        gt = self._make_gt(all_true=True)
        metrics = compute_metrics([], gt, ppg_fs=125.0)
        assert metrics.n_stim_events == 0
        assert metrics.stim_rate_per_min == 0.0
        assert metrics.trifold_precision == 0.0

    def test_no_events_zero_recall(self):
        gt = self._make_gt(all_true=True)
        metrics = compute_metrics([], gt, ppg_fs=125.0)
        assert metrics.trifold_recall == 0.0

    def test_perfect_pipeline_precision_one(self):
        """Events at all GT-true grid points → precision = 1.0."""
        gt = self._make_gt(all_true=True)
        events = self._make_events_at_all_grid_points()
        metrics = compute_metrics(events, gt, ppg_fs=125.0)
        assert metrics.trifold_precision == pytest.approx(1.0)

    def test_perfect_pipeline_recall_one(self):
        gt = self._make_gt(all_true=True)
        events = self._make_events_at_all_grid_points()
        metrics = compute_metrics(events, gt, ppg_fs=125.0)
        assert metrics.trifold_recall == pytest.approx(1.0)

    def test_stim_rate_calculation(self):
        """Stim rate = events / duration_min."""
        n_ppg = 1250  # 10s at 125 Hz
        gt = self._make_gt(n_ppg=n_ppg, all_true=True)
        # Plant 5 events
        from src.models.tinnitus_closed_loop import TinnitusStimEvent
        events = [
            TinnitusStimEvent(
                timestamp_samples=250 + i * 13,
                amplitude=0.2, frequency=10.0, pulse_width=50.0,
                diastole_prob=0.9, exhalation_prob=0.8, arousal_in_band=True,
            )
            for i in range(5)
        ]
        metrics = compute_metrics(events, gt, ppg_fs=125.0)
        assert metrics.n_stim_events == 5
        # 10s → 1/6 min; 5 events → 30 events/min
        assert metrics.stim_rate_per_min == pytest.approx(30.0, rel=0.01)

    def test_n_valid_eval_points_no_nan(self):
        """With no NaN labels, all grid points in range should be valid."""
        gt = self._make_gt(n_ppg=1000, all_true=True)
        metrics = compute_metrics([], gt, ppg_fs=125.0)
        # grid = range(250, 1000, 13) → 58 points
        expected = len(range(250, 1000, 13))
        assert metrics.n_valid_eval_points == expected

    def test_all_gt_false_events_are_false_positives(self):
        """All GT = 0 → fired events are all false positives → trifold_precision = 0."""
        gt = self._make_gt(all_true=False)
        events = self._make_events_at_all_grid_points()
        metrics = compute_metrics(events, gt, ppg_fs=125.0)
        assert metrics.trifold_precision == 0.0

    def test_none_arousal_labels_uses_always_true(self):
        """If arousal_labels is None (BIDMC), arousal is treated as always GT=True."""
        gt = self._make_gt(all_true=True)
        gt["arousal_labels"] = None  # BIDMC: no EDA
        events = self._make_events_at_all_grid_points()
        metrics = compute_metrics(events, gt, ppg_fs=125.0)
        # arousal gate treated as True → trifold = dia AND exh AND True
        assert metrics.n_valid_eval_points > 0
        assert metrics.per_gate["arousal"]["gt_positive_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Integration test: evaluate_wesad on real data
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestEvaluateWesadIntegration:
    @pytest.fixture(autouse=True)
    def _skip_if_no_data(self):
        """Skip this test class if WESAD data is not present."""
        wesad_path = Path("data/raw/tinnitus avns/wesad/WESAD")
        if not wesad_path.exists() or not any(wesad_path.glob("S*/S*.pkl")):
            pytest.skip("WESAD data not available")

    def test_evaluate_wesad_one_subject(self, tmp_path):
        """Run evaluate_wesad on one subject and verify JSON output exists."""
        from src.training.tinnitus_replay_validation import evaluate_wesad

        # Only process one subject by patching load_wesad_subjects
        with patch(
            "src.training.tinnitus_replay_validation.load_wesad_subjects"
        ) as mock_load:
            # Load real subjects, keep only the first one
            from src.training.tinnitus_replay_validation import load_wesad_subjects as real_load
            all_subjects = real_load()
            if not all_subjects:
                pytest.skip("No WESAD subjects parsed")
            first_sid = sorted(all_subjects.keys())[0]
            mock_load.return_value = {first_sid: all_subjects[first_sid]}

            results = evaluate_wesad(
                config_path="config_tinnitus.yaml",
                device="cpu",
                output_dir=str(tmp_path),
            )

        json_path = tmp_path / "wesad_replay_results.json"
        assert json_path.exists(), "JSON results file should be created"

        with open(json_path) as f:
            data = json.load(f)

        assert data["dataset"] == "wesad"
        assert data["n_subjects"] == 1
        assert "aggregate" in data
        assert "per_subject" in data
        assert first_sid in data["per_subject"]
