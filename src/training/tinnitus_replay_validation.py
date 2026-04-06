"""
F16 — End-to-End Offline Replay Validation for Tinnitus Tri-Fold Pipeline.

Feeds real WESAD (PPG 64→125 Hz + EDA 4 Hz + reference resp) and BIDMC
(PPG 125 Hz, phase-only) recordings through TinnitusClosedLoopPipeline.feed()
in 1-second streaming chunks, compares stim events against independently-generated
ground truth labels, and reports per-gate and tri-fold metrics.

Usage:
    python -m src.training.tinnitus_replay_validation --dataset wesad
    python -m src.training.tinnitus_replay_validation --dataset bidmc --device cpu
    python -m src.training.tinnitus_replay_validation --dataset all --output-dir results/
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import resample as scipy_resample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AlwaysInBandGate — drop-in mock for BIDMC (no EDA available)
# ---------------------------------------------------------------------------

class AlwaysInBandGate:
    """Drop-in ArousalGate replacement that always returns in-band.

    Used for BIDMC phase-only validation where no EDA signal is available.
    Implements the same interface as ArousalGate so TinnitusClosedLoopPipeline
    accepts it without modification.
    """

    _calibrated: bool = True

    def calibrate(self, *args, **kwargs) -> None:  # noqa: D401
        pass

    def update(self, *args, **kwargs) -> None:  # noqa: D401
        pass

    def is_in_band(self) -> bool:
        return True

    def get_state(self) -> dict:
        return {
            "in_band": True,
            "calibrated": True,
            "using_classifier": False,
            "current_features": None,
        }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_wesad_subjects(
    config_path: str = "config_tinnitus.yaml",
) -> dict[str, list]:
    """Load WESAD records grouped by session_id (e.g. 'S2').

    Baseline (label=0) epochs are sorted first within each subject so they can
    be used for EDA calibration before replaying stress/other epochs.

    Returns
    -------
    dict mapping session_id -> list of TinnitusRecordDicts
    """
    from src.data.dataset_parsers import load_config
    from src.data.tinnitus_parsers import parse_wesad_dir

    cfg = load_config(config_path)
    wesad_dir = cfg["data"]["wesad_subdir"]
    records = parse_wesad_dir(wesad_dir, config_path=config_path)

    by_subject: dict[str, list] = {}
    for rec in records:
        sid = rec["session_id"]
        by_subject.setdefault(sid, []).append(rec)

    # Sort baseline (label=0) first — needed for calibrate_eda() call
    for sid in by_subject:
        by_subject[sid].sort(key=lambda r: (r["label"], r["subject_id"]))

    logger.info("Loaded %d WESAD subjects, %d total epochs",
                len(by_subject), sum(len(v) for v in by_subject.values()))
    return by_subject


def load_bidmc_records(
    config_path: str = "config_tinnitus.yaml",
) -> list:
    """Load BIDMC records via parse_bidmc_ppg_dir()."""
    from src.data.dataset_parsers import load_config
    from src.data.tinnitus_parsers import parse_bidmc_ppg_dir

    cfg = load_config(config_path)
    bidmc_dir = cfg["data"]["bidmc_subdir"]
    records = parse_bidmc_ppg_dir(bidmc_dir, config_path=config_path)
    logger.info("Loaded %d BIDMC records", len(records))
    return records


# ---------------------------------------------------------------------------
# Pipeline factory helpers
# ---------------------------------------------------------------------------

def _build_wesad_pipeline(
    config_path: str,
    checkpoint_path: Optional[str],
    device: str,
):
    """Build TinnitusClosedLoopPipeline using the standard factory."""
    from src.models.tinnitus_closed_loop import build_tinnitus_closed_loop_pipeline
    return build_tinnitus_closed_loop_pipeline(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )


def _build_bidmc_pipeline(
    config_path: str,
    checkpoint_path: Optional[str],
    device: str,
):
    """Build TinnitusClosedLoopPipeline with AlwaysInBandGate for phase-only use."""
    from src.data.dataset_parsers import load_config
    from src.models.phase_detector import build_phase_detector
    from src.models.autonomic_state import build_autonomic_state
    from src.models.stim_recommender import build_stim_recommender
    from src.models.tinnitus_closed_loop import TinnitusClosedLoopPipeline

    cfg = load_config(config_path)
    cl_cfg = cfg.get("closed_loop", {})

    phase_detector = build_phase_detector(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    phase_detector.eval()

    return TinnitusClosedLoopPipeline(
        phase_detector=phase_detector,
        autonomic_state=build_autonomic_state(config_path=config_path),
        stim_recommender=build_stim_recommender(config_path=config_path),
        arousal_gate=AlwaysInBandGate(),
        ppg_fs=float(cl_cfg.get("ppg_fs", 125.0)),
        eda_fs=float(cl_cfg.get("eda_fs", 4.0)),
        inference_stride_ms=float(cl_cfg.get("inference_stride_ms", 100.0)),
        slow_window_sec=float(cl_cfg.get("slow_window_sec", 60.0)),
        diastole_threshold=float(cl_cfg.get("diastole_threshold", 0.5)),
        exhalation_threshold=float(cl_cfg.get("exhalation_threshold", 0.5)),
        config_path=config_path,
    )


# ---------------------------------------------------------------------------
# Ground truth generation
# ---------------------------------------------------------------------------

def generate_ground_truth(
    record: dict,
    ppg_fs_target: float = 125.0,
    phase_frame_rate_hz: float = 5.0,
    arousal_frame_rate_hz: float = 1.0,
    calibration_eda: Optional[np.ndarray] = None,
    calibration_eda_fs: Optional[float] = None,
    config_path: str = "config_tinnitus.yaml",
) -> dict:
    """Generate independent ground truth labels for one record.

    Resamples PPG (and resp if present) to ppg_fs_target before label generation.

    Returns
    -------
    dict with keys:
        ppg_resampled   ndarray (N,) at ppg_fs_target
        ppg_fs          float
        dia_labels      ndarray (n_phase_frames,) float32 — 1=diastole, NaN=unknown
        exh_labels      ndarray (n_phase_frames,) float32 — 1=exhale, NaN=unknown
        arousal_labels  ndarray (n_arousal_frames,) float32 or None (1=in-band)
        eda_signal      ndarray or None
        eda_fs          float or None
        exh_source      str — "reference" or "ppg_derived"
    """
    from src.features.ppg_phase_labels import generate_ppg_phase_labels
    from src.features.ppg_resp import generate_exhalation_labels_from_ppg
    from src.features.resp_labels import generate_exhalation_labels_from_reference
    from src.features.eda import extract_eda_features

    ppg = np.asarray(record["ppg_signal"], dtype=np.float64)
    src_fs = float(record["ppg_fs"])

    # 1. Resample PPG to target fs if needed
    if abs(src_fs - ppg_fs_target) > 0.5:
        n_target = int(round(len(ppg) * ppg_fs_target / src_fs))
        ppg = scipy_resample(ppg, n_target)

    # 2. Diastole ground truth
    dia_result = generate_ppg_phase_labels(
        ppg,
        ppg_fs_target,
        frame_rate_hz=phase_frame_rate_hz,
        config_path=config_path,
    )
    dia_labels = np.asarray(dia_result["labels"], dtype=np.float32)

    # 3. Exhalation ground truth — prefer reference respiratory signal
    resp = record.get("resp_signal")
    resp_fs = record.get("resp_fs")
    resp_channel = record.get("resp_channel", "chest_belt")
    exh_source = "ppg_derived"

    if resp is not None and resp_fs is not None:
        resp_arr = np.asarray(resp, dtype=np.float64)
        # Resample resp to match PPG target fs if needed
        if abs(float(resp_fs) - ppg_fs_target) > 0.5:
            n_resp_target = int(round(len(resp_arr) * ppg_fs_target / float(resp_fs)))
            resp_arr = scipy_resample(resp_arr, n_resp_target)
        exh_result = generate_exhalation_labels_from_reference(
            resp_arr,
            ppg_fs_target,
            channel_name=resp_channel,
            frame_rate_hz=phase_frame_rate_hz,
            config_path=config_path,
        )
        if exh_result.get("n_resp_cycles", 0) >= 2:
            exh_source = "reference"
        else:
            logger.debug(
                "Reference resp had <2 cycles (%s), falling back to PPG-derived",
                record.get("subject_id", "?"),
            )
            exh_result = generate_exhalation_labels_from_ppg(
                ppg, ppg_fs_target,
                frame_rate_hz=phase_frame_rate_hz,
                config_path=config_path,
            )
    else:
        exh_result = generate_exhalation_labels_from_ppg(
            ppg, ppg_fs_target,
            frame_rate_hz=phase_frame_rate_hz,
            config_path=config_path,
        )

    exh_labels = np.asarray(exh_result["labels"], dtype=np.float32)

    # 4. Arousal ground truth from EDA
    eda_signal = record.get("eda_signal")
    eda_fs = record.get("eda_fs")
    arousal_labels = None

    if eda_signal is not None and eda_fs is not None:
        eda_arr = np.asarray(eda_signal, dtype=np.float64)
        eda_result = extract_eda_features(
            eda_arr,
            float(eda_fs),
            calibration_signal=calibration_eda,
            calibration_fs=calibration_eda_fs,
            config_path=config_path,
        )
        raw = eda_result.get("arousal_in_band")
        if raw is not None:
            arousal_labels = np.asarray(raw, dtype=np.float32)

    return {
        "ppg_resampled": ppg,
        "ppg_fs": ppg_fs_target,
        "dia_labels": dia_labels,
        "exh_labels": exh_labels,
        "arousal_labels": arousal_labels,
        "eda_signal": eda_signal,
        "eda_fs": eda_fs,
        "exh_source": exh_source,
    }


# ---------------------------------------------------------------------------
# Replay engine
# ---------------------------------------------------------------------------

def replay_record(
    pipeline,
    ppg_signal: np.ndarray,
    ppg_fs: float,
    eda_signal: Optional[np.ndarray] = None,
    eda_fs: Optional[float] = None,
    chunk_sec: float = 1.0,
) -> list:
    """Feed one record through the pipeline in streaming 1-second chunks.

    Each chunk feeds chunk_sec * ppg_fs PPG samples and the time-aligned
    EDA samples (chunk_sec * eda_fs). Simulates real-time data acquisition.

    Returns
    -------
    List of all TinnitusStimEvents fired during the replay.
    """
    ppg = np.asarray(ppg_signal, dtype=np.float64)
    chunk_ppg = max(1, int(round(chunk_sec * ppg_fs)))
    chunk_eda = max(1, int(round(chunk_sec * eda_fs))) if (eda_signal is not None and eda_fs) else 0

    events = []
    n_samples = len(ppg)
    n_chunks = math.ceil(n_samples / chunk_ppg)

    for i in range(n_chunks):
        ppg_chunk = ppg[i * chunk_ppg: (i + 1) * chunk_ppg]

        eda_chunk = None
        if eda_signal is not None and chunk_eda > 0:
            eda_chunk = np.asarray(
                eda_signal[i * chunk_eda: (i + 1) * chunk_eda],
                dtype=np.float64,
            )

        fired = pipeline.feed(ppg_chunk, eda_chunk)
        events.extend(fired)

    return events


# ---------------------------------------------------------------------------
# Metrics dataclasses + computation
# ---------------------------------------------------------------------------

@dataclass
class GateMetrics:
    """Per-gate accuracy at fast-path evaluation grid points."""
    name: str
    n_eval_points: int        # Valid GT grid points (non-NaN)
    n_fired_true: int         # Pipeline fired AND GT = True
    n_fired_false: int        # Pipeline fired AND GT = False
    n_missed_true: int        # Pipeline did not fire AND GT = True
    precision: float          # n_fired_true / (n_fired_true + n_fired_false)
    recall: float             # n_fired_true / (n_fired_true + n_missed_true)
    gt_positive_rate: float   # Fraction of GT = True among valid eval points


@dataclass
class TriFoldMetrics:
    """Combined tri-fold pipeline metrics for one record or epoch."""
    duration_sec: float
    n_stim_events: int
    stim_rate_per_min: float
    trifold_precision: float       # Of fired events, fraction where GT all-3 = True
    trifold_recall: float          # Of GT all-3-True points, fraction that fired
    gate_failure_counts: dict      # When gt_all_3=True but no fire: per-gate True counts
    per_gate: dict                 # "diastole"/"exhalation"/"arousal" -> GateMetrics dict
    n_valid_eval_points: int
    n_trifold_gt_true: int         # Eval points where all 3 GT = True


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def compute_metrics(
    events: list,
    ground_truth: dict,
    ppg_fs: float = 125.0,
    stride_samples: int = 13,
    fast_window_samples: int = 250,
    phase_frame_rate_hz: float = 5.0,
    arousal_frame_rate_hz: float = 1.0,
) -> TriFoldMetrics:
    """Compare pipeline stim events against ground truth labels.

    Builds the fast-path evaluation grid (every stride_samples from
    fast_window_samples onward), maps each grid point to the corresponding
    GT frame indices, and computes per-gate and tri-fold precision/recall.

    Parameters
    ----------
    events : list of TinnitusStimEvent from replay_record()
    ground_truth : dict from generate_ground_truth()
    ppg_fs : float — target PPG sampling rate (default 125 Hz)
    stride_samples : int — fast path stride (default 13 = 100ms × 125 Hz)
    fast_window_samples : int — fast path context window (default 250 = 2s × 125 Hz)
    phase_frame_rate_hz : float — GT frame rate for diastole/exhalation
    arousal_frame_rate_hz : float — GT frame rate for arousal
    """
    dia_labels = ground_truth["dia_labels"]
    exh_labels = ground_truth["exh_labels"]
    arousal_labels = ground_truth.get("arousal_labels")
    ppg = ground_truth["ppg_resampled"]
    total_ppg = len(ppg)
    duration_sec = total_ppg / ppg_fs

    samples_per_phase_frame = ppg_fs / phase_frame_rate_hz      # 25 @ 125/5
    samples_per_arousal_frame = ppg_fs / arousal_frame_rate_hz  # 125 @ 125/1

    # Build fast-path evaluation grid: these are the sample indices where the
    # pipeline evaluates (and may fire). timestamp_samples == grid point when it fires.
    grid_points = set(range(fast_window_samples, total_ppg, stride_samples))

    # Map event timestamp → event (events only exist at grid points)
    event_at_grid: dict[int, object] = {e.timestamp_samples: e for e in events}

    # Per-gate counters
    dia_ft = dia_ff = dia_mt = 0
    exh_ft = exh_ff = exh_mt = 0
    ar_ft = ar_ff = ar_mt = 0

    trifold_ft = trifold_ff = trifold_mt = 0
    fail_dia = fail_exh = fail_ar = 0

    dia_gt_pos = exh_gt_pos = ar_gt_pos = 0
    n_valid = 0
    n_trifold_gt_true = 0

    for g in sorted(grid_points):
        phase_idx = int(g / samples_per_phase_frame)
        arousal_idx = int(g / samples_per_arousal_frame)

        # Look up GT — bounds check
        gt_dia = dia_labels[phase_idx] if phase_idx < len(dia_labels) else np.nan
        gt_exh = exh_labels[phase_idx] if phase_idx < len(exh_labels) else np.nan
        if arousal_labels is not None:
            gt_ar = arousal_labels[arousal_idx] if arousal_idx < len(arousal_labels) else np.nan
        else:
            gt_ar = 1.0  # BIDMC / AlwaysInBandGate: always in-band

        # Skip any NaN GT
        if np.isnan(gt_dia) or np.isnan(gt_exh) or np.isnan(gt_ar):
            continue

        n_valid += 1
        gt_dia_b = gt_dia > 0.5
        gt_exh_b = gt_exh > 0.5
        gt_ar_b = gt_ar > 0.5

        if gt_dia_b: dia_gt_pos += 1
        if gt_exh_b: exh_gt_pos += 1
        if gt_ar_b: ar_gt_pos += 1

        all_three = gt_dia_b and gt_exh_b and gt_ar_b
        if all_three:
            n_trifold_gt_true += 1

        fired = g in event_at_grid

        # Per-gate: fired ↔ all 3 gates passed; not-fired ↔ at least one blocked
        if fired:
            dia_ft += int(gt_dia_b); dia_ff += int(not gt_dia_b)
            exh_ft += int(gt_exh_b); exh_ff += int(not gt_exh_b)
            ar_ft  += int(gt_ar_b);  ar_ff  += int(not gt_ar_b)
        else:
            dia_mt += int(gt_dia_b)
            exh_mt += int(gt_exh_b)
            ar_mt  += int(gt_ar_b)

        # Tri-fold
        if fired:
            trifold_ft += int(all_three)
            trifold_ff += int(not all_three)
        else:
            if all_three:
                trifold_mt += 1
                fail_dia += int(gt_dia_b)
                fail_exh += int(gt_exh_b)
                fail_ar  += int(gt_ar_b)

    n_fired = len(events)
    stim_rate = _safe_div(n_fired, duration_sec / 60.0)

    def _gate(name, ft, ff, mt, gp):
        return GateMetrics(
            name=name,
            n_eval_points=n_valid,
            n_fired_true=ft,
            n_fired_false=ff,
            n_missed_true=mt,
            precision=_safe_div(ft, ft + ff),
            recall=_safe_div(ft, ft + mt),
            gt_positive_rate=_safe_div(gp, n_valid),
        )

    per_gate = {
        "diastole":   _gate("diastole",   dia_ft, dia_ff, dia_mt, dia_gt_pos),
        "exhalation": _gate("exhalation", exh_ft, exh_ff, exh_mt, exh_gt_pos),
        "arousal":    _gate("arousal",    ar_ft,  ar_ff,  ar_mt,  ar_gt_pos),
    }

    return TriFoldMetrics(
        duration_sec=duration_sec,
        n_stim_events=n_fired,
        stim_rate_per_min=stim_rate,
        trifold_precision=_safe_div(trifold_ft, trifold_ft + trifold_ff),
        trifold_recall=_safe_div(trifold_ft, trifold_ft + trifold_mt),
        gate_failure_counts={"diastole": fail_dia, "exhalation": fail_exh, "arousal": fail_ar},
        per_gate={k: asdict(v) for k, v in per_gate.items()},
        n_valid_eval_points=n_valid,
        n_trifold_gt_true=n_trifold_gt_true,
    )


# ---------------------------------------------------------------------------
# Dataset-level orchestrators
# ---------------------------------------------------------------------------

def evaluate_wesad(
    config_path: str = "config_tinnitus.yaml",
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    output_dir: Optional[str] = None,
) -> dict:
    """Run full WESAD tri-fold replay validation.

    For each subject:
      1. Build pipeline (PhaseDetector CNN + ArousalGate + ArousalClassifier)
      2. Calibrate EDA from the baseline epoch
      3. Generate independent GT for all epochs
      4. Replay each epoch and compute metrics
      5. Collect per-epoch and per-subject results

    Returns aggregate + per-subject results dict.
    """
    from src.data.dataset_parsers import load_config
    from src.models.tinnitus_closed_loop import TinnitusClosedLoopPipeline

    cfg = load_config(config_path)
    cl_cfg = cfg.get("closed_loop", {})
    rv_cfg = cfg.get("replay_validation", {})
    ppg_fs = float(cl_cfg.get("ppg_fs", 125.0))
    stride_ms = float(cl_cfg.get("inference_stride_ms", 100.0))
    stride_samples = max(1, int(round(stride_ms / 1000.0 * ppg_fs)))
    fast_window = int(round(2.0 * ppg_fs))
    chunk_sec = float(rv_cfg.get("chunk_sec", 1.0))
    phase_frame_rate = float(rv_cfg.get("frame_rate_hz", 5.0))
    arousal_frame_rate = float(rv_cfg.get("arousal_frame_rate_hz", 1.0))

    by_subject = load_wesad_subjects(config_path)

    all_results = {}

    for sid, records in sorted(by_subject.items()):
        logger.info("=== WESAD subject %s (%d epochs) ===", sid, len(records))
        t_subject_start = time.time()

        # Build fresh pipeline per subject
        pipeline = _build_wesad_pipeline(config_path, checkpoint_path, device)

        # Find baseline epoch for EDA calibration
        baseline_eda: Optional[np.ndarray] = None
        baseline_eda_fs: Optional[float] = None
        for rec in records:
            if rec["label"] == 0 and rec.get("eda_signal") is not None:
                baseline_eda = np.asarray(rec["eda_signal"], dtype=np.float64)
                baseline_eda_fs = float(rec["eda_fs"])
                break

        if baseline_eda is not None:
            logger.info("  Calibrating EDA from %s baseline (%.0fs @ %.0f Hz)",
                        sid, len(baseline_eda) / baseline_eda_fs, baseline_eda_fs)
            pipeline.calibrate_eda(baseline_eda, baseline_eda_fs)
        else:
            logger.warning("  No baseline EDA found for %s — ArousalClassifier will handle it", sid)

        subject_epochs = []

        for rec in records:
            epoch_label = rec.get("condition", str(rec["label"]))
            logger.info("  Epoch: %s / %s", rec["subject_id"], epoch_label)

            # Reset PPG buffer between epochs (EDA calibration persists)
            pipeline.reset()

            # Generate ground truth (pass baseline EDA for cross-calibration on stress epochs)
            gt = generate_ground_truth(
                rec,
                ppg_fs_target=ppg_fs,
                phase_frame_rate_hz=phase_frame_rate,
                arousal_frame_rate_hz=arousal_frame_rate,
                calibration_eda=baseline_eda,
                calibration_eda_fs=baseline_eda_fs,
                config_path=config_path,
            )

            # Replay
            t0 = time.time()
            events = replay_record(
                pipeline,
                ppg_signal=gt["ppg_resampled"],
                ppg_fs=ppg_fs,
                eda_signal=gt["eda_signal"],
                eda_fs=gt["eda_fs"],
                chunk_sec=chunk_sec,
            )
            elapsed = time.time() - t0

            # Compute metrics
            metrics = compute_metrics(
                events, gt,
                ppg_fs=ppg_fs,
                stride_samples=stride_samples,
                fast_window_samples=fast_window,
                phase_frame_rate_hz=phase_frame_rate,
                arousal_frame_rate_hz=arousal_frame_rate,
            )

            epoch_result = {
                "subject_id": rec["subject_id"],
                "session_id": sid,
                "condition": epoch_label,
                "label": rec["label"],
                "exh_source": gt["exh_source"],
                "replay_sec": round(elapsed, 2),
                **asdict(metrics),
            }
            subject_epochs.append(epoch_result)

            logger.info(
                "    %s  events=%d  stim_rate=%.1f/min  "
                "trifold_P=%.3f  trifold_R=%.3f  "
                "dia_P=%.3f  exh_P=%.3f  ar_P=%.3f  (%.1fs replay)",
                epoch_label,
                metrics.n_stim_events,
                metrics.stim_rate_per_min,
                metrics.trifold_precision,
                metrics.trifold_recall,
                metrics.per_gate["diastole"]["precision"],
                metrics.per_gate["exhalation"]["precision"],
                metrics.per_gate["arousal"]["precision"],
                elapsed,
            )

        subject_wall = time.time() - t_subject_start
        all_results[sid] = {
            "epochs": subject_epochs,
            "wall_sec": round(subject_wall, 2),
        }

    # Aggregate across all epochs
    all_epochs = [ep for sid_res in all_results.values() for ep in sid_res["epochs"]]
    aggregate = _aggregate_metrics(all_epochs)

    result = {
        "dataset": "wesad",
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "device": device,
        "n_subjects": len(all_results),
        "n_epochs": len(all_epochs),
        "per_subject": all_results,
        "aggregate": aggregate,
    }

    if output_dir:
        save_results(result, output_dir, prefix="wesad")

    return result


def evaluate_bidmc(
    config_path: str = "config_tinnitus.yaml",
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    output_dir: Optional[str] = None,
) -> dict:
    """Run BIDMC phase-only replay validation (no EDA — AlwaysInBandGate used).

    Reports diastole + exhalation metrics only.  Arousal gate always returns
    in-band so the pipeline behaves as a bi-fold (dia ∧ exh) gate.
    """
    from src.data.dataset_parsers import load_config

    cfg = load_config(config_path)
    cl_cfg = cfg.get("closed_loop", {})
    rv_cfg = cfg.get("replay_validation", {})
    ppg_fs = float(cl_cfg.get("ppg_fs", 125.0))
    stride_ms = float(cl_cfg.get("inference_stride_ms", 100.0))
    stride_samples = max(1, int(round(stride_ms / 1000.0 * ppg_fs)))
    fast_window = int(round(2.0 * ppg_fs))
    chunk_sec = float(rv_cfg.get("chunk_sec", 1.0))
    phase_frame_rate = float(rv_cfg.get("frame_rate_hz", 5.0))

    records = load_bidmc_records(config_path)
    pipeline = _build_bidmc_pipeline(config_path, checkpoint_path, device)

    all_epochs = []

    for rec in records:
        logger.info("BIDMC record: %s", rec["subject_id"])
        pipeline.reset()

        gt = generate_ground_truth(
            rec,
            ppg_fs_target=ppg_fs,
            phase_frame_rate_hz=phase_frame_rate,
            arousal_frame_rate_hz=1.0,
            config_path=config_path,
        )

        t0 = time.time()
        events = replay_record(
            pipeline,
            ppg_signal=gt["ppg_resampled"],
            ppg_fs=ppg_fs,
            eda_signal=None,
            eda_fs=None,
            chunk_sec=chunk_sec,
        )
        elapsed = time.time() - t0

        metrics = compute_metrics(
            events, gt,
            ppg_fs=ppg_fs,
            stride_samples=stride_samples,
            fast_window_samples=fast_window,
            phase_frame_rate_hz=phase_frame_rate,
            arousal_frame_rate_hz=1.0,
        )

        epoch_result = {
            "subject_id": rec["subject_id"],
            "exh_source": gt["exh_source"],
            "replay_sec": round(elapsed, 2),
            **asdict(metrics),
        }
        all_epochs.append(epoch_result)

        logger.info(
            "  %s  events=%d  stim_rate=%.1f/min  dia_P=%.3f  exh_P=%.3f",
            rec["subject_id"], metrics.n_stim_events, metrics.stim_rate_per_min,
            metrics.per_gate["diastole"]["precision"],
            metrics.per_gate["exhalation"]["precision"],
        )

    aggregate = _aggregate_metrics(all_epochs)

    result = {
        "dataset": "bidmc",
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "device": device,
        "n_records": len(all_epochs),
        "records": all_epochs,
        "aggregate": aggregate,
    }

    if output_dir:
        save_results(result, output_dir, prefix="bidmc")

    return result


def _aggregate_metrics(epochs: list) -> dict:
    """Aggregate TriFoldMetrics across epochs by simple mean of scalar fields."""
    if not epochs:
        return {}

    scalar_keys = [
        "duration_sec", "n_stim_events", "stim_rate_per_min",
        "trifold_precision", "trifold_recall",
        "n_valid_eval_points", "n_trifold_gt_true",
    ]
    gate_keys = ["precision", "recall", "gt_positive_rate"]

    agg: dict = {k: 0.0 for k in scalar_keys}
    gate_agg: dict = {
        g: {k: 0.0 for k in gate_keys}
        for g in ["diastole", "exhalation", "arousal"]
    }

    for ep in epochs:
        for k in scalar_keys:
            agg[k] += ep.get(k, 0.0)
        for g in gate_agg:
            if "per_gate" in ep and g in ep["per_gate"]:
                for k in gate_keys:
                    gate_agg[g][k] += ep["per_gate"][g].get(k, 0.0)

    n = len(epochs)
    for k in scalar_keys:
        agg[k] = round(agg[k] / n, 4)

    for g in gate_agg:
        for k in gate_keys:
            gate_agg[g][k] = round(gate_agg[g][k] / n, 4)

    agg["per_gate_mean"] = gate_agg
    agg["n_epochs"] = n
    return agg


# ---------------------------------------------------------------------------
# Output: JSON + plots
# ---------------------------------------------------------------------------

def save_results(results: dict, output_dir: str, prefix: str = "replay") -> None:
    """Save JSON metrics and generate a summary matplotlib figure."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / f"{prefix}_replay_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results → %s", json_path)

    _save_plots(results, out, prefix)


def _save_plots(results: dict, out: Path, prefix: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plots")
        return

    dataset = results.get("dataset", "unknown")

    if dataset == "wesad":
        _plot_wesad(results, out, prefix)
    else:
        _plot_bidmc(results, out, prefix)


def _plot_wesad(results: dict, out: Path, prefix: str) -> None:
    import matplotlib.pyplot as plt

    per_subject = results.get("per_subject", {})
    sids = sorted(per_subject.keys())

    # Collect per-subject baseline vs stress stim rates
    base_rates, stress_rates = [], []
    base_tp, stress_tp = [], []  # trifold precision
    base_tr, stress_tr = [], []  # trifold recall

    for sid in sids:
        epochs = per_subject[sid]["epochs"]
        base_eps = [e for e in epochs if e["label"] == 0]
        stress_eps = [e for e in epochs if e["label"] == 1]

        def mean_field(eps, key):
            vals = [e[key] for e in eps if key in e]
            return sum(vals) / len(vals) if vals else 0.0

        base_rates.append(mean_field(base_eps, "stim_rate_per_min"))
        stress_rates.append(mean_field(stress_eps, "stim_rate_per_min"))
        base_tp.append(mean_field(base_eps, "trifold_precision"))
        stress_tp.append(mean_field(stress_eps, "trifold_precision"))
        base_tr.append(mean_field(base_eps, "trifold_recall"))
        stress_tr.append(mean_field(stress_eps, "trifold_recall"))

    x = range(len(sids))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Stim rate baseline vs stress
    ax = axes[0]
    ax.bar([i - width/2 for i in x], base_rates, width, label="Baseline", color="steelblue")
    ax.bar([i + width/2 for i in x], stress_rates, width, label="Stress", color="tomato")
    ax.set_xticks(list(x)); ax.set_xticklabels(sids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Stim events / min"); ax.set_title("A — Stim Rate: Baseline vs Stress")
    ax.legend()

    # Panel B: Tri-fold precision per subject
    ax = axes[1]
    ax.bar([i - width/2 for i in x], base_tp, width, label="Baseline", color="steelblue")
    ax.bar([i + width/2 for i in x], stress_tp, width, label="Stress", color="tomato")
    ax.set_ylim(0, 1); ax.set_xticks(list(x))
    ax.set_xticklabels(sids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Tri-fold Precision"); ax.set_title("B — Tri-fold Precision")
    ax.legend()

    # Panel C: Gate failure breakdown (aggregate)
    agg = results.get("aggregate", {})
    gate_mean = agg.get("per_gate_mean", {})
    gate_names = ["diastole", "exhalation", "arousal"]
    precisions = [gate_mean.get(g, {}).get("precision", 0.0) for g in gate_names]
    recalls = [gate_mean.get(g, {}).get("recall", 0.0) for g in gate_names]
    ax = axes[2]
    xi = range(len(gate_names))
    ax.bar([i - width/2 for i in xi], precisions, width, label="Precision", color="steelblue")
    ax.bar([i + width/2 for i in xi], recalls, width, label="Recall", color="seagreen")
    ax.set_ylim(0, 1); ax.set_xticks(list(xi)); ax.set_xticklabels(gate_names)
    ax.set_ylabel("Score"); ax.set_title("C — Per-Gate Precision / Recall (aggregate)")
    ax.legend()

    fig.suptitle(f"WESAD Tri-Fold Replay Validation  ({len(sids)} subjects)", fontsize=12)
    fig.tight_layout()
    plot_path = out / f"{prefix}_replay_summary.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot → %s", plot_path)


def _plot_bidmc(results: dict, out: Path, prefix: str) -> None:
    import matplotlib.pyplot as plt

    records = results.get("records", [])
    if not records:
        return

    gate_names = ["diastole", "exhalation"]
    precisions = {g: [] for g in gate_names}
    recalls = {g: [] for g in gate_names}

    for rec in records:
        pg = rec.get("per_gate", {})
        for g in gate_names:
            precisions[g].append(pg.get(g, {}).get("precision", 0.0))
            recalls[g].append(pg.get(g, {}).get("recall", 0.0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, g in zip(axes, gate_names):
        ax.hist(precisions[g], bins=15, alpha=0.6, label="Precision", color="steelblue")
        ax.hist(recalls[g], bins=15, alpha=0.6, label="Recall", color="seagreen")
        ax.set_xlabel("Score"); ax.set_ylabel("# Records")
        ax.set_title(f"{g.capitalize()} (n={len(records)} records)")
        ax.legend()
    fig.suptitle("BIDMC Phase-Only Replay Validation", fontsize=12)
    fig.tight_layout()
    plot_path = out / f"{prefix}_replay_summary.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot → %s", plot_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="F16 — tinnitus tri-fold offline replay validation"
    )
    parser.add_argument(
        "--dataset", choices=["wesad", "bidmc", "all"], default="wesad",
        help="Dataset to replay (default: wesad)",
    )
    parser.add_argument(
        "--config", default="config_tinnitus.yaml",
        help="Path to config YAML (default: config_tinnitus.yaml)",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to PhaseDetector checkpoint (default: from config)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="PyTorch device string, e.g. 'cpu' or 'cuda' (default: cpu)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for JSON + plots (default: from config replay_validation.output_dir)",
    )
    args = parser.parse_args()

    # Resolve output dir
    if args.output_dir is None:
        try:
            from src.data.dataset_parsers import load_config
            cfg = load_config(args.config)
            args.output_dir = cfg.get("replay_validation", {}).get(
                "output_dir", "models/artifacts/replay_validation"
            )
        except Exception:
            args.output_dir = "models/artifacts/replay_validation"

    datasets = ["wesad", "bidmc"] if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        logger.info("=== Starting %s replay validation ===", ds.upper())
        if ds == "wesad":
            evaluate_wesad(
                config_path=args.config,
                checkpoint_path=args.checkpoint,
                device=args.device,
                output_dir=args.output_dir,
            )
        else:
            evaluate_bidmc(
                config_path=args.config,
                checkpoint_path=args.checkpoint,
                device=args.device,
                output_dir=args.output_dir,
            )

    logger.info("=== Replay validation complete ===")


if __name__ == "__main__":
    main()
