"""Demo: Tri-fold closed-loop tinnitus aVNS pipeline.

Generates synthetic PPG and EDA, feeds 1-second chunks through the pipeline,
and prints timestamped stimulation events and gate decisions.

Usage:
    .venv/Scripts/python scripts/demo_tinnitus_pipeline.py
    .venv/Scripts/python scripts/demo_tinnitus_pipeline.py --duration 60 --device cpu
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.tinnitus_closed_loop import build_tinnitus_closed_loop_pipeline


# ---------------------------------------------------------------------------
# Synthetic signal generators (same patterns as tests/test_tinnitus_closed_loop.py)
# ---------------------------------------------------------------------------

def _make_ppg(n_samples: int, fs: float = 125.0) -> np.ndarray:
    """Synthetic PPG: 1.2 Hz cardiac + 12 Hz harmonic (dicrotic notch proxy)."""
    t = np.arange(n_samples) / fs
    return (np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 12.0 * t)).astype(np.float32)


def _make_eda(n_samples: int, value: float = 2.0) -> np.ndarray:
    """Flat EDA baseline at given SCL (µS)."""
    return np.full(n_samples, value, dtype=np.float64)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tinnitus aVNS tri-fold pipeline demo")
    p.add_argument("--duration", type=int, default=30, help="Simulation duration in seconds (default: 30)")
    p.add_argument("--ppg-fs", type=float, default=125.0, help="PPG sampling rate Hz (default: 125)")
    p.add_argument("--eda-fs", type=float, default=4.0, help="EDA sampling rate Hz (default: 4)")
    p.add_argument("--calibration-sec", type=int, default=1200, help="EDA calibration duration sec (default: 1200)")
    p.add_argument("--config", default="config_tinnitus.yaml", help="Config YAML path")
    p.add_argument("--device", default="cpu", help="PyTorch device (default: cpu)")
    p.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s")

    print("=" * 65)
    print("Tinnitus aVNS - Tri-Fold Closed-Loop Pipeline Demo")
    print("=" * 65)
    print(f"  Config : {args.config}")
    print(f"  Device : {args.device}")
    print(f"  Duration: {args.duration}s  |  PPG {args.ppg_fs} Hz  |  EDA {args.eda_fs} Hz")

    # ------------------------------------------------------------------
    # 1. Build pipeline from trained checkpoints
    # ------------------------------------------------------------------
    print(f"\n[1/3] Building pipeline...")
    t0 = time.perf_counter()
    pipe = build_tinnitus_closed_loop_pipeline(config_path=args.config, device=args.device)
    build_ms = (time.perf_counter() - t0) * 1000
    using_clf = pipe._arousal_gate._classifier is not None
    print(f"      Built in {build_ms:.0f}ms")
    print(f"      Arousal classifier: {'GBT (trained)' if using_clf else 'rule-based fallback'}")

    # ------------------------------------------------------------------
    # 2. Calibrate EDA arousal gate
    # ------------------------------------------------------------------
    cal_samples = int(args.calibration_sec * args.eda_fs)
    print(f"\n[2/3] Calibrating arousal gate with {args.calibration_sec}s flat EDA ({cal_samples} samples)...")
    cal_eda = _make_eda(cal_samples, value=2.0)
    pipe.calibrate_eda(cal_eda, fs=args.eda_fs)
    # Prime the gate with a small update so is_in_band() returns True immediately
    pipe._arousal_gate.update(_make_eda(int(args.eda_fs * 2), value=2.0), args.eda_fs)
    print(f"      EDA gate calibrated: {pipe.get_state().eda_calibrated}")

    # ------------------------------------------------------------------
    # 3. Streaming simulation
    # ------------------------------------------------------------------
    ppg_chunk = int(args.ppg_fs)   # 1 second of PPG
    eda_chunk = int(args.eda_fs)   # 1 second of EDA
    total_events = 0

    print(f"\n[3/3] Streaming {args.duration}s of synthetic data...\n")
    print(f"{'Time':>5s} | {'Stims':>5s} | Detail")
    print("-" * 65)

    for sec in range(args.duration):
        ppg = _make_ppg(ppg_chunk, fs=args.ppg_fs)
        eda = _make_eda(eda_chunk, value=2.0)

        t0 = time.perf_counter()
        events = pipe.feed(ppg, eda)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        total_events += len(events)

        if events:
            for ev in events:
                print(
                    f"{sec:4d}s | {len(events):5d} | STIM @ sample {ev.timestamp_samples}: "
                    f"dia={ev.diastole_prob:.3f} exh={ev.exhalation_prob:.3f} "
                    f"arousal={'IN' if ev.arousal_in_band else 'OUT'} "
                    f"| {ev.amplitude:.2f}mA {ev.frequency:.0f}Hz {ev.pulse_width:.0f}us"
                    f"  [{elapsed_ms:.1f}ms]"
                )
        elif sec == 0 or sec % 5 == 0 or sec == args.duration - 1:
            state = pipe.get_state()
            print(
                f"{sec:4d}s |     0 | no stim — fast_path_calls={state.fast_path_calls}"
                f"  [{elapsed_ms:.1f}ms]"
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    state = pipe.get_state()
    print("\n" + "=" * 65)
    print("Pipeline State Summary")
    print("=" * 65)
    print(f"  Total PPG samples fed : {state.total_samples_fed:,}")
    stride_samples = max(1, int(round(100 / 1000.0 * args.ppg_fs)))
    expected_fast = max(0, (state.total_samples_fed - int(2.0 * args.ppg_fs)) // stride_samples)
    print(f"  Fast path calls       : {state.fast_path_calls:,}  (~{expected_fast} expected at 100ms stride)")
    print(f"  Slow path calls       : {state.slow_path_calls}  (requires {args.duration}s > 60s window)")
    print(f"  Total stim events     : {total_events}")
    print(f"  Stim rate             : {total_events / args.duration:.2f} events/sec")
    if state.last_stim_params:
        p = state.last_stim_params
        print(f"  Last stim params      : {p['amplitude']:.2f}mA  {p['frequency']:.0f}Hz  {p['pulse_width']:.0f}us")

    if total_events == 0:
        print()
        print("NOTE: Zero stim events is expected with synthetic data.")
        print("  The consecutive-frame gate (N=3, diastole_threshold=0.65) requires")
        print("  sustained high diastole probability across 600ms of PPG — a constraint")
        print("  that the trained model (avg_acc=0.617) rarely satisfies on sine waves.")
        print("  On real PPG recordings, stim rate is ~3.2/min (F22 WESAD replay).")


if __name__ == "__main__":
    main()
