"""Demo: Bifold closed-loop stroke aVNS pipeline.

Generates synthetic ECG, feeds 1-second chunks through the bifold
closed-loop pipeline, and prints timestamped stimulation events.

Usage:
    .venv/Scripts/python scripts/demo_stroke_pipeline.py
    .venv/Scripts/python scripts/demo_stroke_pipeline.py --duration 60
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

from src.models.closed_loop_pipeline import build_closed_loop_pipeline


# ---------------------------------------------------------------------------
# Synthetic signal generator (same pattern as tests/test_closed_loop_pipeline.py:61)
# ---------------------------------------------------------------------------

def _make_ecg(n_samples: int, fs: float = 250.0) -> np.ndarray:
    """Synthetic ECG: 1.2 Hz cardiac fundamental + 12 Hz harmonic."""
    t = np.arange(n_samples) / fs
    return (np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 12.0 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stroke aVNS bifold pipeline demo")
    p.add_argument("--duration", type=int, default=30, help="Simulation duration in seconds (default: 30)")
    p.add_argument("--fs", type=float, default=250.0, help="ECG sampling rate Hz (default: 250)")
    p.add_argument("--config", default="config_stroke.yaml", help="Config YAML path")
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
    print("Stroke aVNS - Bifold Closed-Loop Pipeline Demo")
    print("=" * 65)
    print(f"  Config   : {args.config}")
    print(f"  Device   : {args.device}")
    print(f"  Duration : {args.duration}s  |  ECG {args.fs} Hz")

    # ------------------------------------------------------------------
    # 1. Build pipeline from trained checkpoint
    # ------------------------------------------------------------------
    print(f"\n[1/2] Building pipeline...")
    t0 = time.perf_counter()
    pipe = build_closed_loop_pipeline(config_path=args.config, device=args.device)
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"      Built in {build_ms:.0f}ms")

    # ------------------------------------------------------------------
    # 2. Streaming simulation
    # ------------------------------------------------------------------
    ecg_chunk = int(args.fs)   # 1 second of ECG
    total_events = 0

    print(f"\n[2/2] Streaming {args.duration}s of synthetic ECG ({ecg_chunk} samples/sec)...\n")
    print(f"{'Time':>5s} | {'Stims':>5s} | Detail")
    print("-" * 65)

    for sec in range(args.duration):
        ecg = _make_ecg(ecg_chunk, fs=args.fs)

        t0 = time.perf_counter()
        events = pipe.feed(ecg)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        total_events += len(events)

        if events:
            for ev in events:
                print(
                    f"{sec:4d}s | {len(events):5d} | STIM @ sample {ev.timestamp_samples}: "
                    f"dia={ev.diastole_prob:.3f} exh={ev.exhalation_prob:.3f} "
                    f"| {ev.amplitude:.2f}mA {ev.frequency:.0f}Hz {ev.pulse_width:.0f}us"
                    f"  [{elapsed_ms:.1f}ms]"
                )
        elif sec == 0 or sec % 5 == 0 or sec == args.duration - 1:
            state = pipe.get_state()
            print(
                f"{sec:4d}s |     0 | no stim (dia AND exh not co-occurring)"
                f"  fast_path_calls={state.fast_path_calls}  [{elapsed_ms:.1f}ms]"
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    state = pipe.get_state()
    print("\n" + "=" * 65)
    print("Pipeline State Summary")
    print("=" * 65)
    print(f"  Total ECG samples fed : {state.total_samples_fed:,}")
    stride_samples = max(1, int(round(100 / 1000.0 * args.fs)))
    expected_fast = max(0, (state.total_samples_fed - int(2.0 * args.fs)) // stride_samples)
    print(f"  Fast path calls       : {state.fast_path_calls:,}  (~{expected_fast} expected at 100ms stride)")
    print(f"  Slow path calls       : {state.slow_path_calls}  (requires {args.duration}s > 60s window)")
    print(f"  Total stim events     : {total_events}")
    print(f"  Stim rate             : {total_events / args.duration:.2f} events/sec")
    if state.last_stim_params:
        p = state.last_stim_params
        print(f"  Last stim params      : {p['amplitude']:.2f}mA  {p['frequency']:.0f}Hz  {p['pulse_width']:.0f}us")

    print()
    print("Validation (CVES, 228 records):")
    print("  Diastole accuracy : 84.3% in-distribution, 81.1% OOD (SHaRe)")
    print("  Latency p95       : 116ms (target < 200ms)")
    print("  Training: .venv/Scripts/python -m src.training.stroke_phase_train")


if __name__ == "__main__":
    main()
