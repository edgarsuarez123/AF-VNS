"""Demo: Tri-fold closed-loop tinnitus aVNS pipeline.

Generates synthetic PPG and EDA, feeds 1-second chunks through the pipeline,
and prints per-second model probabilities plus any stimulation events.

Usage:
    .venv/Scripts/python scripts/demo_tinnitus_pipeline.py
    .venv/Scripts/python scripts/demo_tinnitus_pipeline.py --duration 60 --device cpu
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Suppress noisy 3rd-party warnings before any imports that trigger them
warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*sampled at very low frequency.*")
warnings.filterwarnings("ignore", category=UserWarning, module="neurokit2")

import torch
from src.models.tinnitus_closed_loop import build_tinnitus_closed_loop_pipeline


# ---------------------------------------------------------------------------
# Synthetic signal generators (same patterns as tests/test_tinnitus_closed_loop.py)
# ---------------------------------------------------------------------------

def _make_ppg(n_samples: int, fs: float = 125.0) -> np.ndarray:
    """Synthetic PPG: 1.2 Hz cardiac + 12 Hz harmonic (dicrotic notch proxy)."""
    t = np.arange(n_samples) / fs
    return (np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 12.0 * t)).astype(np.float32)


def _make_eda(n_samples: int, value: float = 2.0) -> np.ndarray:
    """Flat EDA baseline at given SCL (uS)."""
    return np.full(n_samples, value, dtype=np.float64)


def _get_last_probs(pipe) -> tuple[float, float] | None:
    """Extract the most recent diastole/exhalation probabilities from the phase detector."""
    try:
        buf = list(pipe._ppg_buffer)
        if len(buf) < pipe._fast_window_samples:
            return None
        window = np.array(buf[-pipe._fast_window_samples:], dtype=np.float32)
        from src.features.ppg_filter import denoise_ppg
        denoised = denoise_ppg(window, pipe._ppg_fs, config_path=pipe._config_path)
        tensor = torch.tensor(denoised, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(pipe._device)
        with torch.no_grad():
            logits = pipe._phase_detector(tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()  # (10, 2)
        return float(probs[-1, 0]), float(probs[-1, 1])  # last frame: dia, exh
    except Exception:
        return None


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
    p.add_argument("--log-level", default="ERROR", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument(
        "--demo-mode", action="store_true",
        help="Lower gates to F16 settings (dia>0.50, N=1) so stim events fire on synthetic data"
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s")

    print("=" * 70)
    print("Tinnitus aVNS - Tri-Fold Closed-Loop Pipeline Demo")
    print("=" * 70)
    print(f"  Config : {args.config}  |  Device: {args.device}")
    print(f"  PPG 125 Hz + EDA 4 Hz  |  Duration: {args.duration}s")
    print()
    if args.demo_mode:
        print("  Mode: DEMO (F16 gates: dia>0.50, N=1 frame) -- stim events will fire")
    else:
        print("  Mode: CLINICAL (F17 gates: dia>0.65, N=3 frames) -- strict, use --demo-mode to see firing")
    print()
    print("  Gate logic: STIM fires when ALL three are true simultaneously:")
    dia_thresh = 0.50 if args.demo_mode else 0.65
    n_consec = 1 if args.demo_mode else 3
    print(f"    [1] Diastole probability > {dia_thresh} for {n_consec} consecutive frame(s)")
    print(f"    [2] Exhalation probability > 0.50")
    print(f"    [3] EDA arousal in-band (GBT classifier, WESAD-trained)")

    # ------------------------------------------------------------------
    # 1. Build pipeline from trained checkpoints
    # ------------------------------------------------------------------
    print(f"\n[1/3] Building pipeline from trained checkpoints...")
    t0 = time.perf_counter()
    pipe = build_tinnitus_closed_loop_pipeline(config_path=args.config, device=args.device)
    if args.demo_mode:
        # Override F17 clinical gates to F16 settings so events fire on synthetic data
        pipe._dia_threshold = 0.50
        pipe._consecutive_n = 1
    build_ms = (time.perf_counter() - t0) * 1000
    using_clf = pipe._arousal_gate._classifier is not None
    print(f"      Done in {build_ms:.0f}ms")
    print(f"      PhaseDetector CNN loaded: tinnitus_phase_detector.pth")
    print(f"      Arousal classifier: {'GBT trained on WESAD (AUROC 0.930)' if using_clf else 'rule-based fallback'}")

    # ------------------------------------------------------------------
    # 2. Calibrate EDA arousal gate
    # ------------------------------------------------------------------
    cal_samples = int(args.calibration_sec * args.eda_fs)
    print(f"\n[2/3] Calibrating EDA arousal gate with {args.calibration_sec}s baseline ({cal_samples} samples)...")
    pipe.calibrate_eda(_make_eda(cal_samples, value=2.0), fs=args.eda_fs)
    pipe._arousal_gate.update(_make_eda(int(args.eda_fs * 2), value=2.0), args.eda_fs)
    arousal_in_band = pipe._arousal_gate.is_in_band()
    print(f"      Calibrated. Arousal gate: {'IN-BAND (safe to stim)' if arousal_in_band else 'OUT-OF-BAND (blocked)'}")

    # ------------------------------------------------------------------
    # 3. Streaming simulation — print raw model output every second
    # ------------------------------------------------------------------
    ppg_chunk = int(args.ppg_fs)
    eda_chunk = int(args.eda_fs)
    total_events = 0

    print(f"\n[3/3] Streaming {args.duration}s of synthetic PPG+EDA through the pipeline...\n")
    print(f"  {'Time':>4s} | {'dia_p':>6s} | {'exh_p':>6s} | {'arousal':>7s} | {'latency':>8s} | Result")
    print(f"  {'-'*4} | {'-'*6} | {'-'*6} | {'-'*7} | {'-'*8} | ------")

    for sec in range(args.duration):
        ppg = _make_ppg(ppg_chunk, fs=args.ppg_fs)
        eda = _make_eda(eda_chunk, value=2.0)

        t0 = time.perf_counter()
        events = pipe.feed(ppg, eda)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        total_events += len(events)

        # Get raw model probabilities for display
        probs = _get_last_probs(pipe)
        dia_str = f"{probs[0]:.3f}" if probs else "  n/a"
        exh_str = f"{probs[1]:.3f}" if probs else "  n/a"
        arousal_str = "in-band" if pipe._arousal_gate.is_in_band() else "blocked"

        if events:
            # Print one summary row per second showing stim count + last event params
            ev = events[-1]
            print(
                f"  {sec:4d}s | {dia_str:>6s} | {exh_str:>6s} | {arousal_str:>7s} |"
                f" {elapsed_ms:6.1f}ms | ** STIM x{len(events)}"
                f"  {ev.amplitude:.2f}mA {ev.frequency:.0f}Hz {ev.pulse_width:.0f}us **"
            )
        else:
            # Which gate is blocking?
            if probs:
                dia_gate = probs[0] > pipe._dia_threshold
                exh_gate = probs[1] > pipe._exh_threshold
                blocked_by = []
                if not dia_gate:
                    blocked_by.append(f"dia({probs[0]:.3f}<{pipe._dia_threshold})")
                if not exh_gate:
                    blocked_by.append(f"exh({probs[1]:.3f}<{pipe._exh_threshold})")
                if not pipe._arousal_gate.is_in_band():
                    blocked_by.append("arousal")
                reason = ", ".join(blocked_by) if blocked_by else "N=3 consecutive gate"
            else:
                reason = "buffer filling"
            print(
                f"  {sec:4d}s | {dia_str:>6s} | {exh_str:>6s} | {arousal_str:>7s} |"
                f" {elapsed_ms:6.1f}ms | blocked: {reason}"
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    state = pipe.get_state()
    stride_samples = max(1, int(round(100 / 1000.0 * args.ppg_fs)))
    expected_fast = max(0, (state.total_samples_fed - int(2.0 * args.ppg_fs)) // stride_samples)

    print()
    print("=" * 70)
    print("Pipeline Summary")
    print("=" * 70)
    print(f"  PPG samples processed : {state.total_samples_fed:,}  ({args.duration}s x {args.ppg_fs:.0f}Hz)")
    print(f"  CNN inference calls   : {state.fast_path_calls:,}  (~{expected_fast} at 100ms stride, ~{elapsed_ms:.0f}ms each)")
    print(f"  HRV slow path calls   : {state.slow_path_calls}  (needs >{60}s of data)")
    print(f"  Stim events fired     : {total_events}")
    print(f"  Stim rate             : {total_events / args.duration:.2f}/sec")
    print()
    print("  Validation (WESAD 15 subjects, F22):")
    print("    Arousal classifier AUROC : 0.930  (target >0.80)")
    print("    Stim rate on real data   : 3.2/min  (after N=3 consecutive gate)")
    print("    EDA suppression example  : S7 stress: 588/min -> 44/min")
    print()
    if total_events == 0:
        print("  NOTE: No stim events with clinical gates (dia>0.65, N=3 frames).")
        print("  Run with --demo-mode to lower gates and see stim events fire.")
    else:
        print("  NOTE: --demo-mode uses F16 gates (dia>0.50, N=1). Clinical mode")
        print("  uses F17 (dia>0.65, N=3 frames) which reduces stim rate by 98%")
        print("  on real data (206/min -> 3.2/min, WESAD F22 replay).")


if __name__ == "__main__":
    main()
