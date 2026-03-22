"""
S-30 — Closed-loop event-to-stimulation latency benchmark for PhaseDetector.

Measures the time from a physiological co-occurrence event (diastole AND
exhalation simultaneously) to the stimulation command being issued.

NFR-1.1 requirement: <200ms total closed-loop system latency.

Closed-loop pipeline:
  ECG stream arrives at 250Hz
  ↓ System maintains a 2s sliding window, advanced every inference_stride_ms
  ↓ Each stride: resample + denoise + model forward + co-occurrence check
  ↓ If diastole[last_frame]=1 AND exhalation[last_frame]=1 → trigger stim

Event-to-stim latency = inference_stride_ms + per_call_processing_ms
  - Worst case: event occurs 1ms after last inference → wait stride_ms → process
  - Best case: event occurs just before inference → detected in ~processing_ms

At 100ms stride (recommended): worst-case ~113ms, well within 200ms NFR.
At 200ms stride (default training): worst-case ~213ms, exceeds NFR.

Usage:
  python -m src.training.phase_latency_benchmark --device cpu
  python -m src.training.phase_latency_benchmark --device cpu --inference-stride-ms 100
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.phase_detector import build_phase_detector
from src.features.wavelet_filter import denoise
from scipy.signal import resample as scipy_resample


CONFIG_PATH = "config_stroke.yaml"


def _make_ecg_stream(duration_sec: float = 10.0, source_fs: int = 500) -> np.ndarray:
    """Synthetic continuous ECG stream at source_fs."""
    n = int(source_fs * duration_sec)
    t = np.arange(n) / source_fs
    return (np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 12 * t)).astype(np.float64)


def _process_one_stride(
    ecg_window: np.ndarray,
    source_fs: int,
    target_fs: int,
    target_samples: int,
    model: torch.nn.Module,
    device: torch.device,
    config_path: str,
) -> tuple:
    """Run one inference stride. Returns (preds, processing_ms)."""
    t0 = time.perf_counter()

    # Resample to target fs
    sig = scipy_resample(ecg_window, target_samples)

    # Wavelet denoise
    sig = denoise(sig, float(target_fs), config_path=config_path)

    # Tensor + device transfer
    tensor = torch.tensor(sig, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Model forward
    with torch.no_grad():
        logits = model(tensor)

    # Postprocess: sigmoid + threshold
    preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()[0]  # (10, 2)

    # Co-occurrence check: last frame diastole AND exhalation
    last_frame = preds[-1]  # [diastole, exhalation]
    stim_trigger = bool(last_frame[0]) and bool(last_frame[1])

    t1 = time.perf_counter()
    return stim_trigger, (t1 - t0) * 1000.0


def run_benchmark(
    config_path: str = CONFIG_PATH,
    checkpoint: str = None,
    device: str = "cpu",
    n_iterations: int = 200,
    inference_stride_ms: int = 100,
    source_fs: int = 500,
    target_fs: int = 250,
    window_sec: float = 2.0,
) -> dict:
    """Simulate closed-loop streaming inference and measure event-to-stim latency."""
    dev = torch.device(device)
    model = build_phase_detector(
        config_path=config_path,
        checkpoint_path=checkpoint,
        device=device,
    )
    model.eval()

    target_samples = int(target_fs * window_sec)
    stride_samples_src = int(source_fs * inference_stride_ms / 1000.0)

    # Generate a continuous ECG stream
    stream_sec = window_sec + (n_iterations * inference_stride_ms / 1000.0) + 1.0
    ecg_stream = _make_ecg_stream(duration_sec=stream_sec, source_fs=source_fs)
    stream_window_samples = int(source_fs * window_sec)

    # Warmup
    for i in range(5):
        start = i * stride_samples_src
        window = ecg_stream[start:start + stream_window_samples]
        _process_one_stride(window, source_fs, target_fs, target_samples, model, dev, config_path)

    # Timed iterations — simulate real-time stride
    processing_times = []
    event_to_stim_times = []

    for i in range(n_iterations):
        start = i * stride_samples_src
        window = ecg_stream[start:start + stream_window_samples]
        if len(window) < stream_window_samples:
            break

        # Simulate: event occurs at a random time within the stride period
        # Worst case = event occurs 1ms after previous inference (full stride wait)
        # We measure worst-case = stride_ms + processing_ms
        _, proc_ms = _process_one_stride(
            window, source_fs, target_fs, target_samples, model, dev, config_path
        )
        processing_times.append(proc_ms)
        # Worst-case event-to-stim = full stride wait + processing time
        event_to_stim_times.append(inference_stride_ms + proc_ms)

    def _stats(arr):
        a = np.array(arr)
        return {
            "mean_ms": float(np.mean(a)),
            "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)),
            "max_ms": float(np.max(a)),
        }

    return {
        "device": device,
        "n_iterations": len(processing_times),
        "params": model.param_count(),
        "inference_stride_ms": inference_stride_ms,
        "nfr_limit_ms": 200,
        "per_call_processing": _stats(processing_times),
        "event_to_stim_worst_case": _stats(event_to_stim_times),
    }


def _print_results(results: dict) -> None:
    nfr = results["nfr_limit_ms"]
    stride = results["inference_stride_ms"]
    proc_p95 = results["per_call_processing"]["p95_ms"]
    e2s_p95 = results["event_to_stim_worst_case"]["p95_ms"]
    status = "PASS" if e2s_p95 < nfr else "FAIL"

    print(f"\n{'='*62}")
    print(f"  PhaseDetector Closed-Loop Latency — {results['device'].upper()}")
    print(f"  Model params: {results['params']:,}  |  Inference stride: {stride}ms")
    print(f"{'='*62}")
    print(f"\n  Per-call processing time (resample + denoise + inference):")
    s = results["per_call_processing"]
    print(f"    mean={s['mean_ms']:.1f}ms  p50={s['p50_ms']:.1f}ms  "
          f"p95={s['p95_ms']:.1f}ms  max={s['max_ms']:.1f}ms")
    print(f"\n  Event-to-stim worst-case latency (stride + processing):")
    s = results["event_to_stim_worst_case"]
    print(f"    = {stride}ms stride + {proc_p95:.1f}ms processing (p95)")
    print(f"    worst-case p95 = {e2s_p95:.1f}ms")
    print(f"\n  NFR-1.1 target: <{nfr}ms  ->  [{status}]")
    if status == "FAIL":
        rec_stride = max(10, int(nfr * 0.9 - proc_p95))
        print(f"  Recommendation: reduce inference stride to <={rec_stride}ms")
    print(f"{'='*62}\n")


def main():
    parser = argparse.ArgumentParser(
        description="PhaseDetector closed-loop event-to-stim latency benchmark."
    )
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--n-iterations", type=int, default=200)
    parser.add_argument("--inference-stride-ms", type=int, default=100,
                        help="How often inference runs in deployment (ms). "
                             "Worst-case latency = stride + processing. Default 100ms.")
    parser.add_argument("--source-fs", type=int, default=500,
                        help="Simulated incoming ECG sampling rate (default 500Hz like CVES)")
    args = parser.parse_args()

    results = run_benchmark(
        config_path=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        n_iterations=args.n_iterations,
        inference_stride_ms=args.inference_stride_ms,
        source_fs=args.source_fs,
    )
    _print_results(results)


if __name__ == "__main__":
    main()
