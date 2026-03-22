"""
S-30 — End-to-end latency benchmark for PhaseDetector inference.

Measures wall-clock time for a single 2s ECG window from raw input to
phase predictions. Target: <200ms (NFR-1.1/NFR-2.1).

Pipeline timed:
  1. Resample (scipy.signal.resample) — simulates incoming data at non-target fs
  2. Wavelet denoise (CWT) — baseline wander + HF noise removal
  3. Tensor creation + device transfer
  4. Model forward pass (torch.no_grad)
  5. Sigmoid + threshold postprocessing

Usage:
  python -m src.training.phase_latency_benchmark --device cpu --n-iterations 100
  python -m src.training.phase_latency_benchmark --device cuda --n-iterations 100
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
from scipy.signal import resample


CONFIG_PATH = "config_stroke.yaml"


def _make_ecg_chunk(window_sec: float = 2.0, source_fs: int = 500) -> np.ndarray:
    """Synthetic ECG chunk at source_fs — simulates incoming raw signal."""
    n = int(source_fs * window_sec)
    t = np.arange(n) / source_fs
    # Simple synthetic ECG-like signal with heartbeat frequency
    ecg = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 12 * t)
    return ecg.astype(np.float64)


def run_benchmark(
    config_path: str = CONFIG_PATH,
    checkpoint: str = None,
    device: str = "cpu",
    n_iterations: int = 100,
    source_fs: int = 500,
    target_fs: int = 250,
    window_sec: float = 2.0,
) -> dict:
    """Run end-to-end latency benchmark. Returns timing stats in ms."""
    dev = torch.device(device)

    # Load model
    model = build_phase_detector(
        config_path=config_path,
        checkpoint_path=checkpoint,
        device=device,
    )
    model.eval()

    target_samples = int(target_fs * window_sec)
    source_samples = int(source_fs * window_sec)
    ecg_raw = _make_ecg_chunk(window_sec=window_sec, source_fs=source_fs)

    # Warmup
    for _ in range(5):
        sig = resample(ecg_raw, target_samples)
        sig = denoise(sig, float(target_fs), config_path=config_path)
        t = torch.tensor(sig, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(dev)
        with torch.no_grad():
            logits = model(t)
        _ = torch.sigmoid(logits) > 0.5

    # Timed iterations
    times_total = []
    times_resample = []
    times_denoise = []
    times_model = []

    for _ in range(n_iterations):
        ecg_in = _make_ecg_chunk(window_sec=window_sec, source_fs=source_fs)

        t0 = time.perf_counter()

        # Step 1: Resample
        t_a = time.perf_counter()
        sig = resample(ecg_in, target_samples)
        t_b = time.perf_counter()

        # Step 2: Wavelet denoise
        sig = denoise(sig, float(target_fs), config_path=config_path)
        t_c = time.perf_counter()

        # Step 3+4: Tensor + model forward
        tensor = torch.tensor(sig, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(dev)
        with torch.no_grad():
            logits = model(tensor)
        t_d = time.perf_counter()

        # Step 5: Postprocess
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        t1 = time.perf_counter()

        times_total.append((t1 - t0) * 1000)
        times_resample.append((t_b - t_a) * 1000)
        times_denoise.append((t_c - t_b) * 1000)
        times_model.append((t_d - t_c) * 1000)

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
        "n_iterations": n_iterations,
        "params": model.param_count(),
        "total": _stats(times_total),
        "resample": _stats(times_resample),
        "denoise": _stats(times_denoise),
        "model_forward": _stats(times_model),
    }


def _print_results(results: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  PhaseDetector Latency Benchmark — {results['device'].upper()}")
    print(f"  Model params: {results['params']:,}")
    print(f"  Iterations: {results['n_iterations']}")
    print(f"{'='*60}")

    nfr_limit = 200.0
    for stage, label in [
        ("resample", "Resample (scipy)"),
        ("denoise", "Wavelet denoise (CWT)"),
        ("model_forward", "Model forward pass"),
        ("total", "TOTAL end-to-end"),
    ]:
        s = results[stage]
        flag = " [PASS]" if stage == "total" and s["p95_ms"] < nfr_limit else (
               " [FAIL - EXCEEDS NFR-1.1]" if stage == "total" and s["p95_ms"] >= nfr_limit else ""
        )
        print(f"\n  {label}:{flag}")
        print(f"    mean={s['mean_ms']:.1f}ms  p50={s['p50_ms']:.1f}ms  "
              f"p95={s['p95_ms']:.1f}ms  p99={s['p99_ms']:.1f}ms  max={s['max_ms']:.1f}ms")

    total_p95 = results["total"]["p95_ms"]
    print(f"\n  NFR-1.1 target: <{nfr_limit}ms")
    status = "PASS" if total_p95 < nfr_limit else "FAIL"
    print(f"  p95 total: {total_p95:.1f}ms — {status}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="PhaseDetector end-to-end latency benchmark.")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path (default: best model from config)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Device to benchmark on")
    parser.add_argument("--n-iterations", type=int, default=100)
    parser.add_argument("--source-fs", type=int, default=500,
                        help="Simulated incoming signal sampling rate (default 500 Hz like CVES)")
    args = parser.parse_args()

    results = run_benchmark(
        config_path=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        n_iterations=args.n_iterations,
        source_fs=args.source_fs,
    )
    _print_results(results)


if __name__ == "__main__":
    main()
