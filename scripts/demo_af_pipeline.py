"""Demo: AF vs NSR classification with HybridEnsemble.

Loads the trained model from checkpoint and runs inference on synthetic
ECG + HRV data, printing the AF probability and predicted class.

Usage:
    .venv/Scripts/python scripts/demo_af_pipeline.py
    .venv/Scripts/python scripts/demo_af_pipeline.py --checkpoint models/checkpoints/best_model.pth
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.build_model import build_model, build_ensemble_config, load_config


# ---------------------------------------------------------------------------
# Synthetic signal generators
# ---------------------------------------------------------------------------

def _make_ecg_waveform(fs: float = 250.0, duration_sec: float = 10.0) -> torch.Tensor:
    """Synthetic ECG: 1.2 Hz cardiac fundamental + 12 Hz harmonic."""
    n = int(fs * duration_sec)
    t = np.arange(n) / fs
    wave = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 12.0 * t)
    # Shape: (1, 1, T) — batch=1, channels=1, time
    return torch.tensor(wave, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def _make_hrv_sequence(seq_len: int = 5, n_features: int = 7) -> torch.Tensor:
    """Synthetic HRV feature sequence (randomized, for architecture demo only)."""
    # Shape: (1, seq_len, n_features)
    return torch.randn(1, seq_len, n_features)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AF vs NSR classification demo")
    p.add_argument("--config", default="config.yaml", help="Config YAML path (default: config.yaml)")
    p.add_argument(
        "--checkpoint",
        default="models/checkpoints/best_model.pth",
        help="Model checkpoint path (default: models/checkpoints/best_model.pth)",
    )
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
    print("AF vs NSR Classification - HybridEnsemble Demo")
    print("=" * 65)
    print(f"  Config     : {args.config}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Device     : {args.device}")

    # ------------------------------------------------------------------
    # 1. Load config and print architecture summary
    # ------------------------------------------------------------------
    cfg = load_config(args.config)
    m = cfg.get("model", {})
    seq_len = int(m.get("hrv_seq_len", 5))
    n_features = int(m.get("hrv_n_features", 7))
    fs = float(cfg.get("target_fs", 250.0))

    print(f"\n[1/3] Model architecture:")
    print(f"      CNN branch : (1, 1, {int(fs * 10)}) -> {m.get('cnn_embed_dim', 128)}-d embedding")
    print(f"      GRU branch : ({seq_len}, {n_features}) HRV seq -> {m.get('rnn_hidden_size', 64)}-d hidden")
    print(f"      Transformer: ({seq_len}, {n_features}) HRV seq -> {m.get('transformer_d_model', 64)}-d encoding")
    fused = int(m.get('cnn_embed_dim', 128)) + int(m.get('rnn_hidden_size', 64)) + int(m.get('transformer_d_model', 64))
    print(f"      Fusion     : concat {fused}-d -> Linear -> 1 logit (BCEWithLogitsLoss)")

    # ------------------------------------------------------------------
    # 2. Build model from checkpoint
    # ------------------------------------------------------------------
    print(f"\n[2/3] Loading model from {args.checkpoint}...")
    ckpt_path = args.checkpoint if Path(args.checkpoint).is_file() else None
    if ckpt_path is None:
        print(f"      WARNING: checkpoint not found — using random weights for architecture demo")
    t0 = time.perf_counter()
    try:
        model = build_model(config_path=args.config, checkpoint_path=ckpt_path, device=args.device)
        loaded_msg = f"checkpoint loaded"
    except (RuntimeError, Exception) as exc:
        # Checkpoint head architecture may differ from current config — fall back to random weights
        print(f"      WARNING: checkpoint mismatch — head architecture changed since training.")
        print(f"      Falling back to random weights for architecture demo.")
        model = build_model(config_path=args.config, checkpoint_path=None, device=args.device)
        loaded_msg = "random weights (checkpoint mismatch)"
    model.eval()
    load_ms = (time.perf_counter() - t0) * 1000
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Loaded in {load_ms:.0f}ms | Parameters: {n_params:,} | {loaded_msg}")

    # ------------------------------------------------------------------
    # 3. Run inference on synthetic data
    # ------------------------------------------------------------------
    print(f"\n[3/3] Running inference on synthetic ECG + HRV...")
    print(f"      ECG input : (1, 1, {int(fs * 10)}) — 10s at {fs:.0f} Hz (sine wave)")
    print(f"      HRV input : (1, {seq_len}, {n_features}) — {seq_len}x60s windows, {n_features} features (random)")

    waveform = _make_ecg_waveform(fs=fs, duration_sec=10.0).to(args.device)
    hrv_seq = _make_hrv_sequence(seq_len=seq_len, n_features=n_features).to(args.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logit = model(waveform, hrv_seq)
    infer_ms = (time.perf_counter() - t0) * 1000

    prob_af = torch.sigmoid(logit).item()
    predicted = "AF" if prob_af >= 0.5 else "NSR"

    print(f"\n  AF probability : {prob_af:.4f}")
    print(f"  Prediction     : {predicted}  (threshold 0.5)")
    print(f"  Inference time : {infer_ms:.1f}ms")

    print("\n" + "=" * 65)
    print("NOTE: HRV input is random — prediction is not clinically meaningful.")
    print("  On real AFDB/NSRDB data: AUROC target >= 0.75 (NFR-2.1).")
    print("  Training: .venv/Scripts/python -m src.training.train --use-cache")
    print("=" * 65)


if __name__ == "__main__":
    main()
