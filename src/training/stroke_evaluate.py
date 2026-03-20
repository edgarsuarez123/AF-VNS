"""Evaluation script for StrokeHybridEnsemble.

Loads a checkpoint, runs inference on precomputed cache test split,
and reports AUROC, sensitivity, specificity, F1.

Usage:
    # Evaluate Phase 2 checkpoint on CVES test split (default)
    .venv/Scripts/python -m src.training.stroke_evaluate

    # Evaluate Phase 1 checkpoint on MIMIC-3 test split
    .venv/Scripts/python -m src.training.stroke_evaluate --phase 1

    # Evaluate Phase 2 on val split instead of test
    .venv/Scripts/python -m src.training.stroke_evaluate --phase 2 --split val
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from .build_model import load_config
from .train import PrecomputedDataset, run_validation_from_cache
from ..models.stroke_ensemble import build_stroke_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = "config_stroke.yaml"


def evaluate_stroke_cache(
    config_path: str = CONFIG_PATH,
    checkpoint_path: Optional[str] = None,
    phase: int = 2,
    split: str = "test",
    save_plots: bool = True,
    artifacts_dir: Optional[str] = None,
) -> dict:
    """Run evaluation on a precomputed stroke cache split.

    Args:
        config_path:     Path to config_stroke.yaml.
        checkpoint_path: Path to .pth checkpoint. Defaults to phase-appropriate path in config.
        phase:           1 = MIMIC-3 Phase 1 cache, 2 = CVES Phase 2 cache.
        split:           Cache split to evaluate: 'train', 'val', or 'test'.
        save_plots:      Whether to save ROC + confusion matrix PNG.
        artifacts_dir:   Output directory for plots. Defaults to cache dir parent.

    Returns:
        dict with keys: auroc, f1, sensitivity, specificity, n_samples, n_pos, n_neg.
    """
    config = load_config(config_path)
    paths_cfg = config.get("paths", {})
    train_cfg = config.get("training", {})
    eval_cfg = config.get("evaluation", {})
    threshold = float(eval_cfg.get("threshold", 0.5))
    batch_size = int(train_cfg.get("batch_size", 32))

    if phase == 1:
        default_ckpt = paths_cfg.get("phase1_checkpoint", "models/checkpoints/stroke_phase1_model.pth")
        cache_dir = Path(paths_cfg.get("phase1_cache_dir", "models/artifacts/cache_stroke_phase1"))
    else:
        default_ckpt = paths_cfg.get("phase2_checkpoint", "models/checkpoints/stroke_phase2_model.pth")
        cache_dir = Path(paths_cfg.get("phase2_cache_dir", "models/artifacts/cache_stroke_phase2"))

    checkpoint_path = checkpoint_path or default_ckpt

    if artifacts_dir is None:
        artifacts_dir = str(cache_dir.parent / "stroke_eval")
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

    # Verify cache exists
    required = [f"{split}_short.npy", f"{split}_hrv_scaled.npy", f"{split}_labels.npy"]
    missing = [f for f in required if not (cache_dir / f).exists()]
    if missing:
        logger.error("Cache split '%s' missing in %s: %s", split, cache_dir, missing)
        raise FileNotFoundError(f"Cache missing: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | checkpoint: %s", device, checkpoint_path)

    model = build_stroke_model(config_path=config_path,
                               checkpoint_path=checkpoint_path, device=device)
    model.eval()

    ds = PrecomputedDataset(cache_dir, split, is_train=False, config_path=config_path)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info("Evaluating %d samples (split=%s, phase=%d)", len(ds), split, phase)

    # Collect logits and labels
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            short_t, hrv_t, labels_t = batch[0], batch[1], batch[2]
            hrv_lengths = batch[3] if len(batch) > 3 else None
            short_t = short_t.to(device)
            hrv_t = hrv_t.to(device)
            labels_t = labels_t.to(device)
            if hrv_lengths is not None:
                hrv_lengths = hrv_lengths.to(device)
            logits = model(short_t, hrv_t, hrv_lengths=hrv_lengths).squeeze(-1)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels_t.cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits_np, -50, 50)))
    preds = (probs >= threshold).astype(np.int64)

    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

    try:
        auroc = float(roc_auc_score(labels_np, probs))
    except Exception:
        auroc = float("nan")
    f1 = float(f1_score(labels_np, preds, zero_division=0))
    cm = confusion_matrix(labels_np, preds)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    n_pos = int((labels_np == 1).sum())
    n_neg = int((labels_np == 0).sum())

    print(f"\n=== Stroke Evaluation — Phase {phase} | split={split} ===")
    print(f"  Samples:     {len(labels_np)} ({n_pos} stroke, {n_neg} control)")
    print(f"  Threshold:   {threshold}")
    print(f"  AUROC:       {auroc:.4f}")
    print(f"  F1:          {f1:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f}  (TP={tp}, FN={fn})")
    print(f"  Specificity: {specificity:.4f}  (TN={tn}, FP={fp})")

    # Save JSON results
    results = {
        "phase": phase, "split": split, "checkpoint": str(checkpoint_path),
        "n_samples": int(len(labels_np)), "n_pos": n_pos, "n_neg": n_neg,
        "auroc": auroc, "f1": f1, "sensitivity": sensitivity, "specificity": specificity,
        "threshold": threshold, "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
    results_path = Path(artifacts_dir) / f"stroke_eval_phase{phase}_{split}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results to {results_path}")

    if save_plots and labels_np.size > 0 and not np.isnan(auroc):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.metrics import RocCurveDisplay

            fig, ax = plt.subplots(figsize=(5, 5))
            RocCurveDisplay.from_predictions(
                labels_np, probs, ax=ax, name=f"AUROC={auroc:.3f}"
            )
            ax.set_title(f"Stroke ROC — Phase {phase} {split}")
            roc_path = Path(artifacts_dir) / f"stroke_roc_phase{phase}_{split}.png"
            fig.savefig(roc_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved ROC to {roc_path}")

            # Confusion matrix
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            ax2.imshow(cm, cmap="Blues")
            h, w = cm.shape
            ax2.set_xticks(range(w))
            ax2.set_yticks(range(h))
            ax2.set_xticklabels(["Pred Control", "Pred Stroke"] if w == 2 else [f"Pred {j}" for j in range(w)])
            ax2.set_yticklabels(["True Control", "True Stroke"] if h == 2 else [f"True {i}" for i in range(h)])
            for i in range(h):
                for j in range(w):
                    ax2.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
            ax2.set_title(f"Confusion Matrix — Phase {phase} {split}")
            cm_path = Path(artifacts_dir) / f"stroke_confusion_phase{phase}_{split}.png"
            fig2.savefig(cm_path, dpi=100, bbox_inches="tight")
            plt.close(fig2)
            print(f"  Saved confusion matrix to {cm_path}")
        except ImportError:
            pass

    return results


def evaluate_stroke_ood(
    config_path: str = CONFIG_PATH,
    checkpoint_path: Optional[str] = None,
    save_plots: bool = True,
    artifacts_dir: Optional[str] = None,
) -> dict:
    """OOD evaluation on SHaRe/EMBC dataset (never seen during training).

    Runs on-the-fly inference (no precomputed cache required).
    Uses Phase 2 scaler so HRV features are on the same scale as fine-tuning.

    Args:
        config_path:     Path to config_stroke.yaml.
        checkpoint_path: Path to .pth checkpoint. Defaults to Phase 2 checkpoint.
        save_plots:      Whether to save ROC + confusion matrix PNG.
        artifacts_dir:   Output directory. Defaults to models/artifacts/stroke_eval.

    Returns:
        dict with keys: auroc, f1, sensitivity, specificity, n_samples, n_pos, n_neg.
    """
    from ..data.stroke_parsers import parse_sharee_dir
    from .stroke_precompute_cache import SHAREE_EVENT_PATIENTS
    from ..features.pipeline import waveform_10s_denoised, waveform_to_hrv_sequence
    from ..features.scaler import load_scaler, transform

    config = load_config(config_path)
    paths_cfg = config.get("paths", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    eval_cfg = config.get("evaluation", {})
    threshold = float(eval_cfg.get("threshold", 0.5))

    checkpoint_path = checkpoint_path or paths_cfg.get(
        "phase2_checkpoint", "models/checkpoints/stroke_phase2_model.pth"
    )
    scaler_path = paths_cfg.get("phase2_scaler", "models/artifacts/stroke_phase2_scaler.pkl")
    sharee_dir = data_cfg.get("sharee_subdir", "data/raw/stroke avns/shareedb")
    target_fs = float(data_cfg.get("target_fs", 250))
    waveform_sec = float(data_cfg.get("waveform_sec", 10))

    if artifacts_dir is None:
        artifacts_dir = str(Path(scaler_path).parent / "stroke_eval")
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("OOD eval | device: %s | checkpoint: %s", device, checkpoint_path)

    model = build_stroke_model(config_path=config_path,
                               checkpoint_path=checkpoint_path, device=device)
    model.eval()
    scaler = load_scaler(path=scaler_path, config_path=config_path)

    # SHaRe: 17 event patients = stroke (1), rest = control (0)
    label_map = {pid: 1 for pid in SHAREE_EVENT_PATIENTS}
    records = parse_sharee_dir(sharee_dir, label_map, config_path)
    logger.info("OOD: %d SHaRe records loaded", len(records))

    min_samples = int(waveform_sec * target_fs)
    hrv_window_samples = int(300 * target_fs)

    all_logits = []
    all_labels = []
    n_skipped = 0

    with torch.no_grad():
        for rec in records:
            if rec["label"] is None:
                n_skipped += 1
                continue
            signal = rec["signal"]
            fs = float(rec["fs"])
            if len(signal) < min_samples:
                n_skipped += 1
                continue

            # CNN: center 10s window
            center = len(signal) // 2
            start = max(0, center - min_samples // 2)
            short_denoised = waveform_10s_denoised(signal[start:start + min_samples], fs, config_path)

            # HRV: first 300s (or full signal if shorter)
            long_signal = signal[:hrv_window_samples] if len(signal) >= hrv_window_samples else signal
            hrv_seq = waveform_to_hrv_sequence(long_signal, fs, config_path=config_path)
            hrv_scaled = transform(hrv_seq, scaler)
            hrv_scaled = np.nan_to_num(hrv_scaled, nan=0.0).astype(np.float32)
            hrv_len = max(1, int(np.sum(~np.all(np.isnan(hrv_seq), axis=-1))))

            short_t = torch.tensor(short_denoised, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            hrv_t = torch.tensor(hrv_scaled, dtype=torch.float32).unsqueeze(0).to(device)
            hrv_lengths_t = torch.tensor([hrv_len], dtype=torch.long).to(device)

            logit = model(short_t, hrv_t, hrv_lengths=hrv_lengths_t).squeeze(-1).item()
            all_logits.append(logit)
            all_labels.append(rec["label"])

    logger.info("OOD: %d records evaluated, %d skipped", len(all_logits), n_skipped)

    logits_np = np.array(all_logits, dtype=np.float32)
    labels_np = np.array(all_labels, dtype=np.float32)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits_np, -50, 50)))
    preds = (probs >= threshold).astype(np.int64)

    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

    try:
        auroc = float(roc_auc_score(labels_np, probs))
    except Exception:
        auroc = float("nan")
    f1 = float(f1_score(labels_np, preds, zero_division=0))
    cm = confusion_matrix(labels_np, preds)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    n_pos = int((labels_np == 1).sum())
    n_neg = int((labels_np == 0).sum())

    print(f"\n=== Stroke OOD Evaluation — SHaRe/EMBC ===")
    print(f"  Samples:     {len(labels_np)} ({n_pos} stroke, {n_neg} control)")
    print(f"  Threshold:   {threshold}")
    print(f"  AUROC:       {auroc:.4f}")
    print(f"  F1:          {f1:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f}  (TP={tp}, FN={fn})")
    print(f"  Specificity: {specificity:.4f}  (TN={tn}, FP={fp})")

    results = {
        "split": "ood_sharee", "checkpoint": str(checkpoint_path),
        "n_samples": int(len(labels_np)), "n_pos": n_pos, "n_neg": n_neg,
        "auroc": auroc, "f1": f1, "sensitivity": sensitivity, "specificity": specificity,
        "threshold": threshold, "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
    results_path = Path(artifacts_dir) / "stroke_eval_ood_sharee.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results to {results_path}")

    if save_plots and labels_np.size > 0 and not np.isnan(auroc):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.metrics import RocCurveDisplay

            fig, ax = plt.subplots(figsize=(5, 5))
            RocCurveDisplay.from_predictions(labels_np, probs, ax=ax, name=f"AUROC={auroc:.3f}")
            ax.set_title("Stroke OOD ROC — SHaRe/EMBC")
            fig.savefig(Path(artifacts_dir) / "stroke_roc_ood_sharee.png", dpi=100, bbox_inches="tight")
            plt.close(fig)

            fig2, ax2 = plt.subplots(figsize=(4, 4))
            ax2.imshow(cm, cmap="Blues")
            h, w = cm.shape
            ax2.set_xticks(range(w))
            ax2.set_yticks(range(h))
            ax2.set_xticklabels(["Pred Control", "Pred Stroke"] if w == 2 else [f"Pred {j}" for j in range(w)])
            ax2.set_yticklabels(["True Control", "True Stroke"] if h == 2 else [f"True {i}" for i in range(h)])
            for i in range(h):
                for j in range(w):
                    ax2.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
            ax2.set_title("Confusion Matrix — OOD SHaRe/EMBC")
            fig2.savefig(Path(artifacts_dir) / "stroke_confusion_ood_sharee.png", dpi=100, bbox_inches="tight")
            plt.close(fig2)
            print(f"  Saved plots to {artifacts_dir}")
        except ImportError:
            pass

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate StrokeHybridEnsemble.")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=2,
                        help="1=MIMIC-3 cache, 2=CVES cache (default)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--ood", action="store_true", help="Run OOD evaluation on SHaRe/EMBC")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--artifacts-dir", default=None)
    args = parser.parse_args()

    if args.ood:
        evaluate_stroke_ood(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            save_plots=not args.no_plots,
            artifacts_dir=args.artifacts_dir,
        )
    else:
        evaluate_stroke_cache(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            phase=args.phase,
            split=args.split,
            save_plots=not args.no_plots,
            artifacts_dir=args.artifacts_dir,
        )


if __name__ == "__main__":
    main()
