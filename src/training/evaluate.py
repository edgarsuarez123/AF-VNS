"""
Evaluation script: load best checkpoint, run on a split, report AUROC/F1/Sensitivity/Specificity.
Save ROC curve and optional confusion matrix to models/artifacts/.

Supports --mimic3 for cross-dataset evaluation on MIMIC-III holdout (NFR-3.1).
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from src.data.dataloaders import get_dataloaders
from src.data.dataset_parsers import parse_mimic3_dir, _resample_to_target_fs
from src.features.pipeline import waveform_10s_denoised, waveform_to_hrv_sequence
from src.features.scaler import load_scaler, transform

from .build_model import build_model, load_config
from .train import _batch_to_device_and_model, _pad_collate

logger = logging.getLogger(__name__)
CONFIG_PATH = "config.yaml"


def _get_loader_for_split(split: str, config_path: str, batch_size: int):
    """Return DataLoader for 'train', 'val', or 'test' with _pad_collate."""
    train_loader, val_loader, test_loader = get_dataloaders(
        config_path=config_path,
        batch_size=batch_size,
        create_split_if_missing=False,
    )
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = loaders[split]
    return torch.utils.data.DataLoader(
        loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_pad_collate,
    )


def run_evaluation(
    config_path: str = CONFIG_PATH,
    checkpoint_path: Optional[str] = None,
    split: str = "test",
    save_plots: bool = True,
    artifacts_dir: Optional[str] = None,
):
    config = load_config(config_path)
    paths_cfg = config.get("paths", {})
    train_cfg = config.get("training", {})

    checkpoint_path = checkpoint_path or paths_cfg.get("checkpoint", "models/checkpoints/best_model.pth")
    scaler_path = paths_cfg.get("scaler", "models/artifacts/scaler.pkl")
    batch_size = int(train_cfg.get("batch_size", 32))

    if artifacts_dir is None:
        artifacts_dir = str(Path(scaler_path).parent)
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config_path=config_path, checkpoint_path=checkpoint_path, device=device)
    model.eval()
    scaler = load_scaler(path=scaler_path, config_path=config_path)

    loader = _get_loader_for_split(split, config_path, batch_size)
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            short_pad, long_pad, labels, fs_list = batch
            short_t, hrv_t, labels_t = _batch_to_device_and_model(
                short_pad, long_pad, labels, fs_list, scaler, device, config_path
            )
            logits = model(short_t, hrv_t).squeeze(-1)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels_t.cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits_np, -50, 50)))
    preds = (probs >= 0.5).astype(np.int64)

    # Metrics
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )

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

    print(f"Split: {split}")
    print(f"  AUROC:       {auroc:.4f}")
    print(f"  F1:          {f1:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")

    if save_plots and labels_np.size > 0:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.metrics import RocCurveDisplay

            # ROC curve
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            RocCurveDisplay.from_predictions(labels_np, probs, ax=ax, name=f"AUROC={auroc:.3f}")
            ax.set_title(f"ROC curve ({split})")
            roc_path = Path(artifacts_dir) / f"roc_{split}.png"
            fig.savefig(roc_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved ROC to {roc_path}")

            # Confusion matrix (may be 1x1 or 2x2)
            if cm.size >= 1:
                fig2, ax2 = plt.subplots(1, 1, figsize=(4, 4))
                ax2.imshow(cm, cmap="Blues")
                h, w = cm.shape
                ax2.set_xticks(range(w))
                ax2.set_yticks(range(h))
                ax2.set_xticklabels([f"Pred {j}" for j in range(w)])
                ax2.set_yticklabels([f"True {i}" for i in range(h)])
                for i in range(h):
                    for j in range(w):
                        ax2.text(j, i, str(cm[i, j]), ha="center", va="center")
                ax2.set_title(f"Confusion matrix ({split})")
                fig2.savefig(Path(artifacts_dir) / f"confusion_{split}.png", dpi=100, bbox_inches="tight")
                plt.close(fig2)
                print(f"  Saved confusion matrix to {Path(artifacts_dir) / f'confusion_{split}.png'}")
        except ImportError:
            pass

    return {"auroc": auroc, "f1": f1, "sensitivity": sensitivity, "specificity": specificity}


def evaluate_mimic3(
    config_path: str = CONFIG_PATH,
    checkpoint_path: Optional[str] = None,
    save_plots: bool = True,
    artifacts_dir: Optional[str] = None,
    holdout_path: Optional[str] = None,
) -> dict:
    """Cross-dataset evaluation on MIMIC-III holdout (NFR-3.1).

    Processes records individually to handle variable lengths (30s-900s)
    without requiring PhysioDataset or split.json.

    If holdout_path is set, evaluates only on the reserved holdout subjects
    (records that were NOT used in mixed-source training).
    """
    config = load_config(config_path)
    paths_cfg = config.get("paths", {})
    data_cfg = config.get("data", {})

    checkpoint_path = checkpoint_path or paths_cfg.get("checkpoint", "models/checkpoints/best_model.pth")
    scaler_path = paths_cfg.get("scaler", "models/artifacts/scaler.pkl")
    mimic3_dir = Path(data_cfg.get("mimic3_subdir", "data/raw/mimic3"))
    target_fs = float(data_cfg.get("target_fs", 250))
    waveform_sec = float(data_cfg.get("waveform_sec", 10))

    # If holdout_path provided, restrict to those subject IDs only
    holdout_ids: Optional[set] = None
    if holdout_path and Path(holdout_path).exists():
        with open(holdout_path) as f:
            import json as _json
            _hd = _json.load(f)
        holdout_ids = set(_hd.get("all", []))
        logger.info("Holdout mode: evaluating %d subjects from %s", len(holdout_ids), holdout_path)

    if artifacts_dir is None:
        artifacts_dir = str(Path(scaler_path).parent / "mimic3_eval")
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config_path=config_path, checkpoint_path=checkpoint_path, device=device)
    model.eval()
    scaler = load_scaler(path=scaler_path, config_path=config_path)

    # Parse MIMIC-III records
    records = parse_mimic3_dir(mimic3_dir)
    logger.info("Parsed %d MIMIC-III records", len(records))

    # Resample to target_fs
    if target_fs > 0:
        for rec in records:
            if abs(rec["fs"] - target_fs) >= 0.1:
                rec["signal"] = _resample_to_target_fs(rec["signal"], rec["fs"], target_fs)
                rec["fs"] = target_fs

    min_samples = int(waveform_sec * target_fs)
    hrv_window_samples = int(300 * target_fs)

    all_logits = []
    all_labels = []
    per_record = []
    n_skipped = 0
    n_short = 0

    with torch.no_grad():
        for rec in records:
            if holdout_ids is not None and rec["subject_id"] not in holdout_ids:
                continue
            if rec["label"] is None:
                n_skipped += 1
                continue
            signal = rec["signal"]
            fs = rec["fs"]

            if len(signal) < min_samples:
                logger.warning("Skipping %s: too short (%d samples, need %d)",
                               rec["subject_id"], len(signal), min_samples)
                n_skipped += 1
                continue

            is_short = len(signal) < hrv_window_samples
            if is_short:
                n_short += 1

            # CNN input: center 10s
            center = len(signal) // 2
            start = max(0, center - min_samples // 2)
            short_signal = signal[start : start + min_samples]
            short_denoised = waveform_10s_denoised(short_signal, fs, config_path)

            # HRV input: first min(len, 300s)
            long_signal = signal[:hrv_window_samples] if len(signal) >= hrv_window_samples else signal
            hrv_seq = waveform_to_hrv_sequence(long_signal, fs, config_path=config_path)
            hrv_scaled = transform(hrv_seq, scaler)
            hrv_scaled = np.nan_to_num(hrv_scaled, nan=0.0).astype(np.float32)

            # Build tensors
            short_t = torch.tensor(short_denoised, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            hrv_t = torch.tensor(hrv_scaled, dtype=torch.float32).unsqueeze(0).to(device)

            logit = model(short_t, hrv_t).squeeze(-1).item()
            prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))

            all_logits.append(logit)
            all_labels.append(rec["label"])
            per_record.append({
                "subject_id": rec["subject_id"],
                "label": rec["label"],
                "prob": float(prob),
                "pred": int(prob >= 0.5),
                "correct": int((prob >= 0.5) == rec["label"]),
                "short": is_short,
                "duration_s": len(rec["signal"]) / fs,
            })

    if not all_labels:
        print("No MIMIC-III records with labels found.")
        return {}

    logits_np = np.array(all_logits)
    labels_np = np.array(all_labels)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits_np, -50, 50)))

    # Filter out NaN predictions (model produced NaN for some records)
    valid_mask = ~np.isnan(probs)
    n_nan = int(np.sum(~valid_mask))
    probs_valid = probs[valid_mask]
    labels_valid = labels_np[valid_mask]
    preds = (probs_valid >= 0.5).astype(np.int64)

    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

    try:
        auroc = float(roc_auc_score(labels_valid, probs_valid))
    except Exception:
        auroc = float("nan")
    f1 = float(f1_score(labels_valid, preds, zero_division=0))
    cm = confusion_matrix(labels_valid, preds)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    # Print results
    n_af = int(np.sum(labels_np == 1))
    n_nsr = int(np.sum(labels_np == 0))
    print()
    print("=" * 55)
    print("  Cross-Dataset Evaluation (MIMIC-III)  -- NFR-3.1")
    print("=" * 55)
    print(f"  Records:    {len(labels_np)} ({n_af} AF, {n_nsr} NSR)")
    print(f"  Skipped:    {n_skipped}")
    print(f"  NaN output: {n_nan} (excluded from metrics)")
    print(f"  Short:      {n_short} (<300s, HRV=zeros, CNN-only)")
    print()
    print(f"  {'Metric':<14} {'MIMIC-III':>10}")
    print(f"  {'-'*14} {'-'*10}")
    print(f"  {'AUROC':<14} {auroc:>10.4f}")
    print(f"  {'F1':<14} {f1:>10.4f}")
    print(f"  {'Sensitivity':<14} {sensitivity:>10.4f}")
    print(f"  {'Specificity':<14} {specificity:>10.4f}")
    print()

    # Per-record details
    print("  Per-record predictions:")
    for pr in per_record:
        mark = "+" if pr["correct"] else "X"
        short_tag = " [short]" if pr["short"] else ""
        print(f"    {mark} {pr['subject_id']}: true={pr['label']} pred={pr['pred']} "
              f"prob={pr['prob']:.3f} ({pr['duration_s']:.0f}s){short_tag}")
    print("=" * 55)

    # Save plots
    if save_plots and labels_valid.size > 0:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.metrics import RocCurveDisplay

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            RocCurveDisplay.from_predictions(labels_valid, probs_valid, ax=ax, name=f"AUROC={auroc:.3f}")
            ax.set_title("ROC curve (MIMIC-III cross-dataset)")
            roc_path = Path(artifacts_dir) / "roc_mimic3.png"
            fig.savefig(roc_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved ROC to {roc_path}")

            if cm.size >= 1:
                fig2, ax2 = plt.subplots(1, 1, figsize=(4, 4))
                ax2.imshow(cm, cmap="Blues")
                h, w = cm.shape
                ax2.set_xticks(range(w))
                ax2.set_yticks(range(h))
                ax2.set_xticklabels([f"Pred {j}" for j in range(w)])
                ax2.set_yticklabels([f"True {i}" for i in range(h)])
                for i in range(h):
                    for j in range(w):
                        ax2.text(j, i, str(cm[i, j]), ha="center", va="center")
                ax2.set_title("Confusion matrix (MIMIC-III)")
                cm_path = Path(artifacts_dir) / "confusion_mimic3.png"
                fig2.savefig(cm_path, dpi=100, bbox_inches="tight")
                plt.close(fig2)
                print(f"  Saved confusion matrix to {cm_path}")
        except ImportError:
            pass

    return {
        "auroc": auroc, "f1": f1, "sensitivity": sensitivity, "specificity": specificity,
        "n_records": len(labels_np), "n_skipped": n_skipped, "n_short": n_short,
        "per_record": per_record,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid Ensemble on a split.")
    parser.add_argument("--config", default=CONFIG_PATH, help="Config YAML path")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (default: from config)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--no-plots", action="store_true", help="Do not save ROC/confusion plots")
    parser.add_argument("--mimic3", action="store_true",
                        help="Cross-dataset evaluation on MIMIC-III holdout (NFR-3.1)")
    parser.add_argument("--holdout", default=None,
                        help="Path to mimic3_holdout.json — restrict evaluation to unseen holdout subjects")
    args = parser.parse_args()

    if args.mimic3:
        evaluate_mimic3(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            save_plots=not args.no_plots,
            holdout_path=args.holdout,
        )
    else:
        run_evaluation(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            split=args.split,
            save_plots=not args.no_plots,
        )


if __name__ == "__main__":
    main()
