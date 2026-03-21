"""
Phase Detector Evaluation — per-frame accuracy, precision, recall, confusion matrices (S-27).

Evaluates PhaseDetector on phase cache splits:
- In-distribution: test split (CVES + MIMIC-3)
- Out-of-distribution: SHaRe test split

Outputs: per-task metrics (diastole, exhalation) + confusion matrices.
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

from src.training.build_model import load_config

logger = logging.getLogger(__name__)
CONFIG_PATH = "config_stroke.yaml"


def _load_phase_cache(cache_dir: Path, split: str) -> tuple:
    """Load phase cache: (ecg_tensor, labels_dict).

    Returns:
        (ecg_np, diastole_np, exhalation_np, quality_np) where each is (N, *) or (N, 10) for frames.
    """
    ecg_file = cache_dir / f"{split}_ecg.npy"
    diastole_file = cache_dir / f"{split}_diastole.npy"
    exhalation_file = cache_dir / f"{split}_exhalation.npy"
    quality_file = cache_dir / f"{split}_quality.npy"

    if not ecg_file.exists():
        raise FileNotFoundError(f"ECG cache not found: {ecg_file}")

    ecg_np = np.load(ecg_file, mmap_mode='r')
    diastole_np = np.load(diastole_file, mmap_mode='r') if diastole_file.exists() else None
    exhalation_np = np.load(exhalation_file, mmap_mode='r') if exhalation_file.exists() else None
    quality_np = np.load(quality_file, mmap_mode='r') if quality_file.exists() else None

    return ecg_np, diastole_np, exhalation_np, quality_np


def _compute_frame_metrics(labels: np.ndarray, probs: np.ndarray, task_name: str = "task") -> dict:
    """Compute per-frame accuracy, precision, recall, F1 for a single task.

    Args:
        labels: (N_frames,) or shape with NaN for invalid frames
        probs: (N_frames,) predicted probabilities [0,1]
        task_name: Name of the task (for logging)

    Returns:
        {accuracy, precision_class0, precision_class1, recall_class0, recall_class1, f1_class0, f1_class1}
    """
    # Remove NaN frames
    valid_mask = ~np.isnan(labels)
    labels_valid = labels[valid_mask]
    probs_valid = probs[valid_mask]

    if len(labels_valid) == 0:
        return {
            "accuracy": float("nan"),
            "precision_0": float("nan"),
            "precision_1": float("nan"),
            "recall_0": float("nan"),
            "recall_1": float("nan"),
            "f1_0": float("nan"),
            "f1_1": float("nan"),
            "n_frames": 0,
        }

    # Threshold at 0.5
    preds = (probs_valid >= 0.5).astype(np.int64)

    # Accuracy
    accuracy = np.mean(preds == labels_valid)

    # Per-class metrics
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    cm = confusion_matrix(labels_valid, preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle edge case where only one class present
        tn = fp = fn = tp = 0

    prec, recall, f1, _ = precision_recall_fscore_support(
        labels_valid, preds, labels=[0, 1], zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "precision_0": float(prec[0]),
        "precision_1": float(prec[1]),
        "recall_0": float(recall[0]),
        "recall_1": float(recall[1]),
        "f1_0": float(f1[0]),
        "f1_1": float(f1[1]),
        "n_frames": int(len(labels_valid)),
        "confusion_matrix": cm,
    }


def evaluate_phase_detector(
    config_path: str = CONFIG_PATH,
    checkpoint_path: Optional[str] = None,
    split: str = "test",
    cache_dir: Optional[Path] = None,
    save_plots: bool = True,
    artifacts_dir: Optional[str] = None,
) -> dict:
    """Evaluate phase detector on a split (test or val).

    Args:
        config_path: Path to config YAML
        checkpoint_path: Path to checkpoint (default: from config)
        split: 'test' or 'val'
        cache_dir: Phase cache directory (default: from config)
        save_plots: Whether to save confusion matrix plots
        artifacts_dir: Where to save outputs (default: same as cache_dir)

    Returns:
        Dictionary with per-task metrics.
    """
    config = load_config(config_path)
    paths_cfg = config.get("paths", {})
    phase_precompute_cfg = config.get("phase_precompute", {})

    checkpoint_path = checkpoint_path or paths_cfg.get("phase_detect_checkpoint", "models/checkpoints/phase_detector.pth")
    cache_dir = cache_dir or Path(phase_precompute_cfg.get("cache_dir", "models/artifacts/cache_phase_detect"))
    cache_dir = Path(cache_dir)

    if artifacts_dir is None:
        artifacts_dir = cache_dir
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model (build_phase_detector handles config mapping + checkpoint unwrapping)
    from src.models.phase_detector import build_phase_detector

    model = build_phase_detector(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=str(device),
    )

    model.eval()

    # Load cache
    try:
        ecg_np, diastole_np, exhalation_np, quality_np = _load_phase_cache(cache_dir, split)
    except FileNotFoundError as e:
        logger.error(f"Failed to load cache: {e}")
        return {}

    logger.info(f"Loaded {len(ecg_np)} windows from {split} split")

    # Inference
    all_diastole_logits = []
    all_diastole_labels = []
    all_exhalation_logits = []
    all_exhalation_labels = []

    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(ecg_np), batch_size):
            batch_end = min(i + batch_size, len(ecg_np))
            ecg_batch = torch.tensor(ecg_np[i:batch_end], dtype=torch.float32).to(device)

            # Model expects (B, 1, 500)
            if ecg_batch.ndim == 2:
                ecg_batch = ecg_batch.unsqueeze(1)

            # Output: (B, 10, 2) → diastole (B, 10) and exhalation (B, 10)
            logits = model(ecg_batch)  # (B, 10, 2)
            diastole_logits = logits[..., 0]  # (B, 10)
            exhalation_logits = logits[..., 1]  # (B, 10)

            all_diastole_logits.append(diastole_logits.cpu().numpy())
            all_exhalation_logits.append(exhalation_logits.cpu().numpy())

            if diastole_np is not None:
                all_diastole_labels.append(diastole_np[i:batch_end])
            if exhalation_np is not None:
                all_exhalation_labels.append(exhalation_np[i:batch_end])

    diastole_logits_np = np.concatenate(all_diastole_logits, axis=0)  # (N, 10)
    exhalation_logits_np = np.concatenate(all_exhalation_logits, axis=0)  # (N, 10)

    # Convert logits to probabilities
    diastole_probs = 1.0 / (1.0 + np.exp(-np.clip(diastole_logits_np, -50, 50)))
    exhalation_probs = 1.0 / (1.0 + np.exp(-np.clip(exhalation_logits_np, -50, 50)))

    # Flatten for evaluation
    diastole_labels_flat = np.concatenate(all_diastole_labels, axis=0).ravel() if all_diastole_labels else None
    exhalation_labels_flat = np.concatenate(all_exhalation_labels, axis=0).ravel() if all_exhalation_labels else None
    diastole_probs_flat = diastole_probs.ravel()
    exhalation_probs_flat = exhalation_probs.ravel()

    # Compute metrics
    results = {}

    if diastole_labels_flat is not None:
        diastole_metrics = _compute_frame_metrics(diastole_labels_flat, diastole_probs_flat, "diastole")
        results["diastole"] = diastole_metrics

    if exhalation_labels_flat is not None:
        exhalation_metrics = _compute_frame_metrics(exhalation_labels_flat, exhalation_probs_flat, "exhalation")
        results["exhalation"] = exhalation_metrics

    # Print results
    print()
    print("=" * 70)
    print(f"  Phase Detector Evaluation ({split.upper()})")
    print("=" * 70)

    for task_name in ["diastole", "exhalation"]:
        if task_name in results:
            metrics = results[task_name]
            print(f"\n  {task_name.upper()}:")
            print(f"    Frames evaluated: {metrics.get('n_frames', 0)}")
            print(f"    Accuracy:         {metrics.get('accuracy', float('nan')):.4f}")
            print(f"    Precision (0):    {metrics.get('precision_0', float('nan')):.4f}")
            print(f"    Precision (1):    {metrics.get('precision_1', float('nan')):.4f}")
            print(f"    Recall (0):       {metrics.get('recall_0', float('nan')):.4f}")
            print(f"    Recall (1):       {metrics.get('recall_1', float('nan')):.4f}")
            print(f"    F1 (0):           {metrics.get('f1_0', float('nan')):.4f}")
            print(f"    F1 (1):           {metrics.get('f1_1', float('nan')):.4f}")

            if "confusion_matrix" in metrics:
                cm = metrics["confusion_matrix"]
                print(f"    Confusion matrix:\n{cm}")

    print("=" * 70)

    # Save plots
    if save_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            for task_name in ["diastole", "exhalation"]:
                if task_name not in results:
                    continue

                metrics = results[task_name]
                if "confusion_matrix" not in metrics:
                    continue

                cm = metrics["confusion_matrix"]
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(cm, cmap="Blues")
                h, w = cm.shape
                ax.set_xticks(range(w))
                ax.set_yticks(range(h))
                ax.set_xticklabels(["Systole/Inhale", "Diastole/Exhale"][:w])
                ax.set_yticklabels(["Systole/Inhale", "Diastole/Exhale"][:h])
                for i in range(h):
                    for j in range(w):
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
                ax.set_title(f"Confusion Matrix — {task_name} ({split})")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                cm_path = Path(artifacts_dir) / f"confusion_{task_name}_{split}.png"
                fig.savefig(cm_path, dpi=100, bbox_inches="tight")
                plt.close(fig)
                logger.info(f"Saved confusion matrix to {cm_path}")

        except ImportError:
            logger.warning("matplotlib not available, skipping plots")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate PhaseDetector on phase cache splits.")
    parser.add_argument("--config", default=CONFIG_PATH, help="Config YAML path")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (default: from config)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Cache split to evaluate")
    parser.add_argument("--no-plots", action="store_true", help="Do not save confusion matrix plots")
    parser.add_argument("--cache-dir", default=None,
                        help="Override cache directory from config")
    parser.add_argument("--source-breakdown", action="store_true",
                        help="Evaluate on CVES and MIMIC caches separately (requires per-source caches)")
    args = parser.parse_args()

    if args.source_breakdown:
        config = load_config(args.config)
        base_dir = Path(config.get("phase_precompute", {}).get(
            "cache_dir", "models/artifacts/cache_phase_detect"
        )).parent

        for source in ["cves", "mimic"]:
            source_cache = base_dir / f"cache_phase_detect_{source}"
            if not source_cache.exists():
                print(f"\n  {source.upper()} cache not found at {source_cache}, skipping")
                continue
            print(f"\n{'=' * 70}")
            print(f"  Source: {source.upper()}")
            print(f"{'=' * 70}")
            evaluate_phase_detector(
                config_path=args.config,
                checkpoint_path=args.checkpoint,
                split=args.split,
                cache_dir=source_cache,
                save_plots=not args.no_plots,
            )
        return

    evaluate_phase_detector(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        save_plots=not args.no_plots,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
