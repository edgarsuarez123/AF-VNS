"""
Evaluation script: load best checkpoint, run on a split, report AUROC/F1/Sensitivity/Specificity.
Save ROC curve and optional confusion matrix to models/artifacts/.
"""

import argparse
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
from src.features.pipeline import waveform_10s_denoised, waveform_to_hrv_sequence
from src.features.scaler import load_scaler, transform

from .build_model import build_model, load_config
from .train import _batch_to_device_and_model, _pad_collate

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


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid Ensemble on a split.")
    parser.add_argument("--config", default=CONFIG_PATH, help="Config YAML path")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (default: from config)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--no-plots", action="store_true", help="Do not save ROC/confusion plots")
    args = parser.parse_args()
    run_evaluation(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        save_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
