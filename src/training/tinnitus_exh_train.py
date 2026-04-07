"""
F20 — Tinnitus exhalation-only PhaseDetector training.

Single-task BCE loss on exhalation frames.  Input: (B, 2, 750) 6s two-channel
PPG+RIIV windows.  Output: (B, 30, 1) frame-level logits.

Usage:
    python -m src.training.tinnitus_exh_train --config config_tinnitus.yaml
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from .build_model import load_config
from ..models.phase_detector import build_phase_detector

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = "config_tinnitus.yaml"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TinnitusExhDataset(Dataset):
    """Load 6s two-channel PPG+RIIV windows + exhalation labels from exh cache."""

    def __init__(self, cache_dir: Path, split: str = "train"):
        cache_dir = Path(cache_dir)
        self._ppg = np.load(cache_dir / f"{split}_ppg.npy", mmap_mode="r")   # (N, 2, 750)
        self._exh = np.load(cache_dir / f"{split}_exhalation.npy", mmap_mode="r")  # (N, 30)
        self._qual = np.load(cache_dir / f"{split}_quality.npy", mmap_mode="r")    # (N, 30)

    def __len__(self) -> int:
        return len(self._ppg)

    def __getitem__(self, i: int):
        ppg = torch.from_numpy(self._ppg[i].astype(np.float32))   # (2, 750)
        exh = torch.from_numpy(self._exh[i].astype(np.float32))   # (30,)
        qual = torch.from_numpy(self._qual[i].astype(np.float32)) # (30,)
        return ppg, exh, qual


# ---------------------------------------------------------------------------
# Loss / validation
# ---------------------------------------------------------------------------

def _compute_pos_weight(dataset: TinnitusExhDataset, device: torch.device) -> torch.Tensor:
    """Compute exhalation positive class weight from training labels."""
    all_labels = dataset._exh.ravel()
    valid = all_labels[~np.isnan(all_labels)]
    n_pos = float(np.sum(valid > 0.5))
    n_neg = float(np.sum(valid <= 0.5))
    pw = n_neg / n_pos if n_pos > 0 else 1.0
    pw = float(np.clip(pw, 0.1, 10.0))
    logger.info("Exhalation pos_weight=%.2f  (n_pos=%d, n_neg=%d)", pw, int(n_pos), int(n_neg))
    return torch.tensor(pw, device=device)


def _exh_bce_loss(
    logits: torch.Tensor,
    exh_labels: torch.Tensor,
    pos_weight: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """BCE loss on exhalation frames with NaN masking and label smoothing.

    Args:
        logits:     (B, 30, 1) raw logits
        exh_labels: (B, 30) float labels in {0, 1, NaN}
        pos_weight: scalar positive class weight
        label_smoothing: smoothing amount (0 = off)
    """
    logits_squeezed = logits.squeeze(-1)  # (B, 30)
    mask = ~torch.isnan(exh_labels)

    if not mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    targets = exh_labels[mask]
    if label_smoothing > 0:
        targets = targets * (1 - label_smoothing) + label_smoothing * 0.5

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return loss_fn(logits_squeezed[mask], targets)


def _run_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor,
) -> tuple[float, float]:
    """Returns (val_loss, exh_accuracy)."""
    model.eval()
    total_loss, total_n = 0.0, 0
    correct, valid_count = 0, 0

    with torch.no_grad():
        for ppg, exh, _qual in loader:
            ppg = ppg.to(device)
            exh = exh.to(device)
            logits = model(ppg)  # (B, 30, 1)
            loss = _exh_bce_loss(logits, exh, pos_weight)
            total_loss += loss.item() * ppg.shape[0]
            total_n += ppg.shape[0]

            # Accuracy
            probs = torch.sigmoid(logits.squeeze(-1))
            preds = (probs > 0.5).float()
            mask = ~torch.isnan(exh)
            correct += int((preds[mask] == exh[mask].float().round()).sum())
            valid_count += int(mask.sum())

    val_loss = total_loss / total_n if total_n > 0 else 0.0
    exh_acc = correct / valid_count if valid_count > 0 else 0.0
    return val_loss, exh_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Train single-task exhalation PhaseDetector (F20)")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--checkpoint-out", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config_path = args.config
    if not Path(config_path).is_absolute() and not Path(config_path).exists():
        config_path = str(root / config_path)

    config = load_config(config_path)
    exh_train_cfg = config.get("exh_phase_training", {})
    paths_exh_cfg = config.get("paths_exh", {})

    def _resolve(p: str) -> str:
        return p if Path(p).is_absolute() else str(root / p)

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(
        _resolve(paths_exh_cfg.get("cache_dir", "models/artifacts/cache_tinnitus_exh")))
    checkpoint_path = args.checkpoint_out or _resolve(
        paths_exh_cfg.get("phase_detect_checkpoint", "models/checkpoints/tinnitus_exh_detector.pth"))

    lr = float(exh_train_cfg.get("learning_rate", 1e-3))
    batch_size = int(exh_train_cfg.get("batch_size", 64))
    max_epochs = args.max_epochs if args.max_epochs is not None else int(
        exh_train_cfg.get("max_epochs", 100))
    patience = int(exh_train_cfg.get("patience", 15))
    grad_clip = float(exh_train_cfg.get("grad_clip", 1.0))
    label_smoothing = float(exh_train_cfg.get("label_smoothing", 0.0))

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Verify cache
    required = ["train_ppg.npy", "train_exhalation.npy", "train_quality.npy", "exh_cache_meta.json"]
    missing = [f for f in required if not (cache_dir / f).exists()]
    if missing:
        logger.error(
            "Exhalation cache missing files: %s. "
            "Run: python -m src.training.tinnitus_precompute_exh_cache --config config_tinnitus.yaml",
            missing)
        sys.exit(1)

    train_ds = TinnitusExhDataset(cache_dir, "train")
    try:
        val_ds = TinnitusExhDataset(cache_dir, "val")
    except FileNotFoundError:
        logger.warning("No val split in exh cache — skipping validation")
        val_ds = None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0) if val_ds else None

    pos_weight = _compute_pos_weight(train_ds, device)

    # Build model — exh_phase_model section: 6s window, 2-channel, n_tasks=1
    model = build_phase_detector(
        config_path=config_path,
        checkpoint_path=args.checkpoint,
        device=str(device),
        model_section="exh_phase_model",
    )
    logger.info("ExhPhaseDetector: %d params  input=(%d, 2, 750)",
                model.param_count(), batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5, min_lr=1e-6)

    best_exh_acc = 0.0
    epochs_without_improvement = 0

    log_path = cache_dir / "exh_training_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_exh_acc", "lr"])

    for epoch in range(max_epochs):
        model.train()
        train_loss_sum, train_n = 0.0, 0

        for ppg, exh, _qual in train_loader:
            ppg = ppg.to(device)
            exh = exh.to(device)

            optimizer.zero_grad()
            logits = model(ppg)   # (B, 30, 1)
            loss = _exh_bce_loss(logits, exh, pos_weight, label_smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            train_loss_sum += loss.item() * ppg.shape[0]
            train_n += ppg.shape[0]

        train_loss = train_loss_sum / train_n if train_n > 0 else 0.0

        if val_loader is not None:
            val_loss, exh_acc = _run_validation(model, val_loader, device, pos_weight)
        else:
            val_loss, exh_acc = 0.0, 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %d  train_loss=%.4f  val_loss=%.4f  exh_acc=%.4f  lr=%.2e",
            epoch + 1, train_loss, val_loss, exh_acc, current_lr)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}",
                f"{exh_acc:.6f}", f"{current_lr:.2e}",
            ])

        scheduler.step(exh_acc)

        if exh_acc > best_exh_acc:
            best_exh_acc = exh_acc
            epochs_without_improvement = 0
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_exh_acc": exh_acc,
            }, checkpoint_path)
            logger.info("Saved checkpoint (exh_acc=%.4f) → %s", exh_acc, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    logger.info("Done. Best exh_acc=%.4f", best_exh_acc)


if __name__ == "__main__":
    main()
