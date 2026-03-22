"""Training entry point for PhaseDetector CNN.

Multi-task training: diastolic phase + exhalation phase detection from 2s ECG windows.
Labels are frame-level (5Hz, 10 frames/window) with NaN for unknown frames.
"""

import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from .build_model import load_config
from ..models.phase_detector import build_phase_detector

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = "config_stroke.yaml"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PhaseDetectorDataset(Dataset):
    """Load 2s ECG windows + frame-level diastole/exhalation labels from phase cache."""

    def __init__(self, cache_dir: Path, split: str = "train"):
        cache_dir = Path(cache_dir)
        self._ecg = np.load(cache_dir / f"{split}_ecg.npy", mmap_mode="r")
        self._dia = np.load(cache_dir / f"{split}_diastole.npy", mmap_mode="r")
        self._exh = np.load(cache_dir / f"{split}_exhalation.npy", mmap_mode="r")
        self._qual = np.load(cache_dir / f"{split}_quality.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self._ecg)

    def __getitem__(self, i: int):
        ecg = torch.from_numpy(self._ecg[i].astype(np.float32)).unsqueeze(0)  # (1, 500)
        dia = torch.from_numpy(self._dia[i].astype(np.float32))               # (10,)
        exh = torch.from_numpy(self._exh[i].astype(np.float32))               # (10,)
        qual = torch.from_numpy(self._qual[i].astype(np.float32))             # (10,)
        return ecg, dia, exh, qual


# ---------------------------------------------------------------------------
# Loss + metrics
# ---------------------------------------------------------------------------

def compute_phase_pos_weights(
    dataset: PhaseDetectorDataset, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-task pos_weight from non-NaN training labels, clamped at 10."""
    weights = []
    for arr in (dataset._dia, dataset._exh):
        flat = np.array(arr).ravel()
        valid = flat[~np.isnan(flat)]
        n_pos = (valid == 1.0).sum()
        n_neg = (valid == 0.0).sum()
        if n_pos > 0:
            pw = min(float(n_neg) / float(n_pos), 10.0)
        else:
            pw = 1.0
        weights.append(torch.tensor(pw, dtype=torch.float32, device=device))
    return weights[0], weights[1]


def multitask_bce_loss(
    logits: torch.Tensor,
    dia_targets: torch.Tensor,
    exh_targets: torch.Tensor,
    dia_pw: torch.Tensor,
    exh_pw: torch.Tensor,
    dia_weight: float = 0.5,
    exh_weight: float = 0.5,
) -> torch.Tensor:
    """Multi-task BCE with NaN masking and per-task loss weights. Returns scalar loss."""
    dia_logits = logits[:, :, 0]
    exh_logits = logits[:, :, 1]
    dia_mask = ~torch.isnan(dia_targets)
    exh_mask = ~torch.isnan(exh_targets)

    # weights should sum to 1.0; default (0.5, 0.5) matches original equal-avg behavior
    loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
    if dia_mask.any():
        dia_loss = F.binary_cross_entropy_with_logits(
            dia_logits[dia_mask], dia_targets[dia_mask], pos_weight=dia_pw)
        loss = loss + dia_weight * dia_loss
    if exh_mask.any():
        exh_loss = F.binary_cross_entropy_with_logits(
            exh_logits[exh_mask], exh_targets[exh_mask], pos_weight=exh_pw)
        loss = loss + exh_weight * exh_loss
    return loss


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_phase_validation(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    dia_pw: torch.Tensor,
    exh_pw: torch.Tensor,
) -> tuple[float, float, float, float]:
    """Compute val loss + per-task accuracy (threshold 0.5).

    Returns (val_loss, dia_acc, exh_acc, avg_acc).
    """
    model.eval()
    loss_sum, n = 0.0, 0
    dia_correct, dia_total = 0, 0
    exh_correct, exh_total = 0, 0

    with torch.no_grad():
        for ecg, dia, exh, _qual in val_loader:
            ecg = ecg.to(device)
            dia = dia.to(device)
            exh = exh.to(device)

            logits = model(ecg)
            loss = multitask_bce_loss(logits, dia, exh, dia_pw, exh_pw)
            loss_sum += loss.item() * ecg.shape[0]
            n += ecg.shape[0]

            probs = torch.sigmoid(logits)

            # Diastole accuracy
            dia_mask = ~torch.isnan(dia)
            if dia_mask.any():
                dia_pred = (probs[:, :, 0][dia_mask] > 0.5).float()
                dia_correct += (dia_pred == dia[dia_mask]).sum().item()
                dia_total += dia_mask.sum().item()

            # Exhalation accuracy
            exh_mask = ~torch.isnan(exh)
            if exh_mask.any():
                exh_pred = (probs[:, :, 1][exh_mask] > 0.5).float()
                exh_correct += (exh_pred == exh[exh_mask]).sum().item()
                exh_total += exh_mask.sum().item()

    val_loss = loss_sum / n if n > 0 else 0.0
    dia_acc = dia_correct / dia_total if dia_total > 0 else 0.0
    exh_acc = exh_correct / exh_total if exh_total > 0 else 0.0
    avg_acc = (dia_acc + exh_acc) / 2.0
    return val_loss, dia_acc, exh_acc, avg_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train PhaseDetector CNN (multi-task phase detection).")
    parser.add_argument("--config", default=CONFIG_PATH, help="Config YAML path")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    parser.add_argument("--cache-dir", default=None, help="Override cache directory from config")
    parser.add_argument("--checkpoint-out", default=None, help="Override checkpoint output path")
    parser.add_argument("--model-section", default="phase_model",
                        help="Config section for model architecture (e.g. phase_model_5s)")
    parser.add_argument("--train-section", default="phase_training",
                        help="Config section for training params (e.g. phase_training_5s)")
    args = parser.parse_args()

    config = load_config(args.config)
    paths_cfg = config.get("paths", {})
    precompute_cfg = config.get("phase_precompute", {})
    train_cfg = config.get(args.train_section, config.get("phase_training", {}))

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(precompute_cfg.get("cache_dir", "models/artifacts/cache_phase_detect"))
    checkpoint_path = args.checkpoint_out or paths_cfg.get("phase_detect_checkpoint", "models/checkpoints/phase_detector.pth")
    lr = float(train_cfg.get("learning_rate", 1e-3))
    batch_size = int(train_cfg.get("batch_size", 64))
    max_epochs = args.max_epochs if args.max_epochs is not None else int(train_cfg.get("max_epochs", 100))
    patience = int(train_cfg.get("patience", 15))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    dia_weight = float(train_cfg.get("task_weight_dia", 0.5))
    exh_weight = float(train_cfg.get("task_weight_exh", 0.5))

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Verify cache
    required = ["train_ecg.npy", "train_diastole.npy", "train_exhalation.npy",
                 "train_quality.npy", "phase_cache_meta.json"]
    missing = [f for f in required if not (cache_dir / f).exists()]
    if missing:
        logger.error(
            "Phase cache missing or incomplete at %s. Missing: %s. "
            "Run: python -m src.training.stroke_precompute_cache --dataset training",
            cache_dir, missing,
        )
        sys.exit(1)

    # Datasets + loaders
    train_ds = PhaseDetectorDataset(cache_dir, "train")
    try:
        val_ds = PhaseDetectorDataset(cache_dir, "val")
    except FileNotFoundError:
        logger.warning("No val split in cache — skipping validation")
        val_ds = None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0) if val_ds else None

    # Per-task pos_weight
    dia_pw, exh_pw = compute_phase_pos_weights(train_ds, device)
    logger.info("pos_weight — diastole=%.2f  exhalation=%.2f", dia_pw.item(), exh_pw.item())
    logger.info("task_weight — diastole=%.2f  exhalation=%.2f", dia_weight, exh_weight)

    # Build model
    model = build_phase_detector(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=str(device),
        model_section=args.model_section,
    )
    logger.info("PhaseDetector: %d params", model.param_count())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5, min_lr=1e-6,
    )
    best_avg_acc = 0.0
    epochs_without_improvement = 0

    log_path = cache_dir / "phase_training_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_dia_acc", "val_exh_acc", "val_avg_acc", "lr"])

    for epoch in range(max_epochs):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}", unit="batch", leave=False)

        for ecg, dia, exh, _qual in pbar:
            ecg = ecg.to(device)
            dia = dia.to(device)
            exh = exh.to(device)

            optimizer.zero_grad()
            logits = model(ecg)
            loss = multitask_bce_loss(logits, dia, exh, dia_pw, exh_pw, dia_weight, exh_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            train_loss_sum += loss.item() * ecg.shape[0]
            train_n += ecg.shape[0]
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = train_loss_sum / train_n if train_n > 0 else 0.0

        if val_loader is not None and len(val_loader.dataset) > 0:
            val_loss, dia_acc, exh_acc, avg_acc = run_phase_validation(
                model, val_loader, device, dia_pw, exh_pw)
        else:
            val_loss, dia_acc, exh_acc, avg_acc = 0.0, 0.0, 0.0, 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %d  train_loss=%.4f  val_loss=%.4f  dia_acc=%.4f  exh_acc=%.4f  avg_acc=%.4f  lr=%.2e",
            epoch + 1, train_loss, val_loss, dia_acc, exh_acc, avg_acc, current_lr,
        )
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}",
                f"{dia_acc:.6f}", f"{exh_acc:.6f}", f"{avg_acc:.6f}", f"{current_lr:.2e}",
            ])

        scheduler.step(avg_acc)

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            epochs_without_improvement = 0
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_dia_acc": dia_acc,
                "val_exh_acc": exh_acc,
                "val_avg_acc": avg_acc,
            }, checkpoint_path)
            logger.info("Saved best checkpoint (avg_acc=%.4f) to %s", avg_acc, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)",
                            epoch + 1, patience)
                break

    logger.info("Training finished. Best avg_acc=%.4f", best_avg_acc)


if __name__ == "__main__":
    main()
