"""Training entry point for PhaseDetector CNN — tinnitus PPG variant.

Reuses phase_train.py loss/validation/model; only swaps ECG→PPG cache filenames
and points at the tinnitus config/cache/checkpoint paths.
"""

import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from .build_model import load_config
from ..models.phase_detector import build_phase_detector
from .phase_train import (
    PhaseDetectorDataset,
    compute_phase_pos_weights,
    multitask_bce_loss,
    run_phase_validation,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = "config_tinnitus.yaml"


# ---------------------------------------------------------------------------
# PPG Dataset — identical to PhaseDetectorDataset but loads _ppg.npy
# ---------------------------------------------------------------------------

class TinnitusPhaseDataset(PhaseDetectorDataset):
    """Load 2s PPG windows + frame-level labels from tinnitus phase cache.

    Supports single-channel (N, 250) and multi-channel (N, C, 250) PPG arrays.
    Multi-channel arrays (C>1) are returned as-is; single-channel arrays get
    unsqueeze(0) to produce (1, 250) tensors for PhaseDetector.
    """

    def __init__(self, cache_dir: Path, split: str = "train"):
        cache_dir = Path(cache_dir)
        # Override: load _ppg.npy instead of _ecg.npy
        self._ecg = np.load(cache_dir / f"{split}_ppg.npy", mmap_mode="r")
        self._dia = np.load(cache_dir / f"{split}_diastole.npy", mmap_mode="r")
        self._exh = np.load(cache_dir / f"{split}_exhalation.npy", mmap_mode="r")
        self._qual = np.load(cache_dir / f"{split}_quality.npy", mmap_mode="r")
        # Detect multi-channel: shape (N, C, W) vs single-channel (N, W)
        self._multichannel = self._ecg.ndim == 3

    def __getitem__(self, i: int):
        raw = self._ecg[i].astype(np.float32)
        if self._multichannel:
            ppg = torch.from_numpy(raw)           # (C, W) — already channel-first
        else:
            ppg = torch.from_numpy(raw).unsqueeze(0)  # (1, W)
        dia = torch.from_numpy(self._dia[i].astype(np.float32))
        exh = torch.from_numpy(self._exh[i].astype(np.float32))
        qual = torch.from_numpy(self._qual[i].astype(np.float32))
        return ppg, dia, exh, qual


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train PhaseDetector CNN on tinnitus PPG cache.")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--checkpoint-out", default=None)
    parser.add_argument("--model-section", default="phase_model")
    parser.add_argument("--train-section", default="phase_training")
    args = parser.parse_args()

    config = load_config(args.config)
    paths_cfg = config.get("paths", {})
    train_cfg = config.get(args.train_section, config.get("phase_training", {}))

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(
        config.get("paths", {}).get("cache_dir", "models/artifacts/cache_tinnitus_phase"))
    checkpoint_path = args.checkpoint_out or paths_cfg.get(
        "phase_detect_checkpoint", "models/checkpoints/tinnitus_phase_detector.pth")
    lr = float(train_cfg.get("learning_rate", 1e-3))
    batch_size = int(train_cfg.get("batch_size", 64))
    max_epochs = args.max_epochs if args.max_epochs is not None else int(
        train_cfg.get("max_epochs", 100))
    patience = int(train_cfg.get("patience", 15))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    dia_weight = float(train_cfg.get("task_weight_dia", 0.5))
    exh_weight = float(train_cfg.get("task_weight_exh", 0.5))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Verify cache
    required = ["train_ppg.npy", "train_diastole.npy", "train_exhalation.npy",
                "train_quality.npy", "tinnitus_cache_meta.json"]
    missing = [f for f in required if not (cache_dir / f).exists()]
    if missing:
        logger.error(
            "Tinnitus phase cache missing at %s. Missing: %s. "
            "Run: python -m src.training.tinnitus_precompute_cache --config config_tinnitus.yaml",
            cache_dir, missing)
        sys.exit(1)

    # Datasets + loaders
    train_ds = TinnitusPhaseDataset(cache_dir, "train")
    try:
        val_ds = TinnitusPhaseDataset(cache_dir, "val")
    except FileNotFoundError:
        logger.warning("No val split in cache — skipping validation")
        val_ds = None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0) if val_ds else None

    # Per-task pos_weight
    dia_pw, exh_pw = compute_phase_pos_weights(train_ds, device)
    logger.info("pos_weight — diastole=%.2f  exhalation=%.2f", dia_pw.item(), exh_pw.item())
    logger.info("task_weight — diastole=%.2f  exhalation=%.2f", dia_weight, exh_weight)
    if label_smoothing > 0:
        logger.info("label_smoothing=%.3f (0→%.3f, 1→%.3f)",
                     label_smoothing, label_smoothing * 0.5, 1 - label_smoothing * 0.5)

    # Build model (input_samples=250 from config_tinnitus.yaml phase_model section)
    model = build_phase_detector(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=str(device),
        model_section=args.model_section,
    )
    logger.info("PhaseDetector: %d params", model.param_count())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5, min_lr=1e-6)
    best_avg_acc = 0.0
    epochs_without_improvement = 0

    log_path = cache_dir / "tinnitus_training_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "val_dia_acc", "val_exh_acc", "val_avg_acc", "lr"])

    for epoch in range(max_epochs):
        model.train()
        train_loss_sum, train_n = 0.0, 0

        for ppg, dia, exh, _qual in train_loader:
            ppg = ppg.to(device)
            dia = dia.to(device)
            exh = exh.to(device)

            if label_smoothing > 0:
                dia = dia * (1 - label_smoothing) + label_smoothing * 0.5
                exh = exh * (1 - label_smoothing) + label_smoothing * 0.5

            optimizer.zero_grad()
            logits = model(ppg)
            loss = multitask_bce_loss(logits, dia, exh, dia_pw, exh_pw, dia_weight, exh_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            train_loss_sum += loss.item() * ppg.shape[0]
            train_n += ppg.shape[0]

        train_loss = train_loss_sum / train_n if train_n > 0 else 0.0

        if val_loader is not None and len(val_loader.dataset) > 0:
            val_loss, dia_acc, exh_acc, avg_acc = run_phase_validation(
                model, val_loader, device, dia_pw, exh_pw, dia_weight, exh_weight)
        else:
            val_loss, dia_acc, exh_acc, avg_acc = 0.0, 0.0, 0.0, 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %d  train_loss=%.4f  val_loss=%.4f  dia_acc=%.4f  exh_acc=%.4f  avg_acc=%.4f  lr=%.2e",
            epoch + 1, train_loss, val_loss, dia_acc, exh_acc, avg_acc, current_lr)
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
