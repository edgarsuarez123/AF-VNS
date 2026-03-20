"""Training entry point for StrokeHybridEnsemble.

Phase 1: Full model (backbone + head) trained on MIMIC-3 stroke cache.
Phase 2: Frozen backbone, head-only fine-tuning on CereVasc (CVES) cache.

Infrastructure (PrecomputedDataset, _pad_collate, run_validation_from_cache, etc.)
is imported directly from train.py — not copied.
"""

import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from .train import (
    PrecomputedDataset,
    _EmptyCacheDataset,
    _pad_collate,
    _fit_scaler_from_train_dataset,
    run_validation_from_cache,
    run_validation,
)
from .build_model import load_config
from ..models.stroke_ensemble import build_stroke_model
from ..features.scaler import load_scaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = "config_stroke.yaml"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train StrokeHybridEnsemble (Phase 1 or 2).")
    parser.add_argument("--config", default=CONFIG_PATH, help="Config YAML path")
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Train from precomputed cache (run stroke_precompute_cache first)",
    )
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1,
                        help="1=pre-train all layers on MIMIC-3, 2=freeze backbone fine-tune head on CVES")
    parser.add_argument("--phase1-checkpoint", default=None,
                        help="Path to Phase 1 checkpoint (required for --phase 2)")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override max epochs (e.g. 1 for smoke run)")
    args = parser.parse_args()

    config = load_config(args.config)
    paths_cfg = config.get("paths", {})
    train_cfg = config.get("training", {})
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))

    phase = args.phase
    if phase == 1:
        checkpoint_path = paths_cfg.get("phase1_checkpoint", "models/checkpoints/stroke_phase1_model.pth")
        scaler_path = paths_cfg.get("phase1_scaler", "models/artifacts/stroke_phase1_scaler.pkl")
        cache_dir = Path(paths_cfg.get("phase1_cache_dir", "models/artifacts/cache_stroke_phase1"))
        split_path = paths_cfg.get("phase1_split", "models/artifacts/stroke_phase1_split.json")
        lr = float(train_cfg.get("learning_rate", 1e-3))
        max_epochs = args.max_epochs if args.max_epochs is not None else int(train_cfg.get("max_epochs", 100))
        patience = 15
    else:  # phase == 2
        checkpoint_path = paths_cfg.get("phase2_checkpoint", "models/checkpoints/stroke_phase2_model.pth")
        scaler_path = paths_cfg.get("phase2_scaler", "models/artifacts/stroke_phase2_scaler.pkl")
        cache_dir = Path(paths_cfg.get("phase2_cache_dir", "models/artifacts/cache_stroke_phase2"))
        split_path = paths_cfg.get("phase2_split", "models/artifacts/stroke_phase2_split.json")
        lr = 5e-4
        max_epochs = args.max_epochs if args.max_epochs is not None else 50
        patience = 10
        if not args.phase1_checkpoint:
            args.phase1_checkpoint = paths_cfg.get("phase1_checkpoint", "models/checkpoints/stroke_phase1_model.pth")

    batch_size = int(train_cfg.get("batch_size", 32))

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    use_cache = args.use_cache
    if use_cache:
        required = ["train_short.npy", "train_hrv_scaled.npy", "train_labels.npy", "cache_meta.json"]
        missing = [f for f in required if not (cache_dir / f).exists()]
        if missing:
            logger.error(
                "Cache missing or incomplete. Required: %s. Missing: %s. "
                "Run: python -m src.training.stroke_precompute_cache",
                required,
                missing,
            )
            sys.exit(1)
        train_ds = PrecomputedDataset(cache_dir, "train", is_train=True, config_path=args.config)
        try:
            val_ds = PrecomputedDataset(cache_dir, "val", is_train=False, config_path=args.config)
        except FileNotFoundError:
            import json
            with open(cache_dir / "cache_meta.json") as f:
                meta = json.load(f)
            max_t = meta.get("max_short_len", 3000)
            val_ds = _EmptyCacheDataset(max_t)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=0
        )
        scaler = None
    else:
        from ..data.stroke_dataloaders import get_stroke_dataloaders
        train_loader, val_loader, _ = get_stroke_dataloaders(
            config_path=args.config,
            split_path=split_path,
            batch_size=batch_size,
            phase=phase,
        )
        train_loader = torch.utils.data.DataLoader(
            train_loader.dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, collate_fn=_pad_collate,
        )
        val_loader = torch.utils.data.DataLoader(
            val_loader.dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=_pad_collate,
        )
        logger.info("Fitting scaler on training set...")
        _fit_scaler_from_train_dataset(train_loader.dataset, args.config, scaler_path)
        scaler = load_scaler(path=scaler_path, config_path=args.config)

    # Compute pos_weight from training label distribution
    all_train_labels = []
    for batch in train_loader:
        all_train_labels.append(batch[2])
    all_train_labels = torch.cat(all_train_labels)
    n_pos = (all_train_labels == 1).sum().float()
    n_neg = (all_train_labels == 0).sum().float()
    if n_pos > 0:
        pos_weight = (n_neg / n_pos).clamp(max=10.0)
    else:
        pos_weight = torch.tensor(1.0)
    pos_weight = pos_weight.to(device)
    logger.info("Class balance: %d positive, %d negative, pos_weight=%.2f",
                int(n_pos), int(n_neg), pos_weight.item())

    # Build model
    if phase == 2:
        logger.info("Phase 2: loading Phase 1 checkpoint from %s", args.phase1_checkpoint)
        model = build_stroke_model(config_path=args.config,
                                   checkpoint_path=args.phase1_checkpoint, device=device)
        model.freeze_backbone()
        model.cnn.eval()
        model.rnn.eval()
        model.transformer.eval()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info("Phase 2: frozen backbone. Trainable: %d / %d params (head only)", trainable, total)
        train_params = model.head.parameters()
    else:
        model = build_stroke_model(config_path=args.config, checkpoint_path=None, device=device)
        train_params = model.parameters()

    optimizer = torch.optim.AdamW(train_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5, min_lr=1e-6
    )
    best_auroc = 0.0
    epochs_without_improvement = 0

    log_path = Path(scaler_path).parent / "stroke_training_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_auroc"])

    # Phase 2: override train() to keep backbone in eval mode
    _original_train = model.train
    if phase == 2:
        def _phase2_train(mode=True):
            _original_train(mode)
            model.cnn.eval()
            model.rnn.eval()
            model.transformer.eval()
            return model
        model.train = _phase2_train

    for epoch in range(max_epochs):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{max_epochs}",
            unit="batch",
            leave=False,
        )
        if use_cache:
            for batch in pbar:
                short_t, hrv_t, labels_t = batch[0], batch[1], batch[2]
                hrv_lengths = batch[3] if len(batch) > 3 else None
                short_t = short_t.to(device)
                hrv_t = hrv_t.to(device)
                labels_t = labels_t.to(device)
                if hrv_lengths is not None:
                    hrv_lengths = hrv_lengths.to(device)
                if label_smoothing > 0:
                    labels_t = labels_t * (1 - label_smoothing) + label_smoothing * 0.5
                optimizer.zero_grad()
                logits = model(short_t, hrv_t, hrv_lengths=hrv_lengths).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, labels_t, pos_weight=pos_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss_sum += loss.item() * short_t.shape[0]
                train_n += short_t.shape[0]
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        else:
            from .train import _batch_to_device_and_model
            for batch in pbar:
                short_pad, long_pad, labels, fs_list = batch
                short_t, hrv_t, labels_t = _batch_to_device_and_model(
                    short_pad, long_pad, labels, fs_list, scaler, device, args.config
                )
                if label_smoothing > 0:
                    labels_t = labels_t * (1 - label_smoothing) + label_smoothing * 0.5
                optimizer.zero_grad()
                logits = model(short_t, hrv_t).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, labels_t, pos_weight=pos_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss_sum += loss.item() * short_t.shape[0]
                train_n += short_t.shape[0]
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = train_loss_sum / train_n if train_n else 0.0
        if use_cache:
            val_loss, val_auroc = run_validation_from_cache(model, val_loader, device)
        else:
            val_loss, val_auroc = run_validation(model, val_loader, scaler, device, args.config)
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %d  train_loss=%.4f  val_loss=%.4f  val_auroc=%.4f  lr=%.2e",
            epoch + 1, train_loss, val_loss, val_auroc, current_lr,
        )
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_auroc:.6f}"])

        scheduler.step(val_auroc)

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            epochs_without_improvement = 0
            torch.save(
                {"state_dict": model.state_dict(), "epoch": epoch + 1, "val_auroc": val_auroc},
                checkpoint_path,
            )
            logger.info("Saved best checkpoint (val_auroc=%.4f) to %s", val_auroc, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)",
                            epoch + 1, patience)
                break

    logger.info("Training finished. Best val_auroc=%.4f", best_auroc)


if __name__ == "__main__":
    main()
