"""
Training script for Hybrid Ensemble (Phase 1).
Fit scaler on train only; on-the-fly HRV from 5-min window; save best checkpoint by val AUROC.
"""

import csv
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Project root for config and imports when run as script
if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from src.data.dataloaders import get_dataloaders
from src.features.pipeline import waveform_10s_denoised, waveform_to_hrv_sequence
from src.features.scaler import fit_scaler, load_scaler, transform

from .build_model import build_model, load_config


class _EmptyCacheDataset(torch.utils.data.Dataset):
    """Placeholder dataset with 0 samples when val cache files are missing."""

    def __init__(self, max_short_len: int):
        self._max_short_len = max_short_len

    def __len__(self):
        return 0

    def __getitem__(self, i):
        raise IndexError("Empty dataset")


class PrecomputedDataset(torch.utils.data.Dataset):
    """Dataset that loads precomputed short (denoised 10s), scaled HRV, and labels from cache.

    Supports HRV lengths for attention masking and on-the-fly augmentation for training.
    """

    def __init__(self, cache_dir: Path, split: str, is_train: bool = False,
                 config_path: str = "config.yaml"):
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.is_train = is_train
        self._short = np.load(self.cache_dir / f"{split}_short.npy", mmap_mode="r")
        self._hrv = np.load(self.cache_dir / f"{split}_hrv_scaled.npy", mmap_mode="r")
        self._labels = np.load(self.cache_dir / f"{split}_labels.npy", mmap_mode="r")

        # HRV lengths (backward compat: default to seq_len if file missing)
        lengths_path = self.cache_dir / f"{split}_hrv_lengths.npy"
        if lengths_path.exists():
            self._hrv_lengths = np.load(lengths_path, mmap_mode="r")
        else:
            self._hrv_lengths = None

        # Augmentation setup (training only)
        self._augment_fn = None
        if is_train:
            config = load_config(config_path)
            aug_cfg = config.get("augmentation", {})
            if aug_cfg.get("enabled", False):
                from src.features.augmentation import augment_waveform
                data_cfg = config.get("data", {})
                self._aug_fs = float(data_cfg.get("target_fs", 250))
                self._aug_cfg = aug_cfg
                self._augment_fn = augment_waveform

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, i):
        short_arr = self._short[i : i + 1].astype(np.float32).copy()  # (1, max_T)

        # On-the-fly augmentation for training
        if self._augment_fn is not None:
            rng = np.random.default_rng()
            short_arr[0] = self._augment_fn(short_arr[0], self._aug_fs,
                                             config=self._aug_cfg, rng=rng)

        short_i = torch.from_numpy(short_arr)  # (1, max_T)
        hrv_i = torch.from_numpy(self._hrv[i].astype(np.float32))  # (seq_len, 7)
        label_i = torch.tensor(self._labels[i], dtype=torch.float32)

        if self._hrv_lengths is not None:
            hrv_len_i = torch.tensor(int(self._hrv_lengths[i]), dtype=torch.long)
        else:
            hrv_len_i = torch.tensor(self._hrv[i].shape[0], dtype=torch.long)

        return short_i, hrv_i, label_i, hrv_len_i

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Default config path when run as script
CONFIG_PATH = "config.yaml"
N_FEATURES = 7
HRV_STEPS = 5


def _pad_collate(batch):
    """Collate 4-tuples (short, long, label, fs); pad short and long to max length in batch."""
    shorts = [b[0] for b in batch]  # each (1, T_i)
    longs = [b[1] for b in batch]
    labels = torch.stack([b[2] for b in batch])
    fs_list = [b[3] for b in batch]

    max_short = max(s.shape[1] for s in shorts)
    max_long = max(s.shape[1] for s in longs)
    short_padded = torch.stack(
        [F.pad(s, (0, max_short - s.shape[1]), value=0.0) for s in shorts]
    )
    long_padded = torch.stack(
        [F.pad(s, (0, max_long - s.shape[1]), value=0.0) for s in longs]
    )
    return short_padded, long_padded, labels, fs_list


def _fit_scaler_from_train_dataset(train_dataset, config_path: str, scaler_path: str):
    """One pass over train dataset: compute HRV per sample, collect rows, fit scaler."""
    all_rows = []
    n = len(train_dataset)
    for i in tqdm(range(n), desc="Fitting scaler", unit="sample"):
        _, long_t, _, fs = train_dataset[i]
        long_np = long_t.squeeze(0).numpy()
        hrv = waveform_to_hrv_sequence(long_np, float(fs), config_path=config_path)
        all_rows.append(hrv)
    if not all_rows:
        raise ValueError("No training samples for scaler fit")
    X = np.vstack(all_rows)
    fit_scaler(X, path=scaler_path, config_path=config_path)
    logger.info("Fitted scaler on %d HRV rows (per-column NaN handled internally), saved to %s", X.shape[0], scaler_path)


def _batch_to_device_and_model(
    short_padded, long_padded, labels, fs_list, scaler, device, config_path: str
):
    """
    Compute HRV and denoised 10s for batch; scale HRV; impute NaN with 0; return tensors on device.
    """
    B = short_padded.shape[0]
    hrv_list = []
    short_denoised_list = []
    for b in range(B):
        long_np = long_padded[b].squeeze(0).numpy()
        short_np = short_padded[b].squeeze(0).numpy()
        fs = float(fs_list[b])
        hrv_b = waveform_to_hrv_sequence(long_np, fs, config_path=config_path)
        short_d = waveform_10s_denoised(short_np, fs, config_path=config_path)
        hrv_list.append(hrv_b)
        short_denoised_list.append(short_d)

    hrv_batch = np.stack(hrv_list, axis=0).astype(np.float32)
    hrv_batch = transform(hrv_batch, scaler)
    np.nan_to_num(hrv_batch, nan=0.0, copy=False)
    hrv_t = torch.from_numpy(hrv_batch).float().to(device)

    # Pad denoised 10s to same length in batch (may vary if fs varies)
    lens = [s.size for s in short_denoised_list]
    max_len = max(lens)
    short_tensors = []
    for s in short_denoised_list:
        t = torch.from_numpy(np.asarray(s, dtype=np.float32))
        short_tensors.append(F.pad(t.unsqueeze(0).unsqueeze(0), (0, max_len - t.numel()), value=0.0))
    short_t = torch.cat(short_tensors, dim=0).to(device)

    labels_t = labels.to(device)
    return short_t, hrv_t, labels_t


def run_validation(model, val_loader, scaler, device, config_path: str):
    """Run validation set; return val_loss and val_auroc."""
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            short_pad, long_pad, labels, fs_list = batch
            short_t, hrv_t, labels_t = _batch_to_device_and_model(
                short_pad, long_pad, labels, fs_list, scaler, device, config_path
            )
            logits = model(short_t, hrv_t).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels_t)
            total_loss += loss.item() * short_t.shape[0]
            n += short_t.shape[0]
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels_t.cpu().numpy())
    if n == 0:
        return 0.0, 0.0
    val_loss = total_loss / n
    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits_np, -50, 50)))
    try:
        from sklearn.metrics import roc_auc_score
        val_auroc = float(roc_auc_score(labels_np, probs))
    except Exception:
        val_auroc = 0.0
    return val_loss, val_auroc


def run_validation_from_cache(model, val_loader, device):
    """Run validation from precomputed cache batches (short, hrv, labels, hrv_lengths)."""
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            short_t, hrv_t, labels_t = batch[0], batch[1], batch[2]
            hrv_lengths = batch[3] if len(batch) > 3 else None
            short_t = short_t.to(device)
            hrv_t = hrv_t.to(device)
            labels_t = labels_t.to(device)
            if hrv_lengths is not None:
                hrv_lengths = hrv_lengths.to(device)
            logits = model(short_t, hrv_t, hrv_lengths=hrv_lengths).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels_t)
            total_loss += loss.item() * short_t.shape[0]
            n += short_t.shape[0]
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels_t.cpu().numpy())
    if n == 0:
        return 0.0, 0.0
    val_loss = total_loss / n
    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits_np, -50, 50)))
    try:
        from sklearn.metrics import roc_auc_score
        val_auroc = float(roc_auc_score(labels_np, probs))
    except Exception:
        val_auroc = 0.0
    return val_loss, val_auroc


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Hybrid Ensemble (Phase 1).")
    parser.add_argument("--config", default=CONFIG_PATH, help="Config YAML path")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs (e.g. 1 for smoke run)")
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Train from precomputed cache (run python -m src.training.precompute_cache first)",
    )
    parser.add_argument("--phase", type=int, choices=[0, 1, 2], default=0,
                        help="0=legacy, 1=pre-train all layers on MIMIC, 2=freeze backbone, fine-tune head")
    parser.add_argument("--phase1-checkpoint", default=None,
                        help="Path to Phase 1 checkpoint (required for --phase 2)")
    args = parser.parse_args()

    config = load_config(args.config)
    paths_cfg = config.get("paths", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))

    # Phase-aware paths and hyperparameters
    phase = args.phase
    if phase == 1:
        checkpoint_path = paths_cfg.get("phase1_checkpoint", "models/checkpoints/phase1_model.pth")
        scaler_path = paths_cfg.get("phase1_scaler", "models/artifacts/phase1_scaler.pkl")
        cache_dir = Path(paths_cfg.get("phase1_cache_dir", "models/artifacts/cache_phase1"))
        split_path = paths_cfg.get("phase1_split", "models/artifacts/phase1_split.json")
        lr = float(train_cfg.get("learning_rate", 1e-3))
        max_epochs = args.max_epochs if args.max_epochs is not None else int(train_cfg.get("max_epochs", 100))
        patience = 15
    elif phase == 2:
        checkpoint_path = paths_cfg.get("phase2_checkpoint", "models/checkpoints/phase2_model.pth")
        scaler_path = paths_cfg.get("phase2_scaler", "models/artifacts/phase2_scaler.pkl")
        cache_dir = Path(paths_cfg.get("phase2_cache_dir", "models/artifacts/cache_phase2"))
        split_path = paths_cfg.get("phase2_split", "models/artifacts/phase2_split.json")
        lr = 5e-4  # lower LR for fine-tuning
        max_epochs = args.max_epochs if args.max_epochs is not None else 50
        patience = 10
        if not args.phase1_checkpoint:
            # Auto-resolve Phase 1 checkpoint
            args.phase1_checkpoint = paths_cfg.get("phase1_checkpoint", "models/checkpoints/phase1_model.pth")
    else:
        checkpoint_path = paths_cfg.get("checkpoint", "models/checkpoints/best_model.pth")
        scaler_path = paths_cfg.get("scaler", "models/artifacts/scaler.pkl")
        cache_dir = Path(paths_cfg.get("cache_dir", "models/artifacts/cache"))
        split_path = data_cfg.get("split_path", "models/artifacts/split.json")
        lr = float(train_cfg.get("learning_rate", 1e-3))
        max_epochs = args.max_epochs if args.max_epochs is not None else int(train_cfg.get("max_epochs", 100))
        patience = 15

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
                "Cache missing or incomplete. Required: %s. Missing: %s. Run: python -m src.training.precompute_cache",
                required,
                missing,
            )
            sys.exit(1)
        train_ds = PrecomputedDataset(cache_dir, "train", is_train=True, config_path=args.config)
        try:
            val_ds = PrecomputedDataset(cache_dir, "val", is_train=False, config_path=args.config)
        except FileNotFoundError:
            # Legacy cache without val files (precompute_cache now writes empty splits)
            import json
            with open(cache_dir / "cache_meta.json") as f:
                meta = json.load(f)
            max_t = meta.get("max_short_len", 3000)
            val_ds = _EmptyCacheDataset(max_t)
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        scaler = None
    else:
        train_loader, val_loader, test_loader = get_dataloaders(
            config_path=args.config,
            split_path=split_path,
            batch_size=batch_size,
            create_split_if_missing=True,
        )
        train_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=_pad_collate,
        )
        val_loader = torch.utils.data.DataLoader(
            val_loader.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=_pad_collate,
        )
        logger.info("Fitting scaler on training set...")
        _fit_scaler_from_train_dataset(train_loader.dataset, args.config, scaler_path)
        scaler = load_scaler(path=scaler_path, config_path=args.config)

    # Compute pos_weight from training label distribution to handle class imbalance
    all_train_labels = []
    for batch in train_loader:
        labels_batch = batch[2]
        all_train_labels.append(labels_batch)
    all_train_labels = torch.cat(all_train_labels)
    n_pos = (all_train_labels == 1).sum().float()
    n_neg = (all_train_labels == 0).sum().float()
    if n_pos > 0:
        pos_weight = (n_neg / n_pos).clamp(max=10.0)  # cap at 10 to avoid instability
    else:
        pos_weight = torch.tensor(1.0)
    pos_weight = pos_weight.to(device)
    logger.info("Class balance: %d positive, %d negative, pos_weight=%.2f", int(n_pos), int(n_neg), pos_weight.item())

    # Build model — Phase 2 loads Phase 1 checkpoint first
    if phase == 2 and args.phase1_checkpoint:
        logger.info("Phase 2: loading Phase 1 checkpoint from %s", args.phase1_checkpoint)
        model = build_model(config_path=args.config, checkpoint_path=args.phase1_checkpoint, device=device)
        # Freeze backbone (CNN, GRU, Transformer) — only head is trainable
        for param in model.cnn.parameters():
            param.requires_grad = False
        for param in model.rnn.parameters():
            param.requires_grad = False
        for param in model.transformer.parameters():
            param.requires_grad = False
        # Keep backbone in eval mode (BatchNorm stays frozen)
        model.cnn.eval()
        model.rnn.eval()
        model.transformer.eval()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info("Phase 2: frozen backbone. Trainable: %d / %d params (head only)", trainable, total)
        train_params = model.head.parameters()
    else:
        model = build_model(config_path=args.config, checkpoint_path=None, device=device)
        train_params = model.parameters()

    optimizer = torch.optim.AdamW(train_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5, min_lr=1e-6
    )
    best_auroc = 0.0
    epochs_without_improvement = 0

    log_path = Path(scaler_path).parent / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "val_auroc"])

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
            torch.save({"state_dict": model.state_dict(), "epoch": epoch + 1, "val_auroc": val_auroc}, checkpoint_path)
            logger.info("Saved best checkpoint (val_auroc=%.4f) to %s", val_auroc, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch + 1, patience)
                break

    logger.info("Training finished. Best val_auroc=%.4f", best_auroc)


if __name__ == "__main__":
    main()
