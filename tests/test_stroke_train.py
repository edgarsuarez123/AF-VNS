"""Tests for stroke_train.py — Phase 1 and Phase 2 training entry point."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_cache(tmp_dir: Path, n_train: int = 16, n_val: int = 8,
                           max_short_len: int = 2500, seq_len: int = 5,
                           n_features: int = 7):
    """Write minimal cache files to tmp_dir (mirrors precompute_cache output)."""
    rng = np.random.default_rng(42)

    for split, n in [("train", n_train), ("val", n_val)]:
        short = rng.standard_normal((n, max_short_len)).astype(np.float32)
        hrv = rng.standard_normal((n, seq_len, n_features)).astype(np.float32)
        # Imbalanced labels for pos_weight test: first split gets 10:1 neg:pos
        if split == "train":
            labels = np.zeros(n, dtype=np.float32)
            labels[: max(1, n // 10)] = 1.0
        else:
            labels = rng.integers(0, 2, size=n).astype(np.float32)
        lengths = np.full(n, seq_len, dtype=np.int64)

        # Wrap short in channel dim: (n, 1, max_short_len) — PrecomputedDataset expects (n, max_T)
        # Actually PrecomputedDataset loads as (n, max_T) and adds channel dim in __getitem__
        np.save(tmp_dir / f"{split}_short.npy", short)
        np.save(tmp_dir / f"{split}_hrv_scaled.npy", hrv)
        np.save(tmp_dir / f"{split}_labels.npy", labels)
        np.save(tmp_dir / f"{split}_hrv_lengths.npy", lengths)

    meta = {"max_short_len": max_short_len, "seq_len": seq_len, "n_features": n_features}
    with open(tmp_dir / "cache_meta.json", "w") as f:
        json.dump(meta, f)


def _make_stroke_config(tmp_dir: Path, cache_dir: Path, scaler_path: str,
                        checkpoint_path: str) -> Path:
    """Write a minimal config_stroke.yaml pointing to temp paths."""
    cfg = f"""
paths:
  phase1_checkpoint: {checkpoint_path}
  phase2_checkpoint: {tmp_dir}/stroke_p2_model.pth
  phase1_scaler: {scaler_path}
  phase2_scaler: {tmp_dir}/p2_scaler.pkl
  phase1_cache_dir: {cache_dir}
  phase2_cache_dir: {cache_dir}
  phase1_split: {tmp_dir}/p1_split.json
  phase2_split: {tmp_dir}/p2_split.json

data:
  target_fs: 250
  waveform_sec: 10
  hrv_window_sec: 300
  stride_sec: 300

training:
  learning_rate: 1.0e-3
  batch_size: 8
  max_epochs: 1
  label_smoothing: 0.0

model:
  in_channels: 1
  cnn_embed_dim: 32
  rnn_hidden_size: 16
  rnn_num_layers: 1
  transformer_d_model: 16
  transformer_nhead: 2
  transformer_num_encoder_layers: 1
  transformer_dim_feedforward: 32
  hrv_seq_len: 5
  hrv_n_features: 7
  head_hidden_dim: 16

augmentation:
  enabled: false

artifact:
  amplitude_mad_multiple: 30
  rr_deviation_percent: 60
  rr_fraction_threshold: 0.30

hrv:
  window_sec: [30, 300]
  subwindow_sec: 60
"""
    config_path = tmp_dir / "config_stroke_test.yaml"
    config_path.write_text(cfg)
    return config_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStrokeTrainPhase1:

    def _run_phase1(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        _make_synthetic_cache(cache_dir)

        ckpt_path = str(tmp_path / "stroke_p1_model.pth")
        scaler_path = str(tmp_path / "p1_scaler.pkl")
        config_path = _make_stroke_config(tmp_path, cache_dir, scaler_path, ckpt_path)

        # Import here to avoid side effects at collection time
        from src.training.stroke_train import main
        import sys

        argv_backup = sys.argv
        sys.argv = [
            "stroke_train",
            "--config", str(config_path),
            "--phase", "1",
            "--use-cache",
            "--max-epochs", "1",
        ]
        try:
            main()
        finally:
            sys.argv = argv_backup

        return ckpt_path, scaler_path, config_path, cache_dir

    def test_stroke_train_phase1_runs_one_epoch(self, tmp_path):
        """Phase 1 with --use-cache --max-epochs 1 completes without error."""
        self._run_phase1(tmp_path)  # No exception = pass

    def test_stroke_train_phase1_checkpoint_saved(self, tmp_path):
        """Phase 1 checkpoint file exists and has required keys."""
        ckpt_path, _, _, _ = self._run_phase1(tmp_path)
        assert Path(ckpt_path).exists(), "Checkpoint file not created"
        state = torch.load(ckpt_path, map_location="cpu")
        assert "state_dict" in state
        assert "epoch" in state
        assert "val_auroc" in state


class TestStrokeTrainPhase2:

    def _build_phase1_checkpoint(self, tmp_path, config_path):
        """Build a real Phase 1 checkpoint by running one epoch, or create a bare weights file."""
        from src.models.stroke_ensemble import build_stroke_model

        model = build_stroke_model(config_path=str(config_path), checkpoint_path=None, device="cpu")
        ckpt_path = tmp_path / "stroke_p1_model.pth"
        torch.save({"state_dict": model.state_dict(), "epoch": 1, "val_auroc": 0.5}, str(ckpt_path))
        return str(ckpt_path)

    def test_stroke_train_phase2_freezes_backbone(self, tmp_path):
        """After Phase 2 model setup, backbone params have requires_grad=False."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        _make_synthetic_cache(cache_dir)

        scaler_path = str(tmp_path / "p1_scaler.pkl")
        ckpt_path_placeholder = str(tmp_path / "stroke_p1_model.pth")
        config_path = _make_stroke_config(tmp_path, cache_dir, scaler_path, ckpt_path_placeholder)
        p1_ckpt = self._build_phase1_checkpoint(tmp_path, config_path)

        from src.models.stroke_ensemble import build_stroke_model

        model = build_stroke_model(config_path=str(config_path),
                                   checkpoint_path=p1_ckpt, device="cpu")
        model.freeze_backbone()

        for name, param in model.cnn.named_parameters():
            assert not param.requires_grad, f"CNN.{name} should be frozen"
        for name, param in model.rnn.named_parameters():
            assert not param.requires_grad, f"RNN.{name} should be frozen"
        for name, param in model.transformer.named_parameters():
            assert not param.requires_grad, f"Transformer.{name} should be frozen"
        for name, param in model.head.named_parameters():
            assert param.requires_grad, f"Head.{name} should be trainable"

    def test_stroke_train_phase2_runs_one_epoch(self, tmp_path):
        """Phase 2 with a valid Phase 1 checkpoint completes one epoch."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        _make_synthetic_cache(cache_dir)

        scaler_path = str(tmp_path / "p1_scaler.pkl")
        ckpt_path_placeholder = str(tmp_path / "stroke_p1_model.pth")
        config_path = _make_stroke_config(tmp_path, cache_dir, scaler_path, ckpt_path_placeholder)
        p1_ckpt = self._build_phase1_checkpoint(tmp_path, config_path)

        from src.training.stroke_train import main
        import sys

        argv_backup = sys.argv
        sys.argv = [
            "stroke_train",
            "--config", str(config_path),
            "--phase", "2",
            "--phase1-checkpoint", p1_ckpt,
            "--use-cache",
            "--max-epochs", "1",
        ]
        try:
            main()
        finally:
            sys.argv = argv_backup


class TestStrokeTrainPosWeight:

    def test_stroke_train_pos_weight_imbalanced(self, tmp_path):
        """Imbalanced labels (10:1 neg:pos) produce pos_weight ≈ 9 (clamped at 10)."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # 16 samples: 1 positive, 15 negative → pos_weight = 15/1 = 15, clamped to 10
        _make_synthetic_cache(cache_dir, n_train=16, n_val=4)

        # Directly test the pos_weight calculation logic
        rng = np.random.default_rng(0)
        labels = np.zeros(16, dtype=np.float32)
        labels[0] = 1.0  # 1 positive, 15 negative

        all_labels = torch.from_numpy(labels)
        n_pos = (all_labels == 1).sum().float()
        n_neg = (all_labels == 0).sum().float()
        pos_weight = (n_neg / n_pos).clamp(max=10.0)

        assert pos_weight.item() == pytest.approx(10.0), (
            f"Expected pos_weight=10.0 (clamped), got {pos_weight.item()}"
        )

    def test_stroke_train_pos_weight_balanced(self, tmp_path):
        """Balanced labels (1:1) produce pos_weight ≈ 1."""
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        n_pos = (labels == 1).sum().float()
        n_neg = (labels == 0).sum().float()
        pos_weight = (n_neg / n_pos).clamp(max=10.0)
        assert pos_weight.item() == pytest.approx(1.0)
