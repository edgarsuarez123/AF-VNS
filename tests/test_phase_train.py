"""Tests for phase_train.py — PhaseDetector training loop (S-21)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_phase_cache(tmp_dir: Path, n_train: int = 32, n_val: int = 16):
    """Write minimal phase detection cache files (mirrors stroke_precompute_cache output)."""
    rng = np.random.default_rng(42)

    for split, n in [("train", n_train), ("val", n_val)]:
        ecg = rng.standard_normal((n, 500)).astype(np.float32)

        # Binary labels with known NaN positions for masking tests
        dia = rng.integers(0, 2, size=(n, 10)).astype(np.float32)
        dia[:, 0] = np.nan  # frame 0 always NaN
        dia[:, 5] = np.nan  # frame 5 always NaN

        exh = rng.integers(0, 2, size=(n, 10)).astype(np.float32)
        exh[:, 3] = np.nan  # frame 3 always NaN
        exh[:, 7] = np.nan  # frame 7 always NaN

        qual = rng.uniform(0.5, 1.0, (n, 10)).astype(np.float32)

        np.save(tmp_dir / f"{split}_ecg.npy", ecg)
        np.save(tmp_dir / f"{split}_diastole.npy", dia)
        np.save(tmp_dir / f"{split}_exhalation.npy", exh)
        np.save(tmp_dir / f"{split}_quality.npy", qual)

    meta = {
        "window_samples": 500,
        "frames_per_window": 10,
        "window_sec": 2.0,
        "stride_sec": 0.2,
        "frame_rate_hz": 5.0,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": 0,
    }
    with open(tmp_dir / "phase_cache_meta.json", "w") as f:
        json.dump(meta, f)


def _make_phase_config(tmp_dir: Path, cache_dir: Path, ckpt_path: str) -> Path:
    """Write minimal config for phase detector training tests."""
    cfg = f"""
paths:
  phase_detect_checkpoint: {ckpt_path}

phase_precompute:
  cache_dir: {cache_dir}
  window_sec: 2.0
  stride_sec: 0.2
  min_valid_frames: 8
  frame_rate_hz: 5.0

phase_model:
  channels: [8, 16, 24]
  kernels: [7, 5, 3]
  strides: [5, 2, 2]
  dropout: 0.0
  n_frames: 10
  n_tasks: 2

phase_training:
  learning_rate: 1.0e-3
  batch_size: 8
  max_epochs: 1
  patience: 5
  grad_clip: 1.0

data:
  target_fs: 250

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
    config_path = tmp_dir / "config_phase_test.yaml"
    config_path.write_text(cfg)
    return config_path


# ---------------------------------------------------------------------------
# TestPhaseDetectorDataset
# ---------------------------------------------------------------------------

class TestPhaseDetectorDataset:

    def test_dataset_length(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        _make_phase_cache(cache_dir, n_train=32)

        from src.training.phase_train import PhaseDetectorDataset
        ds = PhaseDetectorDataset(cache_dir, "train")
        assert len(ds) == 32

    def test_dataset_shapes(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        _make_phase_cache(cache_dir)

        from src.training.phase_train import PhaseDetectorDataset
        ds = PhaseDetectorDataset(cache_dir, "train")
        ecg, dia, exh, qual = ds[0]
        assert ecg.shape == (1, 500)
        assert dia.shape == (10,)
        assert exh.shape == (10,)
        assert qual.shape == (10,)

    def test_dataset_nan_preserved(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        _make_phase_cache(cache_dir)

        from src.training.phase_train import PhaseDetectorDataset
        ds = PhaseDetectorDataset(cache_dir, "train")
        _, dia, exh, _ = ds[0]
        # Diastole: frames 0, 5 are NaN
        assert torch.isnan(dia[0])
        assert torch.isnan(dia[5])
        assert not torch.isnan(dia[1])
        # Exhalation: frames 3, 7 are NaN
        assert torch.isnan(exh[3])
        assert torch.isnan(exh[7])
        assert not torch.isnan(exh[1])


# ---------------------------------------------------------------------------
# TestMultitaskBCELoss
# ---------------------------------------------------------------------------

class TestMultitaskBCELoss:

    def test_loss_nan_masking(self):
        from src.training.phase_train import multitask_bce_loss

        logits = torch.zeros(2, 10, 2)
        dia = torch.ones(2, 10)
        dia[:, 0] = float("nan")
        exh = torch.zeros(2, 10)
        dia_pw = torch.tensor(1.0)
        exh_pw = torch.tensor(1.0)

        loss = multitask_bce_loss(logits, dia, exh, dia_pw, exh_pw)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_loss_all_nan(self):
        from src.training.phase_train import multitask_bce_loss

        logits = torch.zeros(2, 10, 2)
        dia = torch.full((2, 10), float("nan"))
        exh = torch.full((2, 10), float("nan"))
        dia_pw = torch.tensor(1.0)
        exh_pw = torch.tensor(1.0)

        loss = multitask_bce_loss(logits, dia, exh, dia_pw, exh_pw)
        assert loss.item() == pytest.approx(0.0)

    def test_loss_backward(self):
        from src.training.phase_train import multitask_bce_loss

        logits = torch.randn(4, 10, 2, requires_grad=True)
        dia = torch.ones(4, 10)
        dia[:, 0] = float("nan")
        exh = torch.zeros(4, 10)
        dia_pw = torch.tensor(1.0)
        exh_pw = torch.tensor(1.0)

        loss = multitask_bce_loss(logits, dia, exh, dia_pw, exh_pw)
        loss.backward()
        assert logits.grad is not None


# ---------------------------------------------------------------------------
# TestPosWeight
# ---------------------------------------------------------------------------

class TestPosWeight:

    def test_pos_weight_per_task(self, tmp_path):
        """Diastole ~62% positive → lower weight, exhalation ~30% positive → higher weight."""
        from src.training.phase_train import PhaseDetectorDataset, compute_phase_pos_weights

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        n = 100
        rng = np.random.default_rng(99)
        ecg = rng.standard_normal((n, 500)).astype(np.float32)
        # Diastole: 80% positive
        dia = np.ones((n, 10), dtype=np.float32)
        dia[:20, :] = 0.0
        # Exhalation: 20% positive
        exh = np.zeros((n, 10), dtype=np.float32)
        exh[:20, :] = 1.0
        qual = np.ones((n, 10), dtype=np.float32)

        np.save(cache_dir / "train_ecg.npy", ecg)
        np.save(cache_dir / "train_diastole.npy", dia)
        np.save(cache_dir / "train_exhalation.npy", exh)
        np.save(cache_dir / "train_quality.npy", qual)

        ds = PhaseDetectorDataset(cache_dir, "train")
        dia_pw, exh_pw = compute_phase_pos_weights(ds, torch.device("cpu"))
        assert dia_pw.item() < exh_pw.item(), (
            f"Diastole pw ({dia_pw.item()}) should be < exhalation pw ({exh_pw.item()})"
        )

    def test_pos_weight_clamp(self, tmp_path):
        """Extreme imbalance clamped at 10.0."""
        from src.training.phase_train import PhaseDetectorDataset, compute_phase_pos_weights

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        n = 100
        ecg = np.zeros((n, 500), dtype=np.float32)
        # Only 1 positive frame across all samples → weight = 999/1 → clamped to 10
        dia = np.zeros((n, 10), dtype=np.float32)
        dia[0, 0] = 1.0
        exh = np.zeros((n, 10), dtype=np.float32)
        exh[0, 0] = 1.0
        qual = np.ones((n, 10), dtype=np.float32)

        np.save(cache_dir / "train_ecg.npy", ecg)
        np.save(cache_dir / "train_diastole.npy", dia)
        np.save(cache_dir / "train_exhalation.npy", exh)
        np.save(cache_dir / "train_quality.npy", qual)

        ds = PhaseDetectorDataset(cache_dir, "train")
        dia_pw, exh_pw = compute_phase_pos_weights(ds, torch.device("cpu"))
        assert dia_pw.item() == pytest.approx(10.0)
        assert exh_pw.item() == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# TestPhaseValidation
# ---------------------------------------------------------------------------

class TestPhaseValidation:

    def test_accuracy_perfect(self):
        """Large positive logits + all-1 targets → 100% accuracy."""
        from src.training.phase_train import run_phase_validation, PhaseDetectorDataset

        # Build a simple dataset in memory via mock
        class _PerfectDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 8
            def __getitem__(self, i):
                ecg = torch.randn(1, 500)
                dia = torch.ones(10)   # all diastole
                exh = torch.ones(10)   # all exhale
                qual = torch.ones(10)
                return ecg, dia, exh, qual

        from src.models.phase_detector import PhaseDetectorConfig, PhaseDetector

        # Model that outputs large positive logits (sigmoid → ~1.0)
        cfg = PhaseDetectorConfig(channels=(8, 16, 24), dropout=0.0)
        model = PhaseDetector(cfg)

        # Override forward to return constant large positive logits
        class _ConstantModel(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
            def forward(self, x):
                B = x.shape[0]
                return torch.full((B, 10, 2), 10.0)  # sigmoid(10) ≈ 1.0

        mock_model = _ConstantModel(model)
        loader = torch.utils.data.DataLoader(_PerfectDataset(), batch_size=4)
        dia_pw = torch.tensor(1.0)
        exh_pw = torch.tensor(1.0)

        _, dia_acc, exh_acc, avg_acc = run_phase_validation(
            mock_model, loader, torch.device("cpu"), dia_pw, exh_pw)
        assert dia_acc == pytest.approx(1.0)
        assert exh_acc == pytest.approx(1.0)
        assert avg_acc == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestPhaseTrainSmoke
# ---------------------------------------------------------------------------

class TestPhaseTrainSmoke:

    def _run_one_epoch(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        _make_phase_cache(cache_dir, n_train=32, n_val=16)

        ckpt_path = str(tmp_path / "phase_detector.pth")
        config_path = _make_phase_config(tmp_path, cache_dir, ckpt_path)

        from src.training.phase_train import main
        import sys

        argv_backup = sys.argv
        sys.argv = [
            "phase_train",
            "--config", str(config_path),
            "--max-epochs", "1",
        ]
        try:
            main()
        finally:
            sys.argv = argv_backup

        return ckpt_path, config_path, cache_dir

    def test_one_epoch_runs(self, tmp_path):
        """One epoch with synthetic cache completes without error."""
        self._run_one_epoch(tmp_path)

    def test_checkpoint_saved(self, tmp_path):
        """Checkpoint file exists after training."""
        ckpt_path, _, _ = self._run_one_epoch(tmp_path)
        assert Path(ckpt_path).exists(), "Checkpoint file not created"

    def test_checkpoint_keys(self, tmp_path):
        """Checkpoint has required keys."""
        ckpt_path, _, _ = self._run_one_epoch(tmp_path)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "state_dict" in state
        assert "epoch" in state
        assert "val_dia_acc" in state
        assert "val_exh_acc" in state
        assert "val_avg_acc" in state


# ---------------------------------------------------------------------------
# F8: PPG PhaseDetector CNN config verification
# ---------------------------------------------------------------------------

class TestPhaseDetectorPPGConfig:
    """Verify PhaseDetector works with PPG input shape (input_samples=250).

    F8 uses existing PhaseDetector unchanged — only input_samples differs from
    the ECG pipeline (250 = 2s @ 125 Hz PPG vs 500 = 2s @ 250 Hz ECG).
    """

    def test_forward_pass_250_samples(self):
        """(4, 1, 250) input → (4, 10, 2) output — tinnitus PPG PhaseDetector."""
        from src.models.phase_detector import PhaseDetectorConfig, PhaseDetector

        cfg = PhaseDetectorConfig(input_samples=250)
        model = PhaseDetector(cfg)
        model.eval()

        x = torch.randn(4, 1, 250)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (4, 10, 2), (
            f"Expected (4, 10, 2), got {tuple(out.shape)}"
        )

    def test_config_tinnitus_yaml_input_samples(self):
        """config_tinnitus.yaml phase_model.input_samples == 250."""
        import yaml, os
        root = Path(__file__).resolve().parents[1]
        cfg_path = root / "config_tinnitus.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["phase_model"]["input_samples"] == 250, (
            f"Expected input_samples=250, got {cfg['phase_model']['input_samples']}"
        )

    def test_build_phase_detector_from_tinnitus_config(self):
        """build_phase_detector reads config_tinnitus.yaml and produces correct shape."""
        import os
        from src.models.phase_detector import build_phase_detector

        root = Path(__file__).resolve().parents[1]
        cfg_path = str(root / "config_tinnitus.yaml")
        model = build_phase_detector(cfg_path, model_section="phase_model")
        model.eval()

        x = torch.randn(2, 1, 250)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (2, 10, 2), (
            f"Expected (2, 10, 2), got {tuple(out.shape)}"
        )
