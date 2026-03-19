"""
Tests for StrokeResponderHead and StrokeHybridEnsemble (Steps 10+11).
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest

from src.models.stroke_head import StrokeResponderHead
from src.models.stroke_ensemble import StrokeHybridEnsemble, build_stroke_model
from src.models.ensemble import EnsembleConfig
from src.models.cnn import CNNConfig
from src.models.rnn import RNNConfig
from src.models.transformer import TransformerConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH = 4
FUSED_DIM = 256   # 128 + 64 + 64
T = 2500          # 10s @ 250 Hz
SEQ_LEN = 5
N_FEATURES = 7


def _default_cfg() -> EnsembleConfig:
    return EnsembleConfig(
        cnn=CNNConfig(in_channels=1, embed_dim=128),
        rnn=RNNConfig(input_size=7, hidden_size=64, num_layers=1),
        transformer=TransformerConfig(n_features=7, d_model=64, nhead=4,
                                      num_encoder_layers=1, dim_feedforward=128,
                                      seq_len=SEQ_LEN),
        dropout=0.0,  # deterministic for testing
        head_hidden_dim=0,
    )


def _waveform():
    return torch.randn(BATCH, 1, T)


def _hrv():
    return torch.randn(BATCH, SEQ_LEN, N_FEATURES)


# ---------------------------------------------------------------------------
# StrokeResponderHead tests
# ---------------------------------------------------------------------------

def test_stroke_head_output_shape():
    """(B, fused_dim) → (B, 1)."""
    head = StrokeResponderHead(fused_dim=FUSED_DIM)
    x = torch.randn(BATCH, FUSED_DIM)
    out = head(x)
    assert out.shape == (BATCH, 1), f"Expected ({BATCH}, 1), got {out.shape}"


def test_stroke_head_expanded():
    """head_hidden_dim=64 also produces (B, 1)."""
    head = StrokeResponderHead(fused_dim=FUSED_DIM, head_hidden_dim=64)
    x = torch.randn(BATCH, FUSED_DIM)
    out = head(x)
    assert out.shape == (BATCH, 1)


def test_stroke_head_output_dtype():
    """Output is float32."""
    head = StrokeResponderHead(fused_dim=FUSED_DIM)
    out = head(torch.randn(BATCH, FUSED_DIM))
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# StrokeHybridEnsemble tests
# ---------------------------------------------------------------------------

def test_stroke_ensemble_forward_shape():
    """(B,1,T) + (B,5,7) → (B,1) logits."""
    model = StrokeHybridEnsemble(_default_cfg())
    model.eval()
    with torch.no_grad():
        out = model(_waveform(), _hrv())
    assert out.shape == (BATCH, 1), f"Expected ({BATCH}, 1), got {out.shape}"


def test_stroke_ensemble_lengths_optional():
    """forward succeeds with and without hrv_lengths."""
    model = StrokeHybridEnsemble(_default_cfg())
    model.eval()
    wav, hrv = _waveform(), _hrv()
    with torch.no_grad():
        out_no_len = model(wav, hrv)
        lengths = torch.tensor([5, 5, 4, 3])
        out_with_len = model(wav, hrv, hrv_lengths=lengths)
    assert out_no_len.shape == (BATCH, 1)
    assert out_with_len.shape == (BATCH, 1)


def test_stroke_ensemble_freeze_backbone():
    """After freeze_backbone(), encoder params are frozen; head params still trainable."""
    model = StrokeHybridEnsemble(_default_cfg())
    model.freeze_backbone()

    for name, param in model.named_parameters():
        if name.startswith("head."):
            assert param.requires_grad, f"{name} should still be trainable"
        else:
            assert not param.requires_grad, f"{name} should be frozen"


def test_stroke_ensemble_unfreeze_backbone():
    """After unfreeze_backbone(), all params are trainable."""
    model = StrokeHybridEnsemble(_default_cfg())
    model.freeze_backbone()
    model.unfreeze_backbone()

    for name, param in model.named_parameters():
        assert param.requires_grad, f"{name} should be trainable after unfreeze"


def test_stroke_ensemble_backward():
    """loss.backward() completes without error."""
    model = StrokeHybridEnsemble(_default_cfg())
    model.train()
    out = model(_waveform(), _hrv())
    labels = torch.zeros(BATCH, 1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out, labels)
    loss.backward()  # must not raise


def test_stroke_ensemble_freeze_then_backward():
    """With frozen backbone, only head grads are computed."""
    model = StrokeHybridEnsemble(_default_cfg())
    model.freeze_backbone()
    model.train()
    out = model(_waveform(), _hrv())
    labels = torch.zeros(BATCH, 1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out, labels)
    loss.backward()

    for name, param in model.named_parameters():
        if name.startswith("head.") and param.requires_grad:
            assert param.grad is not None, f"Head param {name} should have grad"
        elif not param.requires_grad:
            assert param.grad is None, f"Frozen param {name} should have no grad"


# ---------------------------------------------------------------------------
# build_stroke_model tests
# ---------------------------------------------------------------------------

def test_build_stroke_model_from_config():
    """build_stroke_model returns StrokeHybridEnsemble with correct output shape."""
    model = build_stroke_model("config_stroke.yaml")
    assert isinstance(model, StrokeHybridEnsemble)
    model.eval()
    with torch.no_grad():
        out = model(_waveform(), _hrv())
    assert out.shape == (BATCH, 1)
