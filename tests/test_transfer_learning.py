"""Tests for two-phase transfer learning: phase filtering, backbone freezing, expanded head."""

import re
import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.precompute_cache import _should_include
from src.models.ensemble import EnsembleConfig, HybridEnsemble
from src.models.cnn import CNNConfig
from src.models.rnn import RNNConfig
from src.models.transformer import TransformerConfig


# ---------------------------------------------------------------------------
# Tests: _should_include phase filtering
# ---------------------------------------------------------------------------

class TestPhaseFiltering:

    def test_phase1_includes_only_mimic(self):
        """Phase 1 includes only MIMIC IDs (p######_*)."""
        assert _should_include("p000302_3967145_0062", 1) is True
        assert _should_include("p099797_test", 1) is True
        assert _should_include("af003", 1) is False
        assert _should_include("nsr042", 1) is False
        assert _should_include("ltaf_seg001", 1) is False
        assert _should_include("c17_A00001", 1) is False

    def test_phase2_excludes_mimic_and_c17(self):
        """Phase 2 excludes MIMIC and Challenge 2017; includes AFDB/NSRDB/LTAFDB."""
        assert _should_include("af003", 2) is True
        assert _should_include("nsr042", 2) is True
        assert _should_include("ltaf_seg001", 2) is True
        assert _should_include("p000302_3967145_0062", 2) is False
        assert _should_include("c17_A00001", 2) is False

    def test_phase0_includes_all(self):
        """Phase 0 (legacy) includes everything."""
        assert _should_include("p000302_test", 0) is True
        assert _should_include("af003", 0) is True
        assert _should_include("c17_A00001", 0) is True
        assert _should_include("ltaf_seg001", 0) is True


# ---------------------------------------------------------------------------
# Tests: expanded head
# ---------------------------------------------------------------------------

class TestExpandedHead:

    def _make_config(self, head_hidden_dim=0):
        return EnsembleConfig(
            cnn=CNNConfig(in_channels=1, embed_dim=128),
            rnn=RNNConfig(input_size=7, hidden_size=64),
            transformer=TransformerConfig(n_features=7, d_model=64, nhead=4,
                                          num_encoder_layers=1, dim_feedforward=128,
                                          seq_len=5),
            dropout=0.2,
            head_hidden_dim=head_hidden_dim,
        )

    def test_legacy_head_single_linear(self):
        """head_hidden_dim=0 gives single Linear(256,1) head."""
        cfg = self._make_config(head_hidden_dim=0)
        model = HybridEnsemble(cfg)
        # Should have 2 layers: Dropout, Linear
        assert len(model.head) == 2

    def test_expanded_head_dimensions(self):
        """head_hidden_dim=64 gives expanded head with 5 layers."""
        cfg = self._make_config(head_hidden_dim=64)
        model = HybridEnsemble(cfg)
        # Should have 5 layers: Dropout, Linear(256,64), ReLU, Dropout, Linear(64,1)
        assert len(model.head) == 5
        assert isinstance(model.head[1], torch.nn.Linear)
        assert model.head[1].in_features == 256
        assert model.head[1].out_features == 64
        assert isinstance(model.head[2], torch.nn.ReLU)
        assert isinstance(model.head[4], torch.nn.Linear)
        assert model.head[4].in_features == 64
        assert model.head[4].out_features == 1

    def test_expanded_head_param_count(self):
        """Expanded head should have ~16.5K trainable params."""
        cfg = self._make_config(head_hidden_dim=64)
        model = HybridEnsemble(cfg)
        head_params = sum(p.numel() for p in model.head.parameters())
        # Linear(256,64): 256*64+64=16448, Linear(64,1): 64+1=65 -> 16513
        assert 16000 < head_params < 17000


# ---------------------------------------------------------------------------
# Tests: backbone freezing
# ---------------------------------------------------------------------------

class TestBackboneFreezing:

    def _make_model(self):
        cfg = EnsembleConfig(
            cnn=CNNConfig(in_channels=1, embed_dim=128),
            rnn=RNNConfig(input_size=7, hidden_size=64),
            transformer=TransformerConfig(n_features=7, d_model=64, nhead=4,
                                          num_encoder_layers=1, dim_feedforward=128,
                                          seq_len=5),
            dropout=0.2,
            head_hidden_dim=64,
        )
        return HybridEnsemble(cfg)

    def test_freeze_backbone_only_head_trainable(self):
        """After freezing CNN/GRU/Transformer, only head params have requires_grad=True."""
        model = self._make_model()

        for param in model.cnn.parameters():
            param.requires_grad = False
        for param in model.rnn.parameters():
            param.requires_grad = False
        for param in model.transformer.parameters():
            param.requires_grad = False

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        head_params = sum(p.numel() for p in model.head.parameters())
        assert trainable == head_params

    def test_phase2_forward_gradients_only_in_head(self):
        """Forward+backward pass with frozen backbone: gradients only in head."""
        model = self._make_model()

        for param in model.cnn.parameters():
            param.requires_grad = False
        for param in model.rnn.parameters():
            param.requires_grad = False
        for param in model.transformer.parameters():
            param.requires_grad = False

        # Forward pass
        waveform = torch.randn(2, 1, 2500)
        hrv = torch.randn(2, 5, 7)
        logits = model(waveform, hrv)
        loss = logits.sum()
        loss.backward()

        # Check: backbone params have no grad, head params have grad
        for name, param in model.cnn.named_parameters():
            assert param.grad is None, f"CNN param {name} has gradient but should be frozen"
        for name, param in model.rnn.named_parameters():
            assert param.grad is None, f"RNN param {name} has gradient but should be frozen"
        for name, param in model.transformer.named_parameters():
            assert param.grad is None, f"Transformer param {name} has gradient but should be frozen"
        for name, param in model.head.named_parameters():
            assert param.grad is not None, f"Head param {name} has no gradient but should be trainable"
