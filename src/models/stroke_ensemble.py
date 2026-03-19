"""
StrokeHybridEnsemble — backbone + pluggable head for stroke AVNS.

Reuses CNNEncoder, GRUEncoder, TransformerEncoder from the AF pipeline unchanged.
Identical forward interface to HybridEnsemble so the training loop is drop-in compatible.

Phase 1: all parameters trainable (MIMIC-3 stroke pre-training)
Phase 2: freeze_backbone() → only head trains (CereVasc fine-tuning)
AF init:  build_stroke_model(checkpoint_path=af_ckpt) loads encoder weights via strict=False
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from .cnn import CNNEncoder
from .ensemble import EnsembleConfig
from .rnn import GRUEncoder
from .stroke_head import StrokeResponderHead
from .transformer import TransformerEncoder


class StrokeHybridEnsemble(nn.Module):
    """Hybrid CNN + GRU + Transformer model with a detachable StrokeResponderHead.

    Args:
        cfg: EnsembleConfig — reuses the same config dataclass as HybridEnsemble.
    """

    def __init__(self, cfg: EnsembleConfig):
        super().__init__()
        self.cfg = cfg

        self.cnn = CNNEncoder(cfg.cnn)
        self.rnn = GRUEncoder(cfg.rnn)
        self.transformer = TransformerEncoder(cfg.transformer)

        fused_dim = cfg.cnn.embed_dim + cfg.rnn.hidden_size + cfg.transformer.d_model
        self.head = StrokeResponderHead(fused_dim, cfg.head_hidden_dim, cfg.dropout)

    def forward(
        self,
        waveform_10s: torch.Tensor,
        hrv_sequence: torch.Tensor,
        hrv_lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        waveform_10s: (B, 1, T)
        hrv_sequence: (B, seq_len, 7) — scaled HRV features
        hrv_lengths:  (B,) int tensor of valid HRV timesteps (optional)
        returns:      (B, 1) logits for BCEWithLogitsLoss
        """
        cnn_emb = self.cnn(waveform_10s)
        rnn_emb = self.rnn(hrv_sequence, lengths=hrv_lengths)
        tr_emb = self.transformer(hrv_sequence, lengths=hrv_lengths)
        fused = torch.cat([cnn_emb, rnn_emb, tr_emb], dim=1)
        return self.head(fused)

    def freeze_backbone(self) -> None:
        """Freeze CNN, RNN, Transformer — only head remains trainable (Phase 2)."""
        for module in (self.cnn, self.rnn, self.transformer):
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all encoder parameters."""
        for module in (self.cnn, self.rnn, self.transformer):
            for param in module.parameters():
                param.requires_grad = True


def build_stroke_model(
    config_path: str = "config_stroke.yaml",
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
) -> StrokeHybridEnsemble:
    """Build StrokeHybridEnsemble from config; optionally load checkpoint.

    Passing an AF checkpoint (phase2_model.pth) loads encoder weights via strict=False —
    encoder key names (cnn.*, rnn.*, transformer.*) match; head keys differ and are skipped.

    Args:
        config_path:     Path to config_stroke.yaml (or any compatible config).
        checkpoint_path: Path to .pth checkpoint (AF or stroke Phase 1). Optional.
        device:          If provided, move model to this device before returning.

    Returns:
        StrokeHybridEnsemble
    """
    # Reuse build_ensemble_config — reads same model.* keys present in config_stroke.yaml
    from ..training.build_model import build_ensemble_config

    cfg = build_ensemble_config(config_path)
    model = StrokeHybridEnsemble(cfg)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            import logging
            logging.getLogger(__name__).info(
                "build_stroke_model: %d missing keys (expected for new head): %s...",
                len(missing), missing[:3],
            )

    if device is not None:
        model = model.to(device)
    return model
