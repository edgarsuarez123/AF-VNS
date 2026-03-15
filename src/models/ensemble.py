"""
Hybrid Ensemble Model (Phase 1).

Combines:
- CNN morphology embedding from 10s waveform (FR-3.1)
- GRU temporal embedding from HRV sequence (FR-3.2)
- Transformer attention embedding from HRV sequence (FR-3.3, Option B)

Outputs a single logit per sample for BCEWithLogitsLoss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .cnn import CNNConfig, CNNEncoder
from .rnn import GRUEncoder, RNNConfig
from .transformer import TransformerConfig, TransformerEncoder


@dataclass(frozen=True)
class EnsembleConfig:
    cnn: CNNConfig = CNNConfig()
    rnn: RNNConfig = RNNConfig()
    transformer: TransformerConfig = TransformerConfig()
    dropout: float = 0.2


class HybridEnsemble(nn.Module):
    def __init__(self, cfg: EnsembleConfig):
        super().__init__()
        self.cfg = cfg

        self.cnn = CNNEncoder(cfg.cnn)
        self.rnn = GRUEncoder(cfg.rnn)
        self.transformer = TransformerEncoder(cfg.transformer)

        fused_dim = cfg.cnn.embed_dim + cfg.rnn.hidden_size + cfg.transformer.d_model
        self.head = nn.Sequential(
            nn.Dropout(p=cfg.dropout),
            nn.Linear(fused_dim, 1),
        )

    def forward(self, waveform_10s: torch.Tensor, hrv_sequence: torch.Tensor,
                hrv_lengths: torch.Tensor = None) -> torch.Tensor:
        """
        waveform_10s: (B, 1, T)
        hrv_sequence: (B, seq_len, 7) (scaled)
        hrv_lengths: (B,) int tensor of valid HRV timesteps (optional)
        returns: (B, 1) logits
        """
        cnn_emb = self.cnn(waveform_10s)
        rnn_emb = self.rnn(hrv_sequence, lengths=hrv_lengths)
        tr_emb = self.transformer(hrv_sequence, lengths=hrv_lengths)
        fused = torch.cat([cnn_emb, rnn_emb, tr_emb], dim=1)
        return self.head(fused)

