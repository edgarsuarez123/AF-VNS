"""
RNN/GRU Temporal Module (FR-3.2).

Input:  (B, seq_len=10, n_features=7) HRV feature sequence
Output: (B, hidden_size) temporal embedding
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RNNConfig:
    input_size: int = 7
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0  # GRU dropout applies only if num_layers > 1


class GRUEncoder(nn.Module):
    def __init__(self, cfg: RNNConfig):
        super().__init__()
        self.cfg = cfg

        self.gru = nn.GRU(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, seq_len, input_size)
        returns: (B, hidden_size) using last-layer last hidden state
        """
        if x.ndim != 3:
            raise ValueError(f"GRUEncoder expected input of shape (B, seq_len, F); got {tuple(x.shape)}")
        _, h_n = self.gru(x)
        # h_n: (num_layers, B, hidden_size) -> take last layer
        return h_n[-1]

