"""
RNN/GRU Temporal Module (FR-3.2).

Input:  (B, seq_len, n_features=7) HRV feature sequence
Output: (B, hidden_size) temporal embedding
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


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

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, seq_len, input_size)
        lengths: (B,) int tensor of valid HRV timesteps per sample (optional)
        returns: (B, hidden_size) using last-layer last hidden state
        """
        if x.ndim != 3:
            raise ValueError(f"GRUEncoder expected input of shape (B, seq_len, F); got {tuple(x.shape)}")
        if lengths is not None:
            packed = pack_padded_sequence(
                x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.gru(packed)
        else:
            _, h_n = self.gru(x)
        # h_n: (num_layers, B, hidden_size) -> take last layer
        return h_n[-1]

