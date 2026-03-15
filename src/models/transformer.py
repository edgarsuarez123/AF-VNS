"""
Transformer Attention Module (FR-3.3, Step 4 Option B).

Input:  (B, seq_len, n_features=7) HRV feature sequence
Output: (B, d_model) temporal-attention embedding (default 64)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TransformerConfig:
    n_features: int = 7
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 1
    dim_feedforward: int = 128
    dropout: float = 0.1
    seq_len: int = 5
    pooling: str = "mean"  # "mean" or "last"


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        s = x.size(1)
        return x + self.pe[:s].unsqueeze(0).to(dtype=x.dtype, device=x.device)


class TransformerEncoder(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.input_proj = nn.Linear(cfg.n_features, cfg.d_model)
        self.pos = SinusoidalPositionalEncoding(cfg.d_model, max_len=max(cfg.seq_len, 1))

        # Prefer batch_first when available; fallback to (S, B, D) otherwise.
        layer_kwargs = dict(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
        )
        try:
            encoder_layer = nn.TransformerEncoderLayer(batch_first=True, **layer_kwargs)
            self._batch_first = True
        except TypeError:
            encoder_layer = nn.TransformerEncoderLayer(**layer_kwargs)
            self._batch_first = False

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_encoder_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, S, F)
        lengths: (B,) int tensor of valid timesteps per sample (optional)
        returns: (B, D)
        """
        if x.ndim != 3:
            raise ValueError(f"TransformerEncoder expected input of shape (B, S, F); got {tuple(x.shape)}")
        B, S, _ = x.shape
        h = self.input_proj(x)
        h = self.pos(h)

        # Build padding mask: True = IGNORE position
        mask = None
        if lengths is not None:
            positions = torch.arange(S, device=x.device).unsqueeze(0)  # (1, S)
            mask = positions >= lengths.unsqueeze(1)  # (B, S), True=pad

        if self._batch_first:
            z = self.encoder(h, src_key_padding_mask=mask)  # (B, S, D)
        else:
            z = self.encoder(h.transpose(0, 1), src_key_padding_mask=mask).transpose(0, 1)  # (B, S, D)

        z = self.norm(z)
        if self.cfg.pooling == "last":
            if lengths is not None:
                # Use last valid position per sample
                last_idx = (lengths.clamp(min=1) - 1).long()  # (B,)
                return z[torch.arange(B, device=z.device), last_idx]
            return z[:, -1]
        if self.cfg.pooling == "mean":
            if mask is not None:
                # Masked mean pooling: average only valid positions
                valid_mask = ~mask  # (B, S), True=valid
                valid_mask_f = valid_mask.unsqueeze(-1).float()  # (B, S, 1)
                z_masked = z * valid_mask_f
                return z_masked.sum(dim=1) / valid_mask_f.sum(dim=1).clamp(min=1.0)
            return z.mean(dim=1)
        raise ValueError(f"Unknown pooling='{self.cfg.pooling}' (expected 'mean' or 'last')")

