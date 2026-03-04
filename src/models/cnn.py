"""
CNN Morphology Module (FR-3.1).

Input:  (B, C=1, T) 10-second waveform
Output: (B, embed_dim) morphology embedding
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CNNConfig:
    in_channels: int = 1
    embed_dim: int = 128
    dropout: float = 0.1


class CNNEncoder(nn.Module):
    def __init__(self, cfg: CNNConfig):
        super().__init__()
        self.cfg = cfg

        self.net = nn.Sequential(
            nn.Conv1d(cfg.in_channels, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(128, cfg.embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        returns: (B, embed_dim)
        """
        if x.ndim != 3:
            raise ValueError(f"CNNEncoder expected input of shape (B, C, T); got {tuple(x.shape)}")
        h = self.net(x)
        return self.proj(h)

