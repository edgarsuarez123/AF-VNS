"""
PhaseDetector CNN — real-time diastolic + exhalation phase detection (FR-2.1).

Input:  (B, 1, 500)  2-second ECG window at 250 Hz
Output: (B, 10, 2)   per-frame logits at 5 Hz — [diastole, exhalation]

Architecture (~7.5K params, no padding):
    Conv1d(1→16, k=7, s=5) + BN + ReLU   → (B, 16, 99)
    Conv1d(16→32, k=5, s=2) + BN + ReLU  → (B, 32, 48)
    Conv1d(32→48, k=3, s=2) + BN + ReLU  → (B, 48, 23)
    AdaptiveAvgPool1d(10)                 → (B, 48, 10)
    Dropout(0.1) → Conv1d(48→2, k=1)     → (B, 2, 10)
    permute(0, 2, 1)                      → (B, 10, 2)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PhaseDetectorConfig:
    channels: tuple[int, ...] = (16, 32, 48)
    kernels: tuple[int, ...] = (7, 5, 3)
    strides: tuple[int, ...] = (5, 2, 2)
    dropout: float = 0.1
    n_frames: int = 10       # output frames per window (2s × 5Hz)
    n_tasks: int = 2         # diastole + exhalation
    input_samples: int = 500  # 2s @ 250Hz


class PhaseDetector(nn.Module):
    """Lightweight 1D CNN for frame-level cardiac/respiratory phase detection."""

    def __init__(self, cfg: PhaseDetectorConfig):
        super().__init__()
        self.cfg = cfg

        # Backbone: Conv1d + BN + ReLU blocks (bias=False — BN absorbs it)
        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch, k, s in zip(cfg.channels, cfg.kernels, cfg.strides):
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
        self.backbone = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool1d(cfg.n_frames)
        self.drop = nn.Dropout(p=cfg.dropout)
        # 1×1 conv head — outputs n_tasks logits per frame
        self.head = nn.Conv1d(cfg.channels[-1], cfg.n_tasks, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, input_samples) raw ECG window

        Returns:
            (B, n_frames, n_tasks) raw logits — apply sigmoid externally
        """
        if x.ndim != 3:
            raise ValueError(
                f"PhaseDetector expected (B, 1, {self.cfg.input_samples}); got {tuple(x.shape)}"
            )
        if x.shape[1] != 1:
            raise ValueError(
                f"PhaseDetector expected 1 input channel; got {x.shape[1]}"
            )
        if x.shape[2] != self.cfg.input_samples:
            raise ValueError(
                f"PhaseDetector expected {self.cfg.input_samples} samples; got {x.shape[2]}"
            )

        h = self.backbone(x)   # (B, channels[-1], T')
        h = self.pool(h)       # (B, channels[-1], n_frames)
        h = self.drop(h)
        h = self.head(h)       # (B, n_tasks, n_frames)
        return h.permute(0, 2, 1)  # (B, n_frames, n_tasks)

    def param_count(self) -> int:
        """Total trainable + non-trainable parameters."""
        return sum(p.numel() for p in self.parameters())


def build_phase_detector(
    config_path: str = "config_stroke.yaml",
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
) -> PhaseDetector:
    """Build PhaseDetector from config; optionally load checkpoint.

    Args:
        config_path:     Path to config_stroke.yaml.
        checkpoint_path: Path to .pth checkpoint. Optional.
        device:          Move model to this device before returning. Optional.

    Returns:
        PhaseDetector
    """
    from ..training.build_model import load_config

    config = load_config(config_path)
    m = config.get("phase_model", {})

    cfg = PhaseDetectorConfig(
        channels=tuple(m.get("channels", [16, 32, 48])),
        kernels=tuple(m.get("kernels", [7, 5, 3])),
        strides=tuple(m.get("strides", [5, 2, 2])),
        dropout=float(m.get("dropout", 0.1)),
        n_frames=int(m.get("n_frames", 10)),
        n_tasks=int(m.get("n_tasks", 2)),
    )
    model = PhaseDetector(cfg)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.info(
                "build_phase_detector: %d missing keys: %s...",
                len(missing), missing[:3],
            )
        if unexpected:
            logger.info(
                "build_phase_detector: %d unexpected keys: %s...",
                len(unexpected), unexpected[:3],
            )

    if device is not None:
        model = model.to(device)
    return model
