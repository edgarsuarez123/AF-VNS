"""
Build HybridEnsemble from config.yaml. Single source of truth for model dimensions.
"""

import os
from pathlib import Path
from typing import Optional

import torch
import yaml

from ..models.cnn import CNNConfig
from ..models.ensemble import EnsembleConfig, HybridEnsemble
from ..models.rnn import RNNConfig
from ..models.transformer import TransformerConfig


def load_config(config_path: str = "config.yaml") -> dict:
    """Load config from project root if needed."""
    if not os.path.isabs(config_path) and not os.path.isfile(config_path):
        root = Path(__file__).resolve().parents[2]
        config_path = root / config_path
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_ensemble_config(config_path: str = "config.yaml") -> EnsembleConfig:
    """Build EnsembleConfig from config.yaml model section."""
    config = load_config(config_path)
    m = config.get("model", {})

    cnn_cfg = CNNConfig(
        in_channels=int(m.get("in_channels", 1)),
        embed_dim=int(m.get("cnn_embed_dim", 128)),
        dropout=0.1,
    )
    rnn_cfg = RNNConfig(
        input_size=int(m.get("hrv_n_features", 7)),
        hidden_size=int(m.get("rnn_hidden_size", 64)),
        num_layers=int(m.get("rnn_num_layers", 1)),
        dropout=0.0,
    )
    trans_cfg = TransformerConfig(
        n_features=int(m.get("hrv_n_features", 7)),
        d_model=int(m.get("transformer_d_model", 64)),
        nhead=int(m.get("transformer_nhead", 4)),
        num_encoder_layers=int(m.get("transformer_num_encoder_layers", 1)),
        dim_feedforward=int(m.get("transformer_dim_feedforward", 128)),
        dropout=0.1,
        seq_len=int(m.get("hrv_seq_len", 5)),
        pooling="mean",
    )
    return EnsembleConfig(cnn=cnn_cfg, rnn=rnn_cfg, transformer=trans_cfg, dropout=0.2)


def build_model(
    config_path: str = "config.yaml",
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
) -> HybridEnsemble:
    """
    Build HybridEnsemble from config; optionally load checkpoint.
    If device is None, does not move to device (caller's responsibility).
    """
    cfg = build_ensemble_config(config_path)
    model = HybridEnsemble(cfg)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"], strict=True)
        else:
            model.load_state_dict(state, strict=True)
    if device is not None:
        model = model.to(device)
    return model
