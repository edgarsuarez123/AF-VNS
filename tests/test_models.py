"""
Step 4 tests: verify model module shapes and a dummy forward pass.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.cnn import CNNConfig, CNNEncoder
from src.models.rnn import GRUEncoder, RNNConfig
from src.models.transformer import TransformerConfig, TransformerEncoder
from src.models.ensemble import EnsembleConfig, HybridEnsemble
from src.models.pca_reduction import PCAReducer


def test_cnn_shape():
    model = CNNEncoder(CNNConfig(in_channels=1, embed_dim=128))
    x = torch.randn(32, 1, 2500)
    y = model(x)
    assert y.shape == (32, 128)


def test_rnn_shape():
    model = GRUEncoder(RNNConfig(input_size=7, hidden_size=64, num_layers=1))
    x = torch.randn(32, 5, 7)
    y = model(x)
    assert y.shape == (32, 64)


def test_transformer_shape():
    model = TransformerEncoder(
        TransformerConfig(n_features=7, d_model=64, nhead=4, num_encoder_layers=1, seq_len=5)
    )
    x = torch.randn(32, 5, 7)
    y = model(x)
    assert y.shape == (32, 64)


def test_ensemble_dummy_forward():
    model = HybridEnsemble(EnsembleConfig())
    waveform = torch.randn(32, 1, 2500)
    hrv_seq = torch.randn(32, 5, 7)
    logits = model(waveform, hrv_seq)
    assert logits.shape == (32, 1)


def test_pca_passthrough_single_channel():
    X = np.random.randn(100, 1)
    reducer = PCAReducer(n_components=1).fit(X)
    Y = reducer.transform(X)
    assert Y.shape == (100, 1)
    assert np.allclose(Y, X)


def test_pca_multi_channel_reduces_to_one():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 8))
    reducer = PCAReducer(n_components=1).fit(X)
    Y = reducer.transform(X)
    assert Y.shape == (100, 1)

