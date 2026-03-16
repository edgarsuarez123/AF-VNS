"""Tests for label smoothing in training loop."""
import torch


def _apply_label_smoothing(labels: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Reproduce the label smoothing formula from train.py."""
    if epsilon > 0:
        return labels * (1 - epsilon) + epsilon * 0.5
    return labels


def test_label_smoothing_transforms_labels():
    labels = torch.tensor([0.0, 1.0])
    smoothed = _apply_label_smoothing(labels, 0.1)
    assert torch.allclose(smoothed, torch.tensor([0.05, 0.95])), f"Expected [0.05, 0.95], got {smoothed}"


def test_label_smoothing_zero_is_noop():
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    smoothed = _apply_label_smoothing(labels, 0.0)
    assert torch.equal(smoothed, labels), "epsilon=0 should leave labels unchanged"
