"""
StrokeResponderHead — binary classification head for stroke AVNS.

Extracted from HybridEnsemble so it can be independently frozen/replaced between
Phase 1 (full training) and Phase 2 (backbone frozen, head fine-tuned on CereVasc).

Input:  (B, fused_dim) — concatenated CNN + RNN + Transformer embeddings
Output: (B, 1) logits for BCEWithLogitsLoss
"""

from torch import nn
import torch


class StrokeResponderHead(nn.Module):
    """Binary classification head: fused embedding → 1 logit.

    Args:
        fused_dim:      Dimension of concatenated encoder embeddings (default 256 = 128+64+64).
        head_hidden_dim: If > 0, adds a hidden Linear+ReLU layer before the output projection.
                         Useful for Phase 2 fine-tuning with a larger head capacity.
        dropout:        Dropout probability before first linear layer.
    """

    def __init__(self, fused_dim: int, head_hidden_dim: int = 0, dropout: float = 0.2):
        super().__init__()
        if head_hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(fused_dim, head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(head_hidden_dim, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(fused_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, fused_dim) → (B, 1) logits"""
        return self.net(x)
