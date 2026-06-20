"""Binary cross-entropy loss for NeuMF training."""

from __future__ import annotations

import torch
from torch import nn

_BCE = nn.BCELoss()


def bce_loss(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Compute BCE between a predicted score and a binary label."""
    loss: torch.Tensor = _BCE(pred, label)
    return loss
