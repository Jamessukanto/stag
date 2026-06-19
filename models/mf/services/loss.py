"""Squared loss and L2 regularization gradients for weighted MF."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def squared_loss_and_grad(score: float, label: float) -> tuple[float, float]:
    """Return (loss, d_loss/d_score) for (label - score)^2."""
    residual = label - score
    loss = residual * residual
    grad = -2.0 * residual
    return loss, grad


def l2_grad(embedding: npt.NDArray[np.float64], l2_weight: float) -> npt.NDArray[np.float64]:
    """Gradient of (l2_weight / 2) * ||embedding||^2 w.r.t. embedding."""
    return l2_weight * embedding
