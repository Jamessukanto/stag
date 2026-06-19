"""Embedding initialization and directional dot-product scoring."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def init_embeddings(
    n_users: int,
    dim: int,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Initialize source (p) and target (q) embedding tables with small random values."""
    scale = np.sqrt(2.0 / dim)
    source = rng.normal(0.0, scale, size=(n_users, dim)).astype(np.float64)
    target = rng.normal(0.0, scale, size=(n_users, dim)).astype(np.float64)
    return source, target


def directional_dot(
    source: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    u_idx: int,
    v_idx: int,
) -> float:
    """Directional score s(u->v) = p_u^T q_v."""
    return float(np.dot(source[u_idx], target[v_idx]))
