"""Distractor sampling strategies (boundary copy of data.services.sampling semantics)."""

from __future__ import annotations

import numpy as np


def sample_random(candidates: list[str], n: int, rng: np.random.Generator) -> list[str]:
    """Uniformly sample up to ``n`` distinct ids from ``candidates``."""
    if n <= 0 or not candidates:
        return []
    size = min(n, len(candidates))
    chosen = rng.choice(len(candidates), size=size, replace=False)
    return [candidates[int(i)] for i in chosen]


def sample_popularity_biased(
    candidates: list[str], weights: list[float], n: int, rng: np.random.Generator
) -> list[str]:
    """Sample up to ``n`` distinct ids with probability proportional to ``weights``."""
    if n <= 0 or not candidates:
        return []
    size = min(n, len(candidates))
    weight_arr = np.asarray(weights, dtype=np.float64)
    total = weight_arr.sum()
    if total <= 0:
        return sample_random(candidates, n, rng)
    probabilities = weight_arr / total
    chosen = rng.choice(len(candidates), size=size, replace=False, p=probabilities)
    return [candidates[int(i)] for i in chosen]
