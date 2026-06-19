"""Negative-sampling strategies and per-positive negative downsampling.

Two things live here, both pure mechanics driven by an explicit ``rng`` so they
are reproducible:

- ``sample_random`` / ``sample_popularity_biased``: draw uninteracted users from
  a candidate pool, used by the DataLoader's ``sample_uninteracted_candidates``.
  These are ranking distractors for evaluation, not explicit dislikes and not
  the training signal.
- ``downsample_negatives``: trim the explicit negatives (from binarization) per
  positive within a set of records. The explicit negatives remain the training
  signal; this only thins them when a smaller negative-to-positive ratio is
  wanted.

Sampling is always without replacement and never exceeds the pool size.
"""

from __future__ import annotations

import numpy as np
from core.types import ProcessedInteraction


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


def downsample_negatives(
    records: list[ProcessedInteraction], ratio: float, rng: np.random.Generator
) -> list[ProcessedInteraction]:
    """Keep all positives; trim negatives to ``round(ratio * n_positives)``.

    Records with fewer negatives than the target are returned unchanged. Input
    order of the kept records is preserved.
    """
    positives = [r for r in records if r.label == 1]
    negatives = [r for r in records if r.label == 0]
    target = round(ratio * len(positives))
    if len(negatives) <= target:
        return list(records)
    keep_idx = rng.choice(len(negatives), size=target, replace=False)
    keep_set = {int(i) for i in keep_idx}
    kept_negatives = [neg for i, neg in enumerate(negatives) if i in keep_set]
    return positives + kept_negatives
