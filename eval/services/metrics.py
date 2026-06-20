"""Pure ranking metrics over explicit candidate orderings."""

from __future__ import annotations

import math
from collections.abc import Collection


def recall_at_k(ranking: list[str], relevant: Collection[str], *, k: int) -> float:
    """Fraction of ``relevant`` ids appearing in the top-``k`` positions."""
    if not relevant:
        return 0.0
    top_k = set(ranking[:k])
    hits = sum(1 for item in relevant if item in top_k)
    return hits / len(relevant)


def hr_at_k(ranking: list[str], positive: str, *, k: int) -> float:
    """Hit rate: 1.0 when ``positive`` is in the top-``k``, else 0.0."""
    top_k = ranking[:k]
    return 1.0 if positive in top_k else 0.0


def ndcg_at_k(ranking: list[str], positive: str, *, k: int) -> float:
    """NDCG@K for a single relevant item (IDCG = 1 when ranked first)."""
    top_k = ranking[:k]
    if positive not in top_k:
        return 0.0
    rank = top_k.index(positive) + 1
    return 1.0 / math.log2(rank + 1.0)


def mean_metric(values: list[float]) -> float:
    """Macro-average; returns 0.0 for an empty list."""
    if not values:
        return 0.0
    return sum(values) / len(values)
