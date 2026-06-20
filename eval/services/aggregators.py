"""Reciprocal fusion functions: r(A, B) = f(s(A->B), s(B->A))."""

from __future__ import annotations

from core.interfaces import Aggregator

_PRODUCT = "product"
_HARMONIC = "harmonic"
_WEIGHTED = "weighted"


class ProductAggregator:
    """r = s_ab * s_ba."""

    def aggregate(self, s_ab: float, s_ba: float) -> float:
        return s_ab * s_ba


class HarmonicAggregator:
    """r = 2 * s_ab * s_ba / (s_ab + s_ba); zero when the sum is zero."""

    def aggregate(self, s_ab: float, s_ba: float) -> float:
        total = s_ab + s_ba
        if total == 0.0:
            return 0.0
        return 2.0 * s_ab * s_ba / total


class WeightedAggregator:
    """r = alpha * s_ab + (1 - alpha) * s_ba."""

    def __init__(self, *, alpha: float = 0.5) -> None:
        self._alpha = alpha

    def aggregate(self, s_ab: float, s_ba: float) -> float:
        return self._alpha * s_ab + (1.0 - self._alpha) * s_ba


def get_aggregator(name: str, *, alpha: float = 0.5) -> Aggregator:
    """Return an :class:`Aggregator` by name."""
    if name == _PRODUCT:
        return ProductAggregator()
    if name == _HARMONIC:
        return HarmonicAggregator()
    if name == _WEIGHTED:
        return WeightedAggregator(alpha=alpha)
    raise ValueError(
        f"unknown aggregation {name!r}; expected {_PRODUCT!r}, "
        f"{_HARMONIC!r}, or {_WEIGHTED!r}"
    )
