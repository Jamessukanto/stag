"""Tests for reciprocal aggregation functions (eval.services.aggregators)."""

from __future__ import annotations

import pytest
from core.interfaces import Aggregator
from eval.services.aggregators import (
    HarmonicAggregator,
    ProductAggregator,
    WeightedAggregator,
    get_aggregator,
)


class TestProductAggregator:
    def test_hand_computed_product(self) -> None:
        agg = ProductAggregator()
        assert agg.aggregate(2.0, 3.0) == pytest.approx(6.0)


class TestHarmonicAggregator:
    def test_hand_computed_harmonic(self) -> None:
        agg = HarmonicAggregator()
        assert agg.aggregate(2.0, 3.0) == pytest.approx(2.4)

    def test_zero_sum_returns_zero(self) -> None:
        agg = HarmonicAggregator()
        assert agg.aggregate(0.0, 0.0) == 0.0


class TestWeightedAggregator:
    def test_hand_computed_weighted_at_half(self) -> None:
        agg = WeightedAggregator(alpha=0.5)
        assert agg.aggregate(2.0, 3.0) == pytest.approx(2.5)

    def test_alpha_weights_first_direction(self) -> None:
        agg = WeightedAggregator(alpha=0.8)
        assert agg.aggregate(10.0, 0.0) == pytest.approx(8.0)


class TestGetAggregator:
    def test_returns_protocol_conformant_instances(self) -> None:
        for name in ("product", "harmonic", "weighted"):
            agg = get_aggregator(name)
            assert isinstance(agg, Aggregator)

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown aggregation"):
            get_aggregator("geometric")
