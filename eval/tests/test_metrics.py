"""Tests for ranking metrics (eval.services.metrics)."""

from __future__ import annotations

import math

import pytest
from eval.services.metrics import hr_at_k, ndcg_at_k, recall_at_k


class TestRecallAtK:
    def test_all_relevant_in_top_k(self) -> None:
        ranking = ["rel", "d1", "d2"]
        assert recall_at_k(ranking, {"rel"}, k=1) == pytest.approx(1.0)

    def test_miss_at_k(self) -> None:
        ranking = ["d1", "rel", "d2"]
        assert recall_at_k(ranking, {"rel"}, k=1) == pytest.approx(0.0)

    def test_partial_recall(self) -> None:
        ranking = ["r1", "d1", "r2", "d2"]
        assert recall_at_k(ranking, {"r1", "r2"}, k=2) == pytest.approx(0.5)


class TestHrAtK:
    def test_hit_at_rank_one(self) -> None:
        ranking = ["rel", "d1", "d2"]
        assert hr_at_k(ranking, "rel", k=1) == pytest.approx(1.0)

    def test_miss_at_k(self) -> None:
        ranking = ["d1", "rel", "d2"]
        assert hr_at_k(ranking, "rel", k=1) == pytest.approx(0.0)


class TestNdcgAtK:
    def test_perfect_rank_one(self) -> None:
        ranking = ["rel", "d1", "d2"]
        assert ndcg_at_k(ranking, "rel", k=1) == pytest.approx(1.0)

    def test_rank_two_within_k(self) -> None:
        ranking = ["d1", "rel"]
        expected = 1.0 / math.log2(3.0)
        assert ndcg_at_k(ranking, "rel", k=2) == pytest.approx(expected)

    def test_miss_at_k_is_zero(self) -> None:
        ranking = ["d1", "rel"]
        assert ndcg_at_k(ranking, "rel", k=1) == pytest.approx(0.0)
