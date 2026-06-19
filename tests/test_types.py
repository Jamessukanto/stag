"""Contract tests for the shared types in core.types."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.types import (
    EvaluationResult,
    ProcessedInteraction,
    RawInteraction,
    UserIndex,
)


class TestRawInteraction:
    def test_constructs_with_valid_fields(self) -> None:
        ri = RawInteraction(user_id="a", target_id="b", rating=7)
        assert ri.user_id == "a"
        assert ri.target_id == "b"
        assert ri.rating == 7

    @pytest.mark.parametrize("rating", [0, 11, -1, 100])
    def test_rejects_rating_outside_1_to_10(self, rating: int) -> None:
        with pytest.raises(ValidationError):
            RawInteraction(user_id="a", target_id="b", rating=rating)

    @pytest.mark.parametrize("rating", [1, 5, 10])
    def test_accepts_rating_in_range(self, rating: int) -> None:
        assert RawInteraction(user_id="a", target_id="b", rating=rating).rating == rating


class TestProcessedInteraction:
    def test_constructs_with_valid_fields(self) -> None:
        pi = ProcessedInteraction(user_id="a", target_id="b", label=1, split="train")
        assert pi.label == 1
        assert pi.split == "train"

    @pytest.mark.parametrize("label", [-1, 2, 7])
    def test_rejects_label_outside_0_1(self, label: int) -> None:
        with pytest.raises(ValidationError):
            ProcessedInteraction(user_id="a", target_id="b", label=label, split="train")

    def test_rejects_unknown_split(self) -> None:
        with pytest.raises(ValidationError):
            ProcessedInteraction(user_id="a", target_id="b", label=1, split="holdout")  # type: ignore[arg-type]

    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_accepts_valid_splits(self, split: str) -> None:
        pi = ProcessedInteraction(user_id="a", target_id="b", label=0, split=split)  # type: ignore[arg-type]
        assert pi.split == split


class TestUserIndex:
    def test_from_ids_builds_contiguous_bijection(self) -> None:
        idx = UserIndex.from_ids(["x", "y", "z"])
        assert len(idx) == 3
        for i, uid in enumerate(["x", "y", "z"]):
            assert idx.to_index(uid) == i
            assert idx.to_id(i) == uid

    def test_from_ids_dedups_preserving_first_seen_order(self) -> None:
        idx = UserIndex.from_ids(["b", "a", "b", "c", "a"])
        assert len(idx) == 3
        assert idx.to_index("b") == 0
        assert idx.to_index("a") == 1
        assert idx.to_index("c") == 2

    def test_round_trip_is_bijective(self) -> None:
        idx = UserIndex.from_ids(["u1", "u2", "u3"])
        for uid in ["u1", "u2", "u3"]:
            assert idx.to_id(idx.to_index(uid)) == uid
        for i in range(len(idx)):
            assert idx.to_index(idx.to_id(i)) == i

    def test_rejects_inconsistent_mapping(self) -> None:
        with pytest.raises(ValidationError):
            UserIndex(id_to_index={"a": 0, "b": 1}, index_to_id=["a", "c"])

    def test_rejects_non_contiguous_indices(self) -> None:
        with pytest.raises(ValidationError):
            UserIndex(id_to_index={"a": 0, "b": 5}, index_to_id=["a", "b"])

    def test_unknown_id_raises(self) -> None:
        idx = UserIndex.from_ids(["a"])
        with pytest.raises(KeyError):
            idx.to_index("missing")


class TestEvaluationResult:
    def test_constructs_with_all_metric_fields(self) -> None:
        res = EvaluationResult(
            model_name="mf",
            aggregation="product",
            k=10,
            recall_at_k=0.5,
            hr_at_k=0.4,
            ndcg_at_k=0.3,
            evaluated_at="2026-01-01T00:00:00+00:00",
        )
        assert res.model_name == "mf"
        assert res.aggregation == "product"
        assert res.k == 10
        assert res.recall_at_k == 0.5
        assert res.hr_at_k == 0.4
        assert res.ndcg_at_k == 0.3
        assert res.evaluated_at == "2026-01-01T00:00:00+00:00"
