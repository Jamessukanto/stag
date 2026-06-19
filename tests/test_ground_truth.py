"""Tests for evaluation ground-truth contracts (core.ground_truth)."""

from __future__ import annotations

from pathlib import Path

import pytest
from core.ground_truth import (
    EvaluationDataset,
    build_interacted_targets_by_user,
    filter_split,
    mutual_match_partners,
)
from core.types import ProcessedInteraction


class TestFilterSplit:
    def test_returns_only_requested_split(
        self, processed_interactions: list[ProcessedInteraction]
    ) -> None:
        test_rows = filter_split(processed_interactions, "test")
        assert test_rows
        assert all(row.split == "test" for row in test_rows)


class TestMutualMatchPartners:
    def test_mutual_match_requires_bidirectional_likes(
        self, processed_interactions: list[ProcessedInteraction]
    ) -> None:
        train = filter_split(processed_interactions, "train")
        assert "u2" in mutual_match_partners(train, "u1")
        assert "u3" in mutual_match_partners(train, "u2")
        # u1 likes u3 but u3 dislikes u1 — not mutual
        assert "u3" not in mutual_match_partners(train, "u1")

    def test_one_sided_test_pair_is_excluded(
        self, processed_interactions: list[ProcessedInteraction]
    ) -> None:
        test_rows = filter_split(processed_interactions, "test")
        # u5 likes u2 in test; u2 dislikes u5 in test
        assert "u2" not in mutual_match_partners(test_rows, "u5")


class TestEvaluationDataset:
    def test_from_interactions_filters_split(
        self, processed_interactions: list[ProcessedInteraction]
    ) -> None:
        dataset = EvaluationDataset.from_interactions(
            processed_interactions, split="test"
        )
        assert dataset.split == "test"
        assert all(row.split == "test" for row in dataset.interactions)
        assert dataset.interacted_targets_by_user["u1"] == ["u2", "u3", "u4"]

    def test_build_interacted_targets_from_all_splits(
        self, processed_interactions: list[ProcessedInteraction]
    ) -> None:
        graph = build_interacted_targets_by_user(processed_interactions)
        assert "u5" in graph["u2"]  # test-split interaction still in graph

    def test_rejects_mixed_splits(self) -> None:
        with pytest.raises(ValueError):
            EvaluationDataset(
                split="test",
                interactions=[
                    ProcessedInteraction(
                        user_id="u1",
                        target_id="u2",
                        label=1,
                        split="train",
                    )
                ],
            )

    def test_save_load_round_trip(
        self,
        processed_interactions: list[ProcessedInteraction],
        tmp_path: Path,
    ) -> None:
        original = EvaluationDataset.from_interactions(
            processed_interactions, split="val"
        )
        path = tmp_path / "eval_ground_truth.json"
        original.save(path)
        loaded = EvaluationDataset.load(path)
        assert loaded == original
