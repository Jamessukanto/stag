"""Tests for candidate pools, distractor sampling, and leave-one-out folds."""

from __future__ import annotations

from core.ground_truth import EvaluationDataset
from core.types import ProcessedInteraction, UserIndex
from eval.services.candidates import (
    LeaveOneOutFold,
    build_uninteracted_pool,
    leave_one_out_folds,
    sample_distractors,
)


class TestBuildUninteractedPool:
    def test_excludes_self_and_interacted_targets(
        self,
        processed_interactions: list[ProcessedInteraction],
        user_index: UserIndex,
    ) -> None:
        dataset = EvaluationDataset.from_interactions(
            processed_interactions, split="test"
        )
        pool = build_uninteracted_pool(
            user_id="u1",
            user_index=user_index,
            interacted_targets_by_user=dataset.interacted_targets_by_user,
        )
        assert "u1" not in pool
        assert "u2" not in pool
        assert "u3" not in pool
        assert "u4" not in pool
        assert pool == ["u5"]


class TestSampleDistractors:
    def test_random_is_deterministic_under_seed(self) -> None:
        candidates = ["a", "b", "c", "d", "e"]
        a = sample_distractors(
            candidates, n=3, strategy="random", seed=123, popularity={}
        )
        b = sample_distractors(
            candidates, n=3, strategy="random", seed=123, popularity={}
        )
        assert a == b
        assert len(a) == 3
        assert len(set(a)) == 3

    def test_popularity_biased_is_deterministic_under_seed(self) -> None:
        candidates = ["a", "b", "c", "d"]
        popularity = {"a": 1, "b": 5, "c": 2, "d": 0}
        a = sample_distractors(
            candidates,
            n=2,
            strategy="popularity_biased",
            seed=9,
            popularity=popularity,
        )
        b = sample_distractors(
            candidates,
            n=2,
            strategy="popularity_biased",
            seed=9,
            popularity=popularity,
        )
        assert a == b
        assert len(a) == 2


class TestLeaveOneOutFolds:
    def test_one_fold_per_mutual_match_partner(
        self, processed_interactions: list[ProcessedInteraction]
    ) -> None:
        dataset = EvaluationDataset.from_interactions(
            processed_interactions, split="train"
        )
        folds = leave_one_out_folds(dataset)
        u1_folds = [f for f in folds if f.user_id == "u1"]
        assert len(u1_folds) == 1
        assert u1_folds[0].held_out_partner == "u2"

    def test_excludes_one_sided_test_pair(
        self, processed_interactions: list[ProcessedInteraction]
    ) -> None:
        dataset = EvaluationDataset.from_interactions(
            processed_interactions, split="test"
        )
        folds = leave_one_out_folds(dataset)
        u5_folds = [f for f in folds if f.user_id == "u5"]
        assert u5_folds == []

    def test_fold_dataclass_fields(
        self, processed_interactions: list[ProcessedInteraction]
    ) -> None:
        dataset = EvaluationDataset.from_interactions(
            processed_interactions, split="train"
        )
        folds = leave_one_out_folds(dataset)
        assert all(isinstance(f, LeaveOneOutFold) for f in folds)
        assert all(f.fold_index >= 0 for f in folds)
