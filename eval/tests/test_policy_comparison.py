"""Tests for engagement vs reciprocal ranking policy comparison."""

from __future__ import annotations

from pathlib import Path

import pytest
from core.ground_truth import EvaluationDataset
from core.serialization import ModelArtifact
from core.types import ProcessedInteraction, UserIndex

from eval.services.policy_comparison import compare_ranking_policies


def _policy_fixture() -> tuple[ModelArtifact, EvaluationDataset]:
    """u1 has test mutual u2; u3 is a one-sided crush ranked above u2 by engagement."""
    user_index = UserIndex.from_ids(["u1", "u2", "u3"])
    artifact = ModelArtifact(
        model_name="mf",
        sampling_strategy="random",
        hyperparameters={"embedding_dim": 2},
        user_index=user_index,
        source_embeddings=[
            [1.0, 0.0],  # u1
            [0.0, 1.0],  # u2
            [0.0, 0.1],  # u3
        ],
        target_embeddings=[
            [0.0, 1.0],  # u1
            [1.0, 0.0],  # u2  s(u1->u2)=1, s(u2->u1)=1
            [2.0, 0.0],  # u3  s(u1->u3)=2, s(u3->u1)=0.01
        ],
        extra={},
        trained_on_split="train",
        created_at="2026-01-01T00:00:00+00:00",
    )
    interactions = [
        ProcessedInteraction(user_id="u1", target_id="u2", label=1, split="test"),
        ProcessedInteraction(user_id="u2", target_id="u1", label=1, split="test"),
        ProcessedInteraction(user_id="u1", target_id="u3", label=1, split="test"),
        ProcessedInteraction(user_id="u3", target_id="u1", label=0, split="test"),
    ]
    dataset = EvaluationDataset.from_interactions(interactions, split="test")
    return artifact, dataset


class TestCompareRankingPolicies:
    def test_mutual_recall_beats_engagement_on_fixture(self) -> None:
        artifact, dataset = _policy_fixture()
        result = compare_ranking_policies(
            artifact,
            dataset,
            k=1,
            aggregation="harmonic",
            weighted_alpha=0.5,
        )
        assert result.users_with_test_mutuals == 2
        assert result.mean_recall_engagement == pytest.approx(0.5)
        assert result.mean_recall_mutual == pytest.approx(1.0)
        assert result.mean_recall_delta == pytest.approx(0.5)
        assert result.buried_mutual_rate == pytest.approx(0.5)

    def test_example_case_captures_rank_flip(self) -> None:
        artifact, dataset = _policy_fixture()
        result = compare_ranking_policies(
            artifact,
            dataset,
            k=1,
            aggregation="harmonic",
            weighted_alpha=0.5,
        )
        assert len(result.example_cases) >= 1
        case = result.example_cases[0]
        assert case["user_id"] == "u1"
        assert case["engagement_top1"] == "u3"
        assert case["mutual_top1"] == "u2"
        assert case["engagement_top1_is_mutual"] is False

    def test_loads_from_artifact_path(
        self,
        tmp_path: Path,
    ) -> None:
        artifact, dataset = _policy_fixture()
        path = tmp_path / "mf.json"
        artifact.save(path)
        from eval.services.policy_comparison import compare_ranking_policies_from_path

        result = compare_ranking_policies_from_path(
            path,
            dataset,
            k=1,
            aggregation="harmonic",
            weighted_alpha=0.5,
        )
        assert result.model_name == "mf"
        assert result.mean_recall_mutual == pytest.approx(1.0)
