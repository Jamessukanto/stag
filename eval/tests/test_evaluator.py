"""Integration tests for ReciprocalEvaluator."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from core.config import Config
from core.ground_truth import EvaluationDataset
from core.interfaces import Evaluator
from core.serialization import ModelArtifact
from core.types import ProcessedInteraction
from eval.evaluator import ReciprocalEvaluator


@pytest.fixture
def config() -> Config:
    return Config(random_seed=42)


@pytest.fixture
def evaluator(config: Config) -> ReciprocalEvaluator:
    return ReciprocalEvaluator(config, ncf_distractors=2)


@pytest.fixture
def train_dataset(
    processed_interactions: list[ProcessedInteraction],
) -> EvaluationDataset:
    return EvaluationDataset.from_interactions(processed_interactions, split="train")


class TestReciprocalEvaluator:
    def test_conforms_to_evaluator_protocol(self, evaluator: ReciprocalEvaluator) -> None:
        assert isinstance(evaluator, Evaluator)

    def test_evaluate_returns_result_with_expected_fields(
        self,
        evaluator: ReciprocalEvaluator,
        sample_artifact: ModelArtifact,
        train_dataset: EvaluationDataset,
        tmp_path: Path,
    ) -> None:
        artifact_path = tmp_path / "mf.json"
        sample_artifact.save(artifact_path)
        result = evaluator.evaluate(
            artifact_path, train_dataset, aggregation="product", k=5
        )
        assert result.model_name == "mf"
        assert result.aggregation == "product"
        assert result.k == 5
        assert 0.0 <= result.recall_at_k <= 1.0
        assert 0.0 <= result.hr_at_k <= 1.0
        assert 0.0 <= result.ndcg_at_k <= 1.0
        datetime.fromisoformat(result.evaluated_at)

    def test_identical_embeddings_produce_identical_metrics(
        self,
        evaluator: ReciprocalEvaluator,
        sample_artifact: ModelArtifact,
        train_dataset: EvaluationDataset,
        tmp_path: Path,
    ) -> None:
        mf_path = tmp_path / "mf.json"
        neumf_path = tmp_path / "neumf.json"
        sample_artifact.save(mf_path)
        neumf = sample_artifact.model_copy(update={"model_name": "neumf"})
        neumf.save(neumf_path)

        mf_result = evaluator.evaluate(mf_path, train_dataset, "product", k=5)
        neumf_result = evaluator.evaluate(neumf_path, train_dataset, "product", k=5)

        assert mf_result.recall_at_k == pytest.approx(neumf_result.recall_at_k)
        assert mf_result.hr_at_k == pytest.approx(neumf_result.hr_at_k)
        assert mf_result.ndcg_at_k == pytest.approx(neumf_result.ndcg_at_k)

    def test_unknown_aggregation_raises(
        self,
        evaluator: ReciprocalEvaluator,
        sample_artifact: ModelArtifact,
        train_dataset: EvaluationDataset,
        tmp_path: Path,
    ) -> None:
        artifact_path = tmp_path / "mf.json"
        sample_artifact.save(artifact_path)
        with pytest.raises(ValueError, match="unknown aggregation"):
            evaluator.evaluate(artifact_path, train_dataset, "invalid", k=5)
