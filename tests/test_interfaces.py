"""Structural-conformance tests for the Protocol interfaces.

Protocols are interfaces, not implementations: we assert they import, are
runtime-checkable, and that a minimal conforming stub satisfies isinstance
checks while a non-conforming object does not.
"""

from __future__ import annotations

from pathlib import Path

from core.interfaces import (
    Aggregator,
    DataLoader,
    Evaluator,
    PreferenceModel,
)
from core.types import EvaluationResult, ProcessedInteraction


class _StubDataLoader:
    def load(self) -> list[ProcessedInteraction]:
        return []

    def get_negatives(self, user_id: str, strategy: str, n: int, seed: int) -> list[str]:
        return []


class _StubPreferenceModel:
    def fit(self, interactions: list[ProcessedInteraction]) -> None:
        return None

    def directional_score(self, user_u: str, target_v: str) -> float:
        return 0.0

    def save(self, path: Path) -> None:
        return None

    def load(self, path: Path) -> _StubPreferenceModel:
        return self


class _StubAggregator:
    def aggregate(self, s_ab: float, s_ba: float) -> float:
        return s_ab * s_ba


class _StubEvaluator:
    def evaluate(self, artifact_path: Path, aggregation: str, k: int) -> EvaluationResult:
        return EvaluationResult(
            model_name="x",
            aggregation=aggregation,
            k=k,
            recall_at_k=0.0,
            hr_at_k=0.0,
            ndcg_at_k=0.0,
            evaluated_at="2026-01-01T00:00:00+00:00",
        )


class _NotAModel:
    pass


def test_protocols_are_runtime_checkable() -> None:
    assert isinstance(_StubDataLoader(), DataLoader)
    assert isinstance(_StubPreferenceModel(), PreferenceModel)
    assert isinstance(_StubAggregator(), Aggregator)
    assert isinstance(_StubEvaluator(), Evaluator)


def test_non_conforming_object_is_rejected() -> None:
    assert not isinstance(_NotAModel(), PreferenceModel)
    assert not isinstance(_NotAModel(), DataLoader)
