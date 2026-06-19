"""Plug-in interfaces for the reciprocal-rec project, as typing.Protocols.

These are structural contracts only: any class with the right methods conforms,
no inheritance required. They are interfaces, not implementations, and contain
no logic. Swapping one PreferenceModel for another (MF <-> NeuMF) requires no
change to data, aggregation, or evaluation code because all four communicate
only through these Protocols and on-disk artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from core.ground_truth import EvaluationDataset
from core.types import EvaluationResult, ProcessedInteraction


@runtime_checkable
class DataLoader(Protocol):
    """Turns raw ratings into model-ready supervision and exposes negatives."""

    def load(self) -> list[ProcessedInteraction]: ...

    def get_negatives(self, user_id: str, strategy: str, n: int, seed: int) -> list[str]: ...


@runtime_checkable
class PreferenceModel(Protocol):
    """The single plug-in interface both MF and NeuMF implement.

    Produces a directional score s(u -> v); it never computes the reciprocal
    score r(A, B) - that fusion belongs to the Aggregator in evaluation.
    """

    def fit(self, interactions: list[ProcessedInteraction]) -> None: ...

    def directional_score(self, user_u: str, target_v: str) -> float: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> PreferenceModel: ...


@runtime_checkable
class Aggregator(Protocol):
    """The reciprocal fusion function f: r(A, B) = f(s(A->B), s(B->A))."""

    def aggregate(self, s_ab: float, s_ba: float) -> float: ...


@runtime_checkable
class Evaluator(Protocol):
    """Scores a model from its on-disk artifact, never importing model code.

    Ground truth arrives as an :class:`EvaluationDataset` built by the caller
    (typically ``experiments/`` filtering ``DataLoader.load()`` to the eval
    split). ``eval/`` must not import ``data/`` to obtain it.
    """

    def evaluate(
        self,
        artifact_path: Path,
        ground_truth: EvaluationDataset,
        aggregation: str,
        k: int,
    ) -> EvaluationResult: ...
