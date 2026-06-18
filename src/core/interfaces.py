"""Structural interfaces (typing.Protocol) every module implementation conforms to.

These Protocols are contracts only. They contain method signatures with no
bodies, so downstream classes are checked structurally without inheriting from
an ABC. Downstream modules MUST NOT redefine these.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from src.core.types import EvaluationResult, ProcessedInteraction


@runtime_checkable
class DataLoader(Protocol):
    """Loads processed interactions and produces negative samples."""

    def load(self) -> list[ProcessedInteraction]:
        """Return all processed interactions (with split labels)."""
        ...

    def get_negatives(
        self, user_id: str, strategy: str, n: int, seed: int
    ) -> list[str]:
        """Return ``n`` negative target user_ids for ``user_id``.

        Sampling is deterministic given ``seed`` and selected by ``strategy``.
        """
        ...


@runtime_checkable
class ReciprocityModel(Protocol):
    """A model that scores reciprocal preference between two users."""

    def fit(self, interactions: list[ProcessedInteraction]) -> None:
        """Train the model on processed interactions."""
        ...

    def predict(self, user_a: str, user_b: str) -> float:
        """Return the reciprocal preference score r(A, B)."""
        ...

    def save(self, path: Path) -> None:
        """Serialize the trained model to a ModelArtifact on disk."""
        ...

    def load(self, path: Path) -> None:
        """Restore the model from a ModelArtifact so predict() works."""
        ...


@runtime_checkable
class Evaluator(Protocol):
    """Evaluates a model artifact loaded from disk."""

    def evaluate(self, artifact_path: Path, k: int) -> EvaluationResult:
        """Load the artifact at ``artifact_path`` and compute metrics at ``k``."""
        ...
