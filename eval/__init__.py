"""Evaluation module: aggregation, ranking, and metrics over ModelArtifacts.

Consumes on-disk artifacts and :class:`core.ground_truth.EvaluationDataset`;
never imports ``models/`` or ``data/`` internals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eval.evaluator import ReciprocalEvaluator

__all__ = ["ReciprocalEvaluator"]


def __getattr__(name: str) -> type[ReciprocalEvaluator]:
    if name == "ReciprocalEvaluator":
        from eval.evaluator import ReciprocalEvaluator

        return ReciprocalEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
