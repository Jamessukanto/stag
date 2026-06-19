"""Ground-truth contracts for evaluation without importing ``data/``.

Evaluation never loads raw ratings itself. ``experiments/`` (or any caller) uses
``DataLoader.load()``, builds an :class:`EvaluationDataset` for the target split,
and passes it to ``Evaluator.evaluate``. The dataset can also be serialized to
disk so eval runs are reproducible without re-parsing Libimseti.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from core.types import ProcessedInteraction, Split


class EvaluationDataset(BaseModel):
    """Held-out supervision that ``eval/`` ranks against.

    Typically contains one split (e.g. ``test``) filtered from
    ``DataLoader.load()``. Mutual-match ground truth is derived from these
    interactions via :func:`mutual_match_partners`.
    """

    split: Split
    interactions: list[ProcessedInteraction] = Field(default_factory=list)

    @model_validator(mode="after")
    def _interactions_match_split(self) -> EvaluationDataset:
        for interaction in self.interactions:
            if interaction.split != self.split:
                raise ValueError(
                    f"interaction {interaction.user_id!r} -> "
                    f"{interaction.target_id!r} has split "
                    f"{interaction.split!r}, expected {self.split!r}"
                )
        return self

    @classmethod
    def from_interactions(
        cls,
        interactions: list[ProcessedInteraction],
        *,
        split: Split,
    ) -> EvaluationDataset:
        """Filter ``interactions`` to a single split."""
        filtered = filter_split(interactions, split)
        return cls(split=split, interactions=filtered)

    def save(self, path: Path) -> None:
        """Write the dataset as human-readable JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> EvaluationDataset:
        """Read a dataset previously written by :meth:`save`."""
        data = json.loads(Path(path).read_text())
        return cls.model_validate(data)


def filter_split(
    interactions: list[ProcessedInteraction], split: Split
) -> list[ProcessedInteraction]:
    """Return interactions belonging to ``split``."""
    return [interaction for interaction in interactions if interaction.split == split]


def mutual_match_partners(
    interactions: list[ProcessedInteraction], user_id: str
) -> set[str]:
    """Users ``v`` where both ``user_id -> v`` and ``v -> user_id`` are likes.

    A like is ``label == 1`` (rating >= 7 after binarization). One-sided pairs
    are excluded — this is the mutual-preference ground truth eval metrics use.
    """
    outgoing_likes: set[str] = {
        interaction.target_id
        for interaction in interactions
        if interaction.user_id == user_id and interaction.label == 1
    }
    incoming_likes: set[str] = {
        interaction.user_id
        for interaction in interactions
        if interaction.target_id == user_id and interaction.label == 1
    }
    return outgoing_likes & incoming_likes
