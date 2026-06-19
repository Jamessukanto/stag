"""Shared, frozen data contracts for the reciprocal-rec project.

These types are the vocabulary every module speaks. They hold no behavior
beyond validation and the bidirectional UserIndex lookups; no data loading,
training, aggregation, or evaluation logic lives here.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Split = Literal["train", "val", "test"]


class RawInteraction(BaseModel):
    """A single observed directional rating, as read from the raw dataset."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    target_id: str
    rating: int = Field(ge=1, le=10)


class ProcessedInteraction(BaseModel):
    """A binarized, split-assigned interaction ready for model supervision."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    target_id: str
    label: int
    split: Split

    @field_validator("label")
    @classmethod
    def _label_is_binary(cls, value: int) -> int:
        if value not in (0, 1):
            raise ValueError("label must be 0 (dislike) or 1 (like)")
        return value


class UserIndex(BaseModel):
    """Bidirectional mapping between user_id strings and contiguous int indices."""

    id_to_index: dict[str, int]
    index_to_id: list[str]

    @model_validator(mode="after")
    def _check_bijective_and_contiguous(self) -> UserIndex:
        n = len(self.index_to_id)
        if len(self.id_to_index) != n:
            raise ValueError("id_to_index and index_to_id disagree on size")
        if sorted(self.id_to_index.values()) != list(range(n)):
            raise ValueError("indices must be contiguous starting at 0")
        for uid, idx in self.id_to_index.items():
            if idx < 0 or idx >= n or self.index_to_id[idx] != uid:
                raise ValueError(f"inconsistent mapping for user_id {uid!r}")
        return self

    @classmethod
    def from_ids(cls, user_ids: Iterable[str]) -> UserIndex:
        """Build an index from an iterable of ids, deduping by first appearance."""
        ordered: list[str] = []
        seen: set[str] = set()
        for uid in user_ids:
            if uid not in seen:
                seen.add(uid)
                ordered.append(uid)
        return cls(
            id_to_index={uid: i for i, uid in enumerate(ordered)},
            index_to_id=ordered,
        )

    def to_index(self, user_id: str) -> int:
        return self.id_to_index[user_id]

    def to_id(self, index: int) -> str:
        return self.index_to_id[index]

    def __len__(self) -> int:
        return len(self.index_to_id)


class EvaluationResult(BaseModel):
    """The metrics record an Evaluator emits for one (model, aggregation, k)."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model_name: str
    aggregation: str
    k: int
    recall_at_k: float
    hr_at_k: float
    ndcg_at_k: float
    evaluated_at: str
