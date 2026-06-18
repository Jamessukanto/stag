"""Shared, stable type contracts for the reciprocal-rec project.

These types are the single source of truth consumed by ``data/``, ``models/``,
and ``eval/``. Downstream modules import them and MUST NOT redefine them.

This module contains contracts only - no business logic.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SplitLabel = Literal["train", "val", "test"]
ModelName = Literal["naive", "ref"]
SamplingStrategy = Literal["random", "popularity_biased"]


class RawInteraction(BaseModel):
    """A single directional rating as parsed from the raw dataset."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    target_id: str
    rating: float
    timestamp: int


class ProcessedInteraction(BaseModel):
    """A RawInteraction enriched with its train/val/test split label."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    target_id: str
    rating: float
    timestamp: int
    split: SplitLabel


class UserIndex(BaseModel):
    """Bidirectional mapping between user_id strings and dense integer indices.

    The serialized form is a single ordered ``users`` list where the position of
    each user_id is its integer index. This ordering is the contract that aligns
    every embedding row (in ModelArtifact) to a user. Indices are dense
    ``0..n-1`` with no gaps and no duplicate user_ids.
    """

    users: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_unique(self) -> UserIndex:
        if len(set(self.users)) != len(self.users):
            raise ValueError("UserIndex.users must not contain duplicate user_ids")
        return self

    @classmethod
    def from_user_ids(cls, user_ids: list[str]) -> UserIndex:
        """Build a UserIndex preserving first-seen order of the given ids."""
        seen: dict[str, None] = {}
        for uid in user_ids:
            if uid not in seen:
                seen[uid] = None
        return cls(users=list(seen.keys()))

    @property
    def user_to_index(self) -> dict[str, int]:
        return {uid: i for i, uid in enumerate(self.users)}

    @property
    def index_to_user(self) -> dict[int, str]:
        return dict(enumerate(self.users))

    def get_index(self, user_id: str) -> int:
        """Return the integer index for a user_id, or raise KeyError."""
        return self.user_to_index[user_id]

    def get_user(self, index: int) -> str:
        """Return the user_id for an integer index, or raise IndexError."""
        return self.users[index]

    def __contains__(self, user_id: object) -> bool:
        return user_id in set(self.users)

    def __len__(self) -> int:
        return len(self.users)


class EvaluationResult(BaseModel):
    """The result of evaluating one model artifact at a given K."""

    # protected_namespaces=() so the ``model_name`` field does not collide with
    # Pydantic's reserved ``model_`` attribute namespace.
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model_name: ModelName
    sampling_strategy: SamplingStrategy
    k: int
    reciprocal_precision_at_k: float
    mutual_hit_rate: float
    evaluated_at: str
