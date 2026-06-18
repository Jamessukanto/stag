"""The ModelArtifact serialization contract.

This is the ONLY channel between ``models/`` (which writes artifacts) and
``eval/`` (which reads them). The on-disk format is human-readable JSON.

Row ``i`` of ``source_embeddings`` and ``target_embeddings`` corresponds to
``user_index.users[i]``. This alignment is part of the contract and must not be
broken by any producer or consumer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.types import ModelName, SamplingStrategy, UserIndex

CURRENT_SCHEMA_VERSION = 1


class ModelArtifact(BaseModel):
    """Serialization schema every model must output.

    ``schema_version`` lets future format changes be detected rather than
    silently misread by ``eval/``.
    """

    # protected_namespaces=() so the ``model_name`` field does not collide with
    # Pydantic's reserved ``model_`` attribute namespace.
    model_config = ConfigDict(protected_namespaces=())

    schema_version: int = Field(default=CURRENT_SCHEMA_VERSION)
    model_name: ModelName
    sampling_strategy: SamplingStrategy
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    user_index: UserIndex
    source_embeddings: list[list[float]]
    target_embeddings: list[list[float]]
    trained_on_split: str
    created_at: str

    @model_validator(mode="after")
    def _validate_structure(self) -> ModelArtifact:
        n_users = len(self.user_index)
        if len(self.source_embeddings) != n_users:
            raise ValueError(
                "source_embeddings rows "
                f"({len(self.source_embeddings)}) must equal number of users "
                f"({n_users})"
            )
        if len(self.target_embeddings) != n_users:
            raise ValueError(
                "target_embeddings rows "
                f"({len(self.target_embeddings)}) must equal number of users "
                f"({n_users})"
            )
        self._validate_consistent_dim(self.source_embeddings, "source_embeddings")
        self._validate_consistent_dim(self.target_embeddings, "target_embeddings")
        if self.trained_on_split != "train":
            raise ValueError(
                f"trained_on_split must be 'train', got '{self.trained_on_split}'"
            )
        return self

    @staticmethod
    def _validate_consistent_dim(matrix: list[list[float]], name: str) -> None:
        if not matrix:
            return
        dim = len(matrix[0])
        for row_idx, row in enumerate(matrix):
            if len(row) != dim:
                raise ValueError(
                    f"{name} row {row_idx} has length {len(row)}, expected {dim} "
                    "(all embedding rows must share the same dimension)"
                )

    def save(self, path: Path) -> None:
        """Write the artifact to ``path`` as indented, human-readable JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ModelArtifact:
        """Read and validate a ModelArtifact from ``path``."""
        path = Path(path)
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
