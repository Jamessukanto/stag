"""The ModelArtifact serialization contract.

A ModelArtifact is the only thing a preference model emits and the only thing
evaluation consumes. It decouples models from evaluation: evaluation reads the
artifact from disk and reconstructs directional scores via
``core.scoring.reconstruct_scorer`` WITHOUT importing any model module.

Decoupling rule (frozen):
- ``source_embeddings`` (p_u) and ``target_embeddings`` (q_u) hold the primary
  directional embedding tables.
- ``extra`` is a generic bag for any additional model-specific tensors (e.g. a
  NeuMF MLP's weights and its separate branch embedding tables).
- ``extra["score_program"]`` optionally carries a self-describing compute graph
  (see ``core.scoring``). If absent, the directional score defaults to the dot
  product ``p_u . q_v`` (matrix factorization). This is what lets evaluation
  score any model generically, never branching on ``model_name``.

The JSON encoding is intentionally human-readable.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.types import UserIndex


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class ModelArtifact(BaseModel):
    """The serialized output every preference model emits."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    sampling_strategy: str
    hyperparameters: dict[str, Any]
    user_index: UserIndex
    source_embeddings: list[list[float]]
    target_embeddings: list[list[float]]
    extra: dict[str, Any] = Field(default_factory=dict)
    trained_on_split: str = "train"
    created_at: str = Field(default_factory=_utc_now_iso)

    @field_validator("trained_on_split")
    @classmethod
    def _must_train_on_train(cls, value: str) -> str:
        if value != "train":
            raise ValueError('trained_on_split must be "train"')
        return value

    def save(self, path: Path) -> None:
        """Write the artifact as human-readable JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> ModelArtifact:
        """Read an artifact previously written by :meth:`save`."""
        data = json.loads(Path(path).read_text())
        return cls.model_validate(data)
