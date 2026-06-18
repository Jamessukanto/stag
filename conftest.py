"""Shared synthetic test fixtures inherited by every module's test suite.

Defined once here (in the Architecture chat) so the Data, Models, and Evaluation
chats all test against one canonical, deterministic dataset and artifact. This
keeps parallel sessions consistent. Do not redefine these fixture names in
module-level conftest files; extend them instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import Config
from src.core.serialization import ModelArtifact
from src.core.types import ProcessedInteraction, RawInteraction, UserIndex

# A tiny deterministic interaction set. Timestamps strictly increase so temporal
# splitting is unambiguous, and every user appears across the timeline so that
# train/val/test are all non-empty.
_USERS = ["u1", "u2", "u3", "u4"]

_RAW_ROWS: list[tuple[str, str, float, int]] = [
    ("u1", "u2", 5.0, 1000),
    ("u2", "u1", 4.0, 1001),
    ("u3", "u1", 3.0, 1002),
    ("u1", "u3", 5.0, 1003),
    ("u2", "u4", 2.0, 1004),
    ("u4", "u2", 4.0, 1005),
    ("u3", "u4", 5.0, 1006),
    ("u4", "u3", 3.0, 1007),
    ("u1", "u4", 4.0, 1008),
    ("u4", "u1", 5.0, 1009),
]

# Split assignment by timeline position: first 6 train, next 2 val, last 2 test.
_SPLIT_BY_INDEX = (
    ["train"] * 6 + ["val"] * 2 + ["test"] * 2
)


@pytest.fixture
def synthetic_raw_interactions() -> list[RawInteraction]:
    """A small deterministic list of RawInteraction, timestamp-ordered."""
    return [
        RawInteraction(user_id=u, target_id=t, rating=r, timestamp=ts)
        for (u, t, r, ts) in _RAW_ROWS
    ]


@pytest.fixture
def synthetic_processed_interactions(
    synthetic_raw_interactions: list[RawInteraction],
) -> list[ProcessedInteraction]:
    """The same interactions with train/val/test labels; all splits non-empty."""
    return [
        ProcessedInteraction(
            user_id=raw.user_id,
            target_id=raw.target_id,
            rating=raw.rating,
            timestamp=raw.timestamp,
            split=split,  # type: ignore[arg-type]
        )
        for raw, split in zip(
            synthetic_raw_interactions, _SPLIT_BY_INDEX, strict=True
        )
    ]


@pytest.fixture
def synthetic_user_index() -> UserIndex:
    """A populated UserIndex over the synthetic users."""
    return UserIndex.from_user_ids(list(_USERS))


@pytest.fixture
def synthetic_model_artifact(synthetic_user_index: UserIndex) -> ModelArtifact:
    """A valid ModelArtifact so eval/ can be tested with no model code."""
    emb_dim = 2
    n = len(synthetic_user_index)
    source = [[float(i), float(i) + 0.5] for i in range(n)]
    target = [[float(i) + 0.1, float(i) + 0.2] for i in range(n)]
    assert all(len(row) == emb_dim for row in source + target)
    return ModelArtifact(
        model_name="naive",
        sampling_strategy="random",
        hyperparameters={"embedding_dim": emb_dim, "learning_rate": 0.01, "epochs": 5},
        user_index=synthetic_user_index,
        source_embeddings=source,
        target_embeddings=target,
        trained_on_split="train",
        created_at="2026-06-19T00:00:00Z",
    )


@pytest.fixture
def tmp_artifact_path(tmp_path: Path) -> Path:
    """A tmp file location for ModelArtifact save/load round-trips."""
    return tmp_path / "artifact.json"


@pytest.fixture
def default_config(tmp_path: Path) -> Config:
    """A Config pointing data/artifact paths at tmp locations."""
    return Config(
        data_path=tmp_path / "ratings.dat",
        artifact_dir=tmp_path / "artifacts",
    )
