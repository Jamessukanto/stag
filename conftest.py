"""Shared, deterministic synthetic fixtures for the reciprocal-rec project.

Every downstream chat (data, models, eval) inherits these fixtures so they all
build against one consistent toy dataset and never invent divergent fixtures.

The dataset is hand-designed to exercise the cases the system cares about:
- a mutual match (both users rate each other >= 7): (u1, u2) and (u2, u3)
- a one-sided pair (A likes B, B dislikes A): u1 -> u3, u5 -> u4, u2 -> u5
- explicit dislikes (rating < 7)
- coverage across the train / val / test splits
"""

from __future__ import annotations

import pytest

from core.serialization import ModelArtifact
from core.types import (
    ProcessedInteraction,
    RawInteraction,
    UserIndex,
)

# (rater_id, target_id, rating 1-10, split) - frozen, deterministic.
_SYNTHETIC_RECORDS: list[tuple[str, str, int, str]] = [
    ("u1", "u2", 9, "train"),   # mutual match with u2 -> u1
    ("u2", "u1", 8, "train"),   # mutual match with u1 -> u2
    ("u1", "u3", 10, "train"),  # one-sided: u3 dislikes u1
    ("u3", "u1", 2, "train"),   # one-sided dislike back
    ("u2", "u3", 7, "train"),   # mutual match with u3 -> u2
    ("u3", "u2", 9, "train"),   # mutual match with u2 -> u3
    ("u4", "u5", 6, "train"),   # dislike (below threshold)
    ("u5", "u4", 8, "val"),     # one-sided like
    ("u1", "u4", 5, "val"),     # dislike
    ("u4", "u1", 1, "test"),    # dislike
    ("u5", "u2", 7, "test"),    # like
    ("u2", "u5", 4, "test"),    # one-sided dislike back
]

# Stable user ordering -> contiguous indices. Order follows first appearance.
_USER_IDS: list[str] = ["u1", "u2", "u3", "u4", "u5"]

LIKE_THRESHOLD = 7


def _binarize(rating: int) -> int:
    return 1 if rating >= LIKE_THRESHOLD else 0


@pytest.fixture
def user_ids() -> list[str]:
    return list(_USER_IDS)


@pytest.fixture
def user_index() -> UserIndex:
    return UserIndex.from_ids(_USER_IDS)


@pytest.fixture
def raw_interactions() -> list[RawInteraction]:
    return [
        RawInteraction(user_id=u, target_id=v, rating=r)
        for (u, v, r, _split) in _SYNTHETIC_RECORDS
    ]


@pytest.fixture
def processed_interactions() -> list[ProcessedInteraction]:
    return [
        ProcessedInteraction(
            user_id=u,
            target_id=v,
            label=_binarize(r),
            split=split,  # type: ignore[arg-type]
        )
        for (u, v, r, split) in _SYNTHETIC_RECORDS
    ]


@pytest.fixture
def sample_artifact(user_index: UserIndex) -> ModelArtifact:
    """An 'mf' artifact: no score_program, so scoring uses the implicit-dot default."""
    source = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [-1.0, 0.5],
        [0.5, -0.5],
    ]
    target = [
        [0.5, 0.5],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.25, 0.75],
        [-0.5, 1.0],
    ]
    return ModelArtifact(
        model_name="mf",
        sampling_strategy="random",
        hyperparameters={"embedding_dim": 2, "learning_rate": 0.01, "epochs": 5},
        user_index=user_index,
        source_embeddings=source,
        target_embeddings=target,
        extra={},
        trained_on_split="train",
        created_at="2026-01-01T00:00:00+00:00",
    )


@pytest.fixture
def sample_neumf_artifact(user_index: UserIndex) -> ModelArtifact:
    """A 'neumf' artifact carrying a self-describing score_program in extra.

    GMF branch uses source/target embeddings (element-wise product); the MLP
    branch uses separate embedding tables and dense weights stored in extra.
    """
    source = [  # GMF source table (p^G)
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
        [-0.5, 0.5],
        [0.25, -0.25],
    ]
    target = [  # GMF target table (q^G)
        [0.5, 0.5],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.75, 0.25],
        [-0.25, 0.5],
    ]
    extra = {
        "mlp_source": [
            [0.1, 0.2],
            [0.3, -0.1],
            [-0.2, 0.4],
            [0.0, 0.5],
            [0.2, 0.2],
        ],
        "mlp_target": [
            [0.2, 0.1],
            [-0.3, 0.2],
            [0.4, 0.0],
            [0.1, -0.2],
            [0.5, 0.3],
        ],
        # dense layer: input dim 4 (concat of two dim-2 vectors) -> output dim 2
        "W1": [[0.1, -0.2], [0.3, 0.0], [-0.1, 0.2], [0.2, 0.1]],
        "b1": [0.05, -0.05],
        # output layer: input dim 4 (concat gmf dim2 + mlp dim2) -> output dim 1
        "Wout": [[0.5], [-0.3], [0.2], [0.4]],
        "bout": [0.1],
        "score_program": [
            {"op": "lookup", "out": "g_u", "table": "source_embeddings", "index": "u"},
            {"op": "lookup", "out": "g_v", "table": "target_embeddings", "index": "v"},
            {"op": "multiply", "out": "gmf", "a": "g_u", "b": "g_v"},
            {"op": "lookup", "out": "m_u", "table": "mlp_source", "index": "u"},
            {"op": "lookup", "out": "m_v", "table": "mlp_target", "index": "v"},
            {"op": "concat", "out": "mlp_in", "inputs": ["m_u", "m_v"]},
            {"op": "dense", "out": "mlp_h", "input": "mlp_in", "weight": "W1", "bias": "b1"},
            {"op": "relu", "out": "mlp_act", "input": "mlp_h"},
            {"op": "concat", "out": "fused", "inputs": ["gmf", "mlp_act"]},
            {"op": "dense", "out": "logit", "input": "fused", "weight": "Wout", "bias": "bout"},
            {"op": "sigmoid", "out": "score", "input": "logit"},
        ],
    }
    return ModelArtifact(
        model_name="neumf",
        sampling_strategy="popularity_biased",
        hyperparameters={"embedding_dim": 2, "mlp_layers": [4, 2], "epochs": 5},
        user_index=user_index,
        source_embeddings=source,
        target_embeddings=target,
        extra=extra,
        trained_on_split="train",
        created_at="2026-01-01T00:00:00+00:00",
    )
