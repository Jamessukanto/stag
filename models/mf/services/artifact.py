"""Build and restore ModelArtifact payloads for matrix factorization."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from core.config import Config
from core.serialization import ModelArtifact
from core.types import UserIndex

DEFAULT_L2_WEIGHT = 0.01


def build_artifact(
    *,
    config: Config,
    sampling_strategy: str,
    l2_weight: float,
    user_index: UserIndex,
    source: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
) -> ModelArtifact:
    """Serialize trained embeddings into a standard ModelArtifact."""
    return ModelArtifact(
        model_name="mf",
        sampling_strategy=sampling_strategy,
        hyperparameters={
            "embedding_dim": config.embedding_dim,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "negative_downsample_ratio": config.negative_downsample_ratio,
            "l2_weight": l2_weight,
            "optimizer": "adam",
        },
        user_index=user_index,
        source_embeddings=source.tolist(),
        target_embeddings=target.tolist(),
        extra={},
        trained_on_split="train",
    )


def restore_embeddings(
    artifact: ModelArtifact,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], UserIndex]:
    """Load embedding tables and user index from an artifact."""
    source = np.asarray(artifact.source_embeddings, dtype=np.float64)
    target = np.asarray(artifact.target_embeddings, dtype=np.float64)
    return source, target, artifact.user_index
