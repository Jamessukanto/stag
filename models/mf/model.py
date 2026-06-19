"""Matrix-factorization preference model orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from core.config import Config
from core.serialization import ModelArtifact
from core.types import ProcessedInteraction, UserIndex

from models.mf.services.artifact import DEFAULT_L2_WEIGHT, build_artifact, restore_embeddings
from models.mf.services.embeddings import directional_dot, init_embeddings
from models.mf.services.training import AdamState, train_epoch


class MatrixFactorizationModel:
    """Weighted MF with directional score s(u->v) = p_u^T q_v."""

    def __init__(
        self,
        config: Config,
        *,
        sampling_strategy: str = "random",
        l2_weight: float = DEFAULT_L2_WEIGHT,
    ) -> None:
        self._config = config
        self._sampling_strategy = sampling_strategy
        self._l2_weight = l2_weight
        self._user_index: UserIndex | None = None
        self._source_embeddings: npt.NDArray[np.float64] | None = None
        self._target_embeddings: npt.NDArray[np.float64] | None = None
        self._artifact: ModelArtifact | None = None

    def fit(self, interactions: list[ProcessedInteraction]) -> None:
        """Train on the train split only; build UserIndex from all users present."""
        if not interactions:
            raise ValueError("interactions must not be empty")

        user_ids: list[str] = []
        seen: set[str] = set()
        for interaction in interactions:
            for uid in (interaction.user_id, interaction.target_id):
                if uid not in seen:
                    seen.add(uid)
                    user_ids.append(uid)

        self._user_index = UserIndex.from_ids(user_ids)
        train_rows = [p for p in interactions if p.split == "train"]
        if not train_rows:
            raise ValueError("no train-split interactions to fit on")

        rng = np.random.default_rng(self._config.random_seed)
        source, target = init_embeddings(
            n_users=len(self._user_index),
            dim=self._config.embedding_dim,
            rng=rng,
        )
        adam_states: dict[tuple[str, int], AdamState] = {}

        for epoch in range(self._config.epochs):
            epoch_rng = np.random.default_rng(self._config.random_seed + epoch)
            train_epoch(
                train_rows,
                source,
                target,
                self._user_index,
                learning_rate=self._config.learning_rate,
                l2_weight=self._l2_weight,
                rng=epoch_rng,
                adam_states=adam_states,
            )

        self._source_embeddings = source
        self._target_embeddings = target
        self._artifact = build_artifact(
            config=self._config,
            sampling_strategy=self._sampling_strategy,
            l2_weight=self._l2_weight,
            user_index=self._user_index,
            source=source,
            target=target,
        )

    def directional_score(self, user_u: str, target_v: str) -> float:
        """Return s(u->v) = p_u^T q_v for learned embeddings."""
        if (
            self._user_index is None
            or self._source_embeddings is None
            or self._target_embeddings is None
        ):
            raise RuntimeError("model must be fit or loaded before scoring")
        u_idx = self._user_index.to_index(user_u)
        v_idx = self._user_index.to_index(target_v)
        return directional_dot(self._source_embeddings, self._target_embeddings, u_idx, v_idx)

    def save(self, path: Path) -> None:
        """Write the trained ModelArtifact to disk."""
        if self._artifact is None:
            raise RuntimeError("model must be fit before save")
        self._artifact.save(path)

    def load(self, path: Path) -> MatrixFactorizationModel:
        """Restore a model from a saved ModelArtifact."""
        artifact = ModelArtifact.load(path)
        if artifact.model_name != "mf":
            raise ValueError(f"expected model_name 'mf', got {artifact.model_name!r}")

        source, target, user_index = restore_embeddings(artifact)
        hyper = artifact.hyperparameters
        config = Config(
            embedding_dim=int(hyper["embedding_dim"]),
            learning_rate=float(hyper["learning_rate"]),
            epochs=int(hyper["epochs"]),
            negative_downsample_ratio=float(hyper["negative_downsample_ratio"]),
            random_seed=self._config.random_seed,
        )
        loaded = MatrixFactorizationModel(
            config,
            sampling_strategy=artifact.sampling_strategy,
            l2_weight=float(hyper.get("l2_weight", DEFAULT_L2_WEIGHT)),
        )
        loaded._user_index = user_index
        loaded._source_embeddings = source
        loaded._target_embeddings = target
        loaded._artifact = artifact
        return loaded
