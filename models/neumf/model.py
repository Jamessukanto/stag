"""NeuMF preference model orchestration."""

from __future__ import annotations

from pathlib import Path

import torch
from core.config import Config
from core.serialization import ModelArtifact
from core.types import ProcessedInteraction, UserIndex
from torch.optim import Adam

from models.neumf.services.artifact import build_artifact, restore_network
from models.neumf.services.network import NeuMFNetwork, default_mlp_layers
from models.neumf.services.training import (
    compute_split_bce,
    seed_all,
    should_stop_early,
    train_epoch,
)


class NeuMFModel:
    """NeuMF with GMF + MLP branches fused through sigmoid output."""

    def __init__(
        self,
        config: Config,
        *,
        sampling_strategy: str = "random",
        mlp_layers: list[int] | None = None,
        early_stopping_patience: int = 3,
    ) -> None:
        self._config = config
        self._sampling_strategy = sampling_strategy
        self._mlp_layers = mlp_layers
        self._early_stopping_patience = early_stopping_patience
        self._user_index: UserIndex | None = None
        self._network: NeuMFNetwork | None = None
        self._artifact: ModelArtifact | None = None

    def _resolved_mlp_layers(self) -> list[int]:
        if self._mlp_layers is not None:
            return list(self._mlp_layers)
        return default_mlp_layers(self._config.embedding_dim)

    def fit(self, interactions: list[ProcessedInteraction]) -> None:
        """Train on the train split only; use val for early stopping."""
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
        val_rows = [p for p in interactions if p.split == "val"]
        if not train_rows:
            raise ValueError("no train-split interactions to fit on")

        mlp_layers = self._resolved_mlp_layers()
        seed_all(self._config.random_seed)
        self._network = NeuMFNetwork(
            n_users=len(self._user_index),
            embedding_dim=self._config.embedding_dim,
            mlp_layers=mlp_layers,
        )
        optimizer = Adam(self._network.parameters(), lr=self._config.learning_rate)

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_state: dict[str, torch.Tensor] | None = None
        for epoch in range(self._config.epochs):
            train_epoch(
                self._network,
                train_rows,
                self._user_index,
                optimizer=optimizer,
                seed=self._config.random_seed + epoch,
            )
            if val_rows:
                val_loss = compute_split_bce(self._network, val_rows, self._user_index)
                stop, best_val_loss, epochs_without_improvement = should_stop_early(
                    best_val_loss=best_val_loss,
                    current_val_loss=val_loss,
                    epochs_without_improvement=epochs_without_improvement,
                    patience=self._early_stopping_patience,
                )
                if val_loss <= best_val_loss:
                    best_state = {
                        key: value.detach().clone()
                        for key, value in self._network.state_dict().items()
                    }
                if stop:
                    break

        if best_state is not None:
            self._network.load_state_dict(best_state)

        assert self._network is not None
        assert self._user_index is not None
        self._artifact = build_artifact(
            config=self._config,
            sampling_strategy=self._sampling_strategy,
            mlp_layers=mlp_layers,
            early_stopping_patience=self._early_stopping_patience,
            user_index=self._user_index,
            network=self._network,
        )

    def directional_score(self, user_u: str, target_v: str) -> float:
        """Return s(u->v) from the learned NeuMF forward pass."""
        if self._network is None or self._user_index is None:
            raise RuntimeError("model must be fit or loaded before scoring")
        u_idx = torch.tensor(self._user_index.to_index(user_u), dtype=torch.long)
        v_idx = torch.tensor(self._user_index.to_index(target_v), dtype=torch.long)
        self._network.eval()
        with torch.no_grad():
            score = self._network.forward_pair(u_idx, v_idx)
        return float(score)

    def save(self, path: Path) -> None:
        """Write the trained ModelArtifact to disk."""
        if self._artifact is None:
            raise RuntimeError("model must be fit before save")
        self._artifact.save(path)

    def load(self, path: Path) -> NeuMFModel:
        """Restore a model from a saved ModelArtifact."""
        artifact = ModelArtifact.load(path)
        network, user_index = restore_network(artifact)
        hyper = artifact.hyperparameters
        config = Config(
            embedding_dim=int(hyper["embedding_dim"]),
            learning_rate=float(hyper["learning_rate"]),
            epochs=int(hyper["epochs"]),
            negative_downsample_ratio=float(hyper["negative_downsample_ratio"]),
            random_seed=self._config.random_seed,
        )
        loaded = NeuMFModel(
            config,
            sampling_strategy=artifact.sampling_strategy,
            mlp_layers=list(hyper.get("mlp_layers", default_mlp_layers(config.embedding_dim))),
            early_stopping_patience=int(hyper.get("early_stopping_patience", 3)),
        )
        loaded._user_index = user_index
        loaded._network = network
        loaded._artifact = artifact
        return loaded
