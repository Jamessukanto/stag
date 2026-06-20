"""Tests for NeuMF BCE loss and training mechanics."""

from __future__ import annotations

import pytest
import torch
from core.config import Config
from core.types import ProcessedInteraction, UserIndex
from models.neumf.services.network import NeuMFNetwork, default_mlp_layers
from models.neumf.services.training import (
    compute_split_bce,
    seed_all,
    train_epoch,
)


def _train_interactions(
    processed_interactions: list[ProcessedInteraction],
) -> list[ProcessedInteraction]:
    return [p for p in processed_interactions if p.split == "train"]


class TestBCELoss:
    def test_zero_when_prediction_matches_label(self) -> None:
        from models.neumf.services.loss import bce_loss

        pred = torch.tensor(1.0)
        label = torch.tensor(1.0)
        loss = bce_loss(pred, label)
        assert float(loss) == pytest.approx(0.0, abs=1e-7)


class TestTrainEpoch:
    def test_loss_decreases_over_epochs(
        self,
        neumf_config: Config,
        processed_interactions: list[ProcessedInteraction],
        user_index: UserIndex,
    ) -> None:
        train_rows = _train_interactions(processed_interactions)
        seed_all(neumf_config.random_seed)
        network = NeuMFNetwork(
            n_users=len(user_index),
            embedding_dim=neumf_config.embedding_dim,
            mlp_layers=default_mlp_layers(neumf_config.embedding_dim),
        )
        optimizer = torch.optim.Adam(network.parameters(), lr=neumf_config.learning_rate)

        initial_loss = compute_split_bce(network, train_rows, user_index)
        for epoch in range(neumf_config.epochs):
            train_epoch(
                network,
                train_rows,
                user_index,
                optimizer=optimizer,
                seed=neumf_config.random_seed + epoch,
            )
        final_loss = compute_split_bce(network, train_rows, user_index)
        assert final_loss < initial_loss


class TestDeterministicTraining:
    def test_same_seed_produces_identical_weights(
        self,
        neumf_config: Config,
        processed_interactions: list[ProcessedInteraction],
        user_index: UserIndex,
    ) -> None:
        train_rows = _train_interactions(processed_interactions)

        def _train_once() -> dict[str, torch.Tensor]:
            seed_all(neumf_config.random_seed)
            network = NeuMFNetwork(
                n_users=len(user_index),
                embedding_dim=neumf_config.embedding_dim,
                mlp_layers=default_mlp_layers(neumf_config.embedding_dim),
            )
            optimizer = torch.optim.Adam(network.parameters(), lr=neumf_config.learning_rate)
            for epoch in range(neumf_config.epochs):
                train_epoch(
                    network,
                    train_rows,
                    user_index,
                    optimizer=optimizer,
                    seed=neumf_config.random_seed + epoch,
                )
            return {name: param.detach().clone() for name, param in network.named_parameters()}

        weights_a = _train_once()
        weights_b = _train_once()
        for name in weights_a:
            assert torch.allclose(weights_a[name], weights_b[name])
