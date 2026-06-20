"""Tests for MF loss computation and training loop mechanics."""

from __future__ import annotations

import numpy as np
import pytest
from core.config import Config
from core.types import ProcessedInteraction, UserIndex
from models.mf.model import MatrixFactorizationModel
from models.mf.services.embeddings import init_embeddings
from models.mf.services.loss import squared_loss_and_grad
from models.mf.services.training import (
    AdamState,
    compute_epoch_loss,
    train_epoch,
)


class TestSquaredLoss:
    def test_zero_when_prediction_matches_label(self) -> None:
        loss, grad = squared_loss_and_grad(score=1.0, label=1.0)
        assert loss == pytest.approx(0.0)
        assert grad == pytest.approx(0.0)

    def test_gradient_sign(self) -> None:
        _, grad = squared_loss_and_grad(score=0.0, label=1.0)
        assert grad == pytest.approx(-2.0)


def _train_interactions(
    processed_interactions: list[ProcessedInteraction],
) -> list[ProcessedInteraction]:
    return [p for p in processed_interactions if p.split == "train"]


class TestTrainEpoch:
    def test_loss_decreases_over_epochs(
        self,
        mf_config: Config,
        processed_interactions: list[ProcessedInteraction],
        user_index: UserIndex,
    ) -> None:
        train_rows = _train_interactions(processed_interactions)
        rng = np.random.default_rng(mf_config.random_seed)
        source, target = init_embeddings(
            n_users=len(user_index), dim=mf_config.embedding_dim, rng=rng
        )
        adam_states: dict[tuple[str, int], AdamState] = {}
        initial_loss = compute_epoch_loss(
            train_rows, source, target, user_index, l2_weight=0.01
        )
        for epoch in range(mf_config.epochs):
            epoch_rng = np.random.default_rng(mf_config.random_seed + epoch)
            train_epoch(
                train_rows,
                source,
                target,
                user_index,
                learning_rate=mf_config.learning_rate,
                l2_weight=0.01,
                rng=epoch_rng,
                adam_states=adam_states,
            )
        final_loss = compute_epoch_loss(
            train_rows, source, target, user_index, l2_weight=0.01
        )
        assert final_loss < initial_loss


class TestDeterministicTraining:
    def test_same_seed_produces_identical_embeddings(
        self,
        mf_config: Config,
        processed_interactions: list[ProcessedInteraction],
    ) -> None:
        model_a = MatrixFactorizationModel(mf_config)
        model_a.fit(processed_interactions)
        model_b = MatrixFactorizationModel(mf_config)
        model_b.fit(processed_interactions)
        assert model_a._source_embeddings is not None
        assert model_b._source_embeddings is not None
        assert model_a._target_embeddings is not None
        assert model_b._target_embeddings is not None
        np.testing.assert_array_equal(model_a._source_embeddings, model_b._source_embeddings)
        np.testing.assert_array_equal(model_a._target_embeddings, model_b._target_embeddings)
