"""Tests for MatrixFactorizationModel public surface and artifact round-trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from core.config import Config
from core.interfaces import PreferenceModel
from core.scoring import verify_scorer_matches_directional
from core.serialization import ModelArtifact
from core.types import ProcessedInteraction, UserIndex
from models.mf.model import MatrixFactorizationModel
from models.mf.services.artifact import build_artifact

_GOLDEN_PAIRS: list[tuple[str, str]] = [
    ("u1", "u2"),
    ("u2", "u1"),
    ("u3", "u4"),
]


class TestProtocolConformance:
    def test_is_a_preference_model(self, mf_config: Config) -> None:
        model = MatrixFactorizationModel(mf_config)
        assert isinstance(model, PreferenceModel)


class TestFit:
    def test_uses_train_split_only(
        self,
        mf_config: Config,
        processed_interactions: list[ProcessedInteraction],
    ) -> None:
        model = MatrixFactorizationModel(mf_config)
        model.fit(processed_interactions)
        assert model._artifact is not None
        assert model._artifact.trained_on_split == "train"

    def test_directional_score_matches_dot_product_after_fit(
        self,
        mf_config: Config,
        processed_interactions: list[ProcessedInteraction],
    ) -> None:
        model = MatrixFactorizationModel(mf_config)
        model.fit(processed_interactions)
        assert model._source_embeddings is not None
        assert model._target_embeddings is not None
        assert model._user_index is not None
        u_idx = model._user_index.to_index("u1")
        v_idx = model._user_index.to_index("u2")
        expected = float(
            np.dot(model._source_embeddings[u_idx], model._target_embeddings[v_idx])
        )
        assert model.directional_score("u1", "u2") == pytest.approx(expected)


class TestSaveLoad:
    def test_round_trip_preserves_directional_scores(
        self,
        mf_config: Config,
        processed_interactions: list[ProcessedInteraction],
        tmp_path: Path,
    ) -> None:
        model = MatrixFactorizationModel(mf_config)
        model.fit(processed_interactions)
        pairs = [("u1", "u2"), ("u3", "u4"), ("u5", "u2")]
        before = {pair: model.directional_score(*pair) for pair in pairs}

        artifact_path = tmp_path / "mf.json"
        model.save(artifact_path)
        loaded = model.load(artifact_path)
        for pair in pairs:
            assert loaded.directional_score(*pair) == pytest.approx(before[pair], rel=1e-5)

    def test_golden_scorer_matches_after_save_load(
        self,
        mf_config: Config,
        processed_interactions: list[ProcessedInteraction],
        tmp_path: Path,
    ) -> None:
        model = MatrixFactorizationModel(mf_config)
        model.fit(processed_interactions)
        artifact_path = tmp_path / "mf.json"
        model.save(artifact_path)
        loaded = model.load(artifact_path)
        artifact = ModelArtifact.load(artifact_path)
        verify_scorer_matches_directional(
            artifact,
            loaded.directional_score,
            _GOLDEN_PAIRS,
        )

    def test_artifact_has_expected_fields(
        self,
        mf_config: Config,
        processed_interactions: list[ProcessedInteraction],
        user_index: UserIndex,
    ) -> None:
        model = MatrixFactorizationModel(mf_config, sampling_strategy="random", l2_weight=0.01)
        model.fit(processed_interactions)
        assert model._source_embeddings is not None
        assert model._target_embeddings is not None
        assert model._user_index is not None
        artifact = build_artifact(
            config=mf_config,
            sampling_strategy="random",
            l2_weight=0.01,
            user_index=model._user_index,
            source=model._source_embeddings,
            target=model._target_embeddings,
        )
        assert artifact.model_name == "mf"
        assert artifact.sampling_strategy == "random"
        assert artifact.extra == {}
        assert "score_program" not in artifact.extra
        assert artifact.hyperparameters["optimizer"] == "adam"
        assert artifact.hyperparameters["l2_weight"] == 0.01
        assert len(artifact.source_embeddings) == len(user_index)
        assert len(artifact.target_embeddings) == len(user_index)
