"""Tests for the stable contracts owned by the Architecture chat."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.config import Config
from src.core.serialization import CURRENT_SCHEMA_VERSION, ModelArtifact
from src.core.types import (
    EvaluationResult,
    ProcessedInteraction,
    RawInteraction,
    UserIndex,
)


class TestLiteralEnforcement:
    def test_processed_interaction_rejects_unknown_split(self) -> None:
        with pytest.raises(ValidationError):
            ProcessedInteraction(
                user_id="u1",
                target_id="u2",
                rating=1.0,
                timestamp=1,
                split="holdout",  # type: ignore[arg-type]
            )

    def test_raw_interaction_accepts_valid_fields(self) -> None:
        raw = RawInteraction(user_id="u1", target_id="u2", rating=3.5, timestamp=10)
        assert raw.rating == 3.5
        assert raw.timestamp == 10

    def test_evaluation_result_rejects_unknown_model_name(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationResult(
                model_name="mystery",  # type: ignore[arg-type]
                sampling_strategy="random",
                k=10,
                reciprocal_precision_at_k=0.5,
                mutual_hit_rate=0.5,
                evaluated_at="2026-06-19T00:00:00Z",
            )


class TestUserIndex:
    def test_bidirectional_mapping(self) -> None:
        idx = UserIndex.from_user_ids(["a", "b", "c"])
        assert idx.get_index("a") == 0
        assert idx.get_index("c") == 2
        assert idx.get_user(1) == "b"
        assert "a" in idx
        assert "z" not in idx
        assert len(idx) == 3

    def test_from_user_ids_deduplicates_preserving_order(self) -> None:
        idx = UserIndex.from_user_ids(["b", "a", "b", "c", "a"])
        assert idx.users == ["b", "a", "c"]

    def test_rejects_duplicate_users(self) -> None:
        with pytest.raises(ValidationError):
            UserIndex(users=["a", "a"])

    def test_json_round_trip(self) -> None:
        idx = UserIndex.from_user_ids(["x", "y"])
        restored = UserIndex.model_validate_json(idx.model_dump_json())
        assert restored.users == ["x", "y"]
        assert restored.user_to_index == {"x": 0, "y": 1}


class TestModelArtifact:
    def _valid_kwargs(self) -> dict[str, object]:
        return {
            "model_name": "ref",
            "sampling_strategy": "popularity_biased",
            "hyperparameters": {"embedding_dim": 2},
            "user_index": UserIndex.from_user_ids(["a", "b"]),
            "source_embeddings": [[0.1, 0.2], [0.3, 0.4]],
            "target_embeddings": [[0.5, 0.6], [0.7, 0.8]],
            "trained_on_split": "train",
            "created_at": "2026-06-19T00:00:00Z",
        }

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        artifact = ModelArtifact(**self._valid_kwargs())
        path = tmp_path / "artifact.json"
        artifact.save(path)
        loaded = ModelArtifact.load(path)
        assert loaded == artifact
        assert loaded.schema_version == CURRENT_SCHEMA_VERSION

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        artifact = ModelArtifact(**self._valid_kwargs())
        path = tmp_path / "nested" / "dir" / "artifact.json"
        artifact.save(path)
        assert path.exists()

    def test_artifact_json_is_human_readable(self, tmp_path: Path) -> None:
        artifact = ModelArtifact(**self._valid_kwargs())
        path = tmp_path / "artifact.json"
        artifact.save(path)
        text = path.read_text(encoding="utf-8")
        assert "\n" in text
        assert '"user_index"' in text

    def test_rejects_embedding_row_count_mismatch(self) -> None:
        kwargs = self._valid_kwargs()
        kwargs["source_embeddings"] = [[0.1, 0.2]]  # only 1 row for 2 users
        with pytest.raises(ValidationError):
            ModelArtifact(**kwargs)

    def test_rejects_inconsistent_embedding_dim(self) -> None:
        kwargs = self._valid_kwargs()
        kwargs["target_embeddings"] = [[0.5, 0.6], [0.7]]  # ragged
        with pytest.raises(ValidationError):
            ModelArtifact(**kwargs)

    def test_rejects_non_train_split(self) -> None:
        kwargs = self._valid_kwargs()
        kwargs["trained_on_split"] = "val"
        with pytest.raises(ValidationError):
            ModelArtifact(**kwargs)

    def test_rejects_unknown_model_name(self) -> None:
        kwargs = self._valid_kwargs()
        kwargs["model_name"] = "deep"
        with pytest.raises(ValidationError):
            ModelArtifact(**kwargs)


class TestConfig:
    def test_default_ratios_sum_to_one(self) -> None:
        cfg = Config()
        assert cfg.train_ratio + cfg.val_ratio + cfg.test_ratio == pytest.approx(1.0)

    def test_rejects_ratios_not_summing_to_one(self) -> None:
        with pytest.raises(ValidationError):
            Config(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)

    def test_rejects_ratio_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            Config(train_ratio=1.5, val_ratio=-0.25, test_ratio=-0.25)

    def test_rejects_empty_k_values(self) -> None:
        with pytest.raises(ValidationError):
            Config(k_values=[])

    def test_rejects_nonpositive_k_values(self) -> None:
        with pytest.raises(ValidationError):
            Config(k_values=[5, 0, 10])


class TestSharedFixtures:
    def test_processed_fixture_has_all_splits(
        self, synthetic_processed_interactions: list[ProcessedInteraction]
    ) -> None:
        splits = {i.split for i in synthetic_processed_interactions}
        assert splits == {"train", "val", "test"}

    def test_artifact_fixture_round_trips(
        self, synthetic_model_artifact: ModelArtifact, tmp_artifact_path: Path
    ) -> None:
        synthetic_model_artifact.save(tmp_artifact_path)
        assert ModelArtifact.load(tmp_artifact_path) == synthetic_model_artifact

    def test_default_config_fixture_is_valid(self, default_config: Config) -> None:
        assert default_config.embedding_dim > 0
