"""Contract tests for ModelArtifact (de)serialization."""

from __future__ import annotations

from pathlib import Path

import pytest
from core.serialization import ModelArtifact
from core.types import UserIndex
from pydantic import ValidationError


def _make_artifact(**overrides: object) -> ModelArtifact:
    base: dict[str, object] = dict(
        model_name="mf",
        sampling_strategy="random",
        hyperparameters={"embedding_dim": 2},
        user_index=UserIndex.from_ids(["a", "b"]),
        source_embeddings=[[0.1, 0.2], [0.3, 0.4]],
        target_embeddings=[[0.5, 0.6], [0.7, 0.8]],
        extra={},
        trained_on_split="train",
        created_at="2026-01-01T00:00:00+00:00",
    )
    base.update(overrides)
    return ModelArtifact(**base)  # type: ignore[arg-type]

class TestModelArtifactSchema:
    def test_constructs_with_full_schema(self) -> None:
        art = _make_artifact()
        assert art.model_name == "mf"
        assert art.trained_on_split == "train"
        assert art.source_embeddings == [[0.1, 0.2], [0.3, 0.4]]

    def test_trained_on_split_must_be_train(self) -> None:
        with pytest.raises(ValidationError):
            _make_artifact(trained_on_split="test")

    def test_extra_defaults_to_empty_dict(self) -> None:
        art = ModelArtifact(
            model_name="mf",
            sampling_strategy="random",
            hyperparameters={},
            user_index=UserIndex.from_ids(["a"]),
            source_embeddings=[[1.0]],
            target_embeddings=[[1.0]],
            created_at="2026-01-01T00:00:00+00:00",
        )
        assert art.extra == {}


class TestModelArtifactRoundTrip:
    def test_save_then_load_round_trips_losslessly(self, tmp_path: Path) -> None:
        program = [{"op": "dot", "out": "s", "a": "x", "b": "y"}]
        art = _make_artifact(extra={"score_program": program})
        path = tmp_path / "artifact.json"
        art.save(path)
        loaded = ModelArtifact.load(path)
        assert loaded == art

    def test_saved_file_is_human_readable_json(self, tmp_path: Path) -> None:
        art = _make_artifact()
        path = tmp_path / "artifact.json"
        art.save(path)
        text = path.read_text()
        assert '"model_name"' in text
        assert "mf" in text
        # JSON, not a binary pickle: should be parseable as text with newlines.
        assert "\n" in text

    def test_user_index_survives_round_trip(self, tmp_path: Path) -> None:
        art = _make_artifact()
        path = tmp_path / "artifact.json"
        art.save(path)
        loaded = ModelArtifact.load(path)
        assert loaded.user_index.to_index("b") == art.user_index.to_index("b")
