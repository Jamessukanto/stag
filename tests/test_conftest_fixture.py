"""Tests that the shared synthetic fixture yields the documented records.

Downstream chats rely on these guarantees, so they are pinned here.
"""

from __future__ import annotations

from core.serialization import ModelArtifact
from core.types import ProcessedInteraction, RawInteraction, UserIndex


def test_raw_interactions_shape(raw_interactions: list[RawInteraction]) -> None:
    assert len(raw_interactions) == 12
    assert all(1 <= ri.rating <= 10 for ri in raw_interactions)


def test_processed_labels_match_binarization(
    raw_interactions: list[RawInteraction],
    processed_interactions: list[ProcessedInteraction],
) -> None:
    assert len(processed_interactions) == len(raw_interactions)
    for raw, proc in zip(raw_interactions, processed_interactions, strict=True):
        expected = 1 if raw.rating >= 7 else 0
        assert proc.label == expected


def test_all_splits_are_represented(
    processed_interactions: list[ProcessedInteraction],
) -> None:
    splits = {pi.split for pi in processed_interactions}
    assert splits == {"train", "val", "test"}


def test_contains_a_mutual_match(
    processed_interactions: list[ProcessedInteraction],
) -> None:
    likes = {(pi.user_id, pi.target_id) for pi in processed_interactions if pi.label == 1}
    # u1 <-> u2 both like each other.
    assert ("u1", "u2") in likes
    assert ("u2", "u1") in likes


def test_contains_a_one_sided_pair(
    processed_interactions: list[ProcessedInteraction],
) -> None:
    by_pair = {(pi.user_id, pi.target_id): pi.label for pi in processed_interactions}
    # u1 likes u3 but u3 dislikes u1.
    assert by_pair[("u1", "u3")] == 1
    assert by_pair[("u3", "u1")] == 0


def test_user_index_is_deterministic(user_index: UserIndex) -> None:
    assert len(user_index) == 5
    assert user_index.to_index("u1") == 0
    assert user_index.to_id(0) == "u1"


def test_sample_artifacts_have_expected_model_names(
    sample_artifact: ModelArtifact, sample_neumf_artifact: ModelArtifact
) -> None:
    assert sample_artifact.model_name == "mf"
    assert "score_program" not in sample_artifact.extra
    assert sample_neumf_artifact.model_name == "neumf"
    assert "score_program" in sample_neumf_artifact.extra
