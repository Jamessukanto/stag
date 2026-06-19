"""Tests for the indexing service: UserIndex + interaction statistics."""

from __future__ import annotations

from core.types import RawInteraction, UserIndex

from data.services.indexing import build_user_index, index_interactions


class TestBuildUserIndex:
    def test_covers_raters_and_targets(
        self, raw_interactions: list[RawInteraction], user_ids: list[str]
    ) -> None:
        idx = build_user_index(raw_interactions)
        assert isinstance(idx, UserIndex)
        assert set(idx.id_to_index) == set(user_ids)

    def test_is_deterministic_first_seen_order(self) -> None:
        raws = [
            RawInteraction(user_id="b", target_id="z", rating=8),
            RawInteraction(user_id="a", target_id="b", rating=3),
        ]
        idx = build_user_index(raws)
        # first-seen order across (user_id, target_id): b, z, a
        assert idx.to_index("b") == 0
        assert idx.to_index("z") == 1
        assert idx.to_index("a") == 2


class TestIndexInteractions:
    def test_interacted_targets_per_user(
        self, raw_interactions: list[RawInteraction]
    ) -> None:
        stats = index_interactions(raw_interactions)
        assert stats.interacted_by_user["u1"] == {"u2", "u3", "u4"}
        assert stats.interacted_by_user["u3"] == {"u1", "u2"}

    def test_target_popularity_counts(
        self, raw_interactions: list[RawInteraction]
    ) -> None:
        stats = index_interactions(raw_interactions)
        assert stats.target_popularity["u1"] == 3
        assert stats.target_popularity["u2"] == 3
        assert stats.target_popularity["u3"] == 2
        assert stats.target_popularity["u4"] == 2
        assert stats.target_popularity["u5"] == 2

    def test_all_targets_is_deduped(self, raw_interactions: list[RawInteraction]) -> None:
        stats = index_interactions(raw_interactions)
        assert set(stats.all_targets) == {"u1", "u2", "u3", "u4", "u5"}
        assert len(stats.all_targets) == len(set(stats.all_targets))
