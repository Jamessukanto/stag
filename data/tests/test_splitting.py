"""Tests for per-user stratified, seeded splitting."""

from __future__ import annotations

from collections import Counter

from core.types import ProcessedInteraction, RawInteraction

from data.services.splitting import allocate_counts, assign_splits

RATIOS = (0.7, 0.15, 0.15)


class TestAllocateCounts:
    def test_empty(self) -> None:
        assert allocate_counts(0, RATIOS) == (0, 0, 0)

    def test_one_goes_to_train_only(self) -> None:
        assert allocate_counts(1, RATIOS) == (1, 0, 0)

    def test_two_cover_train_and_val(self) -> None:
        assert allocate_counts(2, RATIOS) == (1, 1, 0)

    def test_three_cover_all_splits(self) -> None:
        assert allocate_counts(3, RATIOS) == (1, 1, 1)

    def test_large_stratum_is_proportional_and_covers_all(self) -> None:
        assert allocate_counts(10, RATIOS) == (7, 2, 1)

    def test_counts_always_sum_to_n(self) -> None:
        for n in range(0, 50):
            assert sum(allocate_counts(n, RATIOS)) == n


def _records_for_single_user(n: int, label: int = 1) -> list[tuple[str, str, int]]:
    return [("u", f"t{i}", label) for i in range(n)]


class TestAssignSplits:
    def test_returns_processed_interactions(self) -> None:
        out = assign_splits([("u", "t", 1)], RATIOS, seed=42)
        assert all(isinstance(p, ProcessedInteraction) for p in out)

    def test_preserves_every_record_exactly_once(
        self, raw_interactions: list[RawInteraction]
    ) -> None:
        records = [(r.user_id, r.target_id, 1 if r.rating >= 7 else 0) for r in raw_interactions]
        out = assign_splits(records, RATIOS, seed=42)
        assert Counter((p.user_id, p.target_id, p.label) for p in out) == Counter(records)

    def test_no_interaction_appears_in_multiple_splits(
        self, raw_interactions: list[RawInteraction]
    ) -> None:
        """Each (user, target) pair must land in exactly one split — no leakage."""
        records = [(r.user_id, r.target_id, 1 if r.rating >= 7 else 0) for r in raw_interactions]
        out = assign_splits(records, RATIOS, seed=42)
        by_pair: dict[tuple[str, str], set[str]] = {}
        for p in out:
            by_pair.setdefault((p.user_id, p.target_id), set()).add(p.split)
        assert all(len(splits) == 1 for splits in by_pair.values())

    def test_deterministic_for_same_seed(self) -> None:
        records = _records_for_single_user(20)
        a = assign_splits(records, RATIOS, seed=7)
        b = assign_splits(records, RATIOS, seed=7)
        assert [(p.target_id, p.split) for p in a] == [(p.target_id, p.split) for p in b]

    def test_different_seed_changes_assignment(self) -> None:
        records = _records_for_single_user(20)
        a = {p.target_id: p.split for p in assign_splits(records, RATIOS, seed=1)}
        b = {p.target_id: p.split for p in assign_splits(records, RATIOS, seed=2)}
        assert a != b

    def test_large_user_spans_all_three_splits(self) -> None:
        records = _records_for_single_user(20)
        splits = {p.split for p in assign_splits(records, RATIOS, seed=3)}
        assert splits == {"train", "val", "test"}

    def test_stratifies_by_label(self) -> None:
        # 10 positives + 10 negatives for one user; each label spans the splits
        # independently, so train holds ~70% of each class, not 70% overall.
        records = _records_for_single_user(10, label=1) + _records_for_single_user(10, label=0)
        out = assign_splits(records, RATIOS, seed=5)
        per_label_train = Counter(
            p.label for p in out if p.split == "train"
        )
        assert per_label_train[1] == 7
        assert per_label_train[0] == 7

    def test_every_user_present_in_train(
        self, raw_interactions: list[RawInteraction]
    ) -> None:
        records = [(r.user_id, r.target_id, 1 if r.rating >= 7 else 0) for r in raw_interactions]
        out = assign_splits(records, RATIOS, seed=42)
        train_users = {p.user_id for p in out if p.split == "train"}
        all_users = {r[0] for r in records}
        assert train_users == all_users
