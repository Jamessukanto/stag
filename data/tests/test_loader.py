"""Tests for the LibimsetiDataLoader public entry point."""

from __future__ import annotations

import pytest
from core.config import Config
from core.interfaces import DataLoader
from core.types import ProcessedInteraction

from data.loader import LibimsetiDataLoader


class TestProtocolConformance:
    def test_is_a_dataloader(self, data_config: Config) -> None:
        loader = LibimsetiDataLoader(data_config)
        assert isinstance(loader, DataLoader)


class TestLoad:
    def test_returns_processed_interactions(self, data_config: Config) -> None:
        out = LibimsetiDataLoader(data_config, downsample=False).load()
        assert out
        assert all(isinstance(p, ProcessedInteraction) for p in out)

    def test_without_downsampling_keeps_every_record(self, data_config: Config) -> None:
        out = LibimsetiDataLoader(data_config, downsample=False).load()
        assert len(out) == 12

    def test_binarizes_with_threshold_seven(self, data_config: Config) -> None:
        out = LibimsetiDataLoader(data_config, downsample=False).load()
        labels = {(p.user_id, p.target_id): p.label for p in out}
        assert labels[("u1", "u2")] == 1  # rating 9
        assert labels[("u2", "u3")] == 1  # rating 7 (boundary)
        assert labels[("u1", "u4")] == 0  # rating 5
        assert labels[("u4", "u5")] == 0  # rating 6

    def test_all_splits_are_valid(self, data_config: Config) -> None:
        out = LibimsetiDataLoader(data_config, downsample=False).load()
        assert {p.split for p in out} <= {"train", "val", "test"}

    def test_deterministic_for_same_seed(self, data_config: Config) -> None:
        a = LibimsetiDataLoader(data_config, downsample=False).load()
        b = LibimsetiDataLoader(data_config, downsample=False).load()
        assert a == b

    def test_no_train_val_test_leakage(self, data_config: Config) -> None:
        """End-to-end: each (user, target) interaction appears in one split only."""
        out = LibimsetiDataLoader(data_config, downsample=False).load()
        by_pair: dict[tuple[str, str], set[str]] = {}
        for p in out:
            by_pair.setdefault((p.user_id, p.target_id), set()).add(p.split)
        assert all(len(splits) == 1 for splits in by_pair.values())

    def test_downsampling_reduces_train_negatives(self, data_config: Config) -> None:
        full = LibimsetiDataLoader(data_config, downsample=False).load()
        trimmed = LibimsetiDataLoader(data_config, downsample=True).load()
        n_train_neg_full = sum(1 for p in full if p.split == "train" and p.label == 0)
        n_train_neg_trimmed = sum(1 for p in trimmed if p.split == "train" and p.label == 0)
        assert n_train_neg_trimmed <= n_train_neg_full
        # positives are never trimmed
        assert sum(1 for p in trimmed if p.label == 1) == sum(1 for p in full if p.label == 1)


class TestSampleUninteractedCandidates:
    def test_excludes_interacted_and_self(self, data_config: Config) -> None:
        loader = LibimsetiDataLoader(data_config)
        negs = loader.sample_uninteracted_candidates("u1", strategy="random", n=10, seed=0)
        # u1 interacted with u2, u3, u4; never itself
        assert "u1" not in negs
        assert {"u2", "u3", "u4"}.isdisjoint(negs)

    def test_reproducible_for_same_seed(self, data_config: Config) -> None:
        loader = LibimsetiDataLoader(data_config)
        a = loader.sample_uninteracted_candidates("u4", strategy="random", n=2, seed=11)
        b = loader.sample_uninteracted_candidates("u4", strategy="random", n=2, seed=11)
        assert a == b

    def test_popularity_biased_returns_valid_candidates(self, data_config: Config) -> None:
        loader = LibimsetiDataLoader(data_config)
        negs = loader.sample_uninteracted_candidates(
            "u4", strategy="popularity_biased", n=2, seed=1
        )
        assert set(negs) <= {"u2", "u3"}

    def test_caps_at_candidate_pool(self, data_config: Config) -> None:
        loader = LibimsetiDataLoader(data_config)
        negs = loader.sample_uninteracted_candidates("u1", strategy="random", n=100, seed=0)
        assert set(negs) == {"u5"}

    def test_unknown_strategy_raises(self, data_config: Config) -> None:
        loader = LibimsetiDataLoader(data_config)
        with pytest.raises(ValueError):
            loader.sample_uninteracted_candidates("u1", strategy="bogus", n=1, seed=0)
