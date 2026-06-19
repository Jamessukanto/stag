"""Tests for negative-sampling strategies and per-positive downsampling."""

from __future__ import annotations

from collections import Counter

import numpy as np
from core.types import ProcessedInteraction

from data.services.sampling import (
    downsample_negatives,
    sample_popularity_biased,
    sample_random,
)


class TestSampleRandom:
    def test_returns_subset_without_replacement(self) -> None:
        candidates = ["a", "b", "c", "d", "e"]
        out = sample_random(candidates, 3, np.random.default_rng(0))
        assert len(out) == 3
        assert len(set(out)) == 3
        assert set(out) <= set(candidates)

    def test_caps_at_pool_size(self) -> None:
        out = sample_random(["a", "b"], 10, np.random.default_rng(0))
        assert sorted(out) == ["a", "b"]

    def test_reproducible_for_same_seed(self) -> None:
        candidates = ["a", "b", "c", "d", "e"]
        a = sample_random(candidates, 3, np.random.default_rng(123))
        b = sample_random(candidates, 3, np.random.default_rng(123))
        assert a == b

    def test_empty_pool(self) -> None:
        assert sample_random([], 5, np.random.default_rng(0)) == []


class TestSamplePopularityBiased:
    def test_subset_without_replacement(self) -> None:
        candidates = ["a", "b", "c", "d"]
        weights = [1.0, 1.0, 1.0, 1.0]
        out = sample_popularity_biased(candidates, weights, 2, np.random.default_rng(0))
        assert len(out) == 2
        assert len(set(out)) == 2

    def test_reproducible_for_same_seed(self) -> None:
        candidates = ["a", "b", "c", "d"]
        weights = [1.0, 2.0, 3.0, 4.0]
        a = sample_popularity_biased(candidates, weights, 2, np.random.default_rng(9))
        b = sample_popularity_biased(candidates, weights, 2, np.random.default_rng(9))
        assert a == b

    def test_skews_toward_popular_targets(self) -> None:
        candidates = ["rare", "popular"]
        weights = [1.0, 99.0]
        picks: Counter[str] = Counter()
        for seed in range(500):
            rng = np.random.default_rng(seed)
            picks.update(sample_popularity_biased(candidates, weights, 1, rng))
        assert picks["popular"] > picks["rare"] * 5


def _train_records(n_pos: int, n_neg: int) -> list[ProcessedInteraction]:
    pos = [
        ProcessedInteraction(user_id="u", target_id=f"p{i}", label=1, split="train")
        for i in range(n_pos)
    ]
    neg = [
        ProcessedInteraction(user_id="u", target_id=f"n{i}", label=0, split="train")
        for i in range(n_neg)
    ]
    return pos + neg


class TestDownsampleNegatives:
    def test_keeps_all_positives(self) -> None:
        out = downsample_negatives(_train_records(3, 10), ratio=1.0, rng=np.random.default_rng(0))
        assert sum(1 for p in out if p.label == 1) == 3

    def test_trims_negatives_to_ratio_times_positives(self) -> None:
        out = downsample_negatives(_train_records(3, 10), ratio=1.0, rng=np.random.default_rng(0))
        assert sum(1 for p in out if p.label == 0) == 3

    def test_ratio_two_keeps_twice_as_many(self) -> None:
        out = downsample_negatives(_train_records(3, 10), ratio=2.0, rng=np.random.default_rng(0))
        assert sum(1 for p in out if p.label == 0) == 6

    def test_keeps_all_when_fewer_negatives_than_target(self) -> None:
        out = downsample_negatives(_train_records(5, 2), ratio=1.0, rng=np.random.default_rng(0))
        assert sum(1 for p in out if p.label == 0) == 2

    def test_reproducible_for_same_seed(self) -> None:
        a = downsample_negatives(_train_records(3, 10), ratio=1.0, rng=np.random.default_rng(42))
        b = downsample_negatives(_train_records(3, 10), ratio=1.0, rng=np.random.default_rng(42))
        assert [p.target_id for p in a] == [p.target_id for p in b]
