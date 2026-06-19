"""The DataLoader: the data module's single public surface.

``LibimsetiDataLoader`` owns the domain orchestration (the "why/when"): it reads
the raw ratings, binarizes them, assigns a stratified per-user split, and
optionally downsamples training negatives. The reusable mechanics it composes
live in ``data.services``. It conforms structurally to ``core.DataLoader`` and
imports nothing from ``models/`` or ``eval/``.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from core.config import Config
from core.types import ProcessedInteraction, RawInteraction, UserIndex

from data.services.indexing import InteractionStats, build_user_index, index_interactions
from data.services.parsing import binarize, parse_ratings
from data.services.sampling import (
    downsample_negatives,
    sample_popularity_biased,
    sample_random,
)
from data.services.splitting import assign_splits

_RANDOM = "random"
_POPULARITY_BIASED = "popularity_biased"


class LibimsetiDataLoader:
    """Turns raw Libimseti ratings into model-ready supervision.

    Parameters
    ----------
    config:
        Shared run configuration (paths, split ratios, seed, downsample ratio).
    downsample:
        When ``True`` (default), trim training negatives per positive to
        ``config.negative_downsample_ratio``. Explicit negatives remain the
        training signal; this only thins them for efficiency/ablation. The flag
        lives here because ``Config`` intentionally carries no boolean toggle.
    """

    def __init__(self, config: Config, *, downsample: bool = True) -> None:
        self._config = config
        self._downsample = downsample
        self._raws: list[RawInteraction] | None = None
        self._stats: InteractionStats | None = None
        self._user_index: UserIndex | None = None

    def _ensure_loaded(self) -> None:
        if self._raws is not None:
            return
        raws = list(parse_ratings(self._config.data_path))
        self._raws = raws
        self._user_index = build_user_index(raws)
        self._stats = index_interactions(raws)

    def load(self) -> list[ProcessedInteraction]:
        """Return binarized, split-assigned interactions for all users."""
        self._ensure_loaded()
        assert self._raws is not None
        records = [(r.user_id, r.target_id, binarize(r.rating)) for r in self._raws]
        ratios = (
            self._config.train_ratio,
            self._config.val_ratio,
            self._config.test_ratio,
        )
        split = assign_splits(records, ratios, seed=self._config.random_seed)
        if not self._downsample:
            return split
        return self._downsample_train_negatives(split)

    def _downsample_train_negatives(
        self, interactions: list[ProcessedInteraction]
    ) -> list[ProcessedInteraction]:
        """Trim training negatives per user; leave val/test untouched."""
        rng = np.random.default_rng(self._config.random_seed)
        train_by_user: dict[str, list[ProcessedInteraction]] = defaultdict(list)
        held_out: list[ProcessedInteraction] = []
        for interaction in interactions:
            if interaction.split == "train":
                train_by_user[interaction.user_id].append(interaction)
            else:
                held_out.append(interaction)

        kept_train: list[ProcessedInteraction] = []
        for user_id in sorted(train_by_user):
            kept_train.extend(
                downsample_negatives(
                    train_by_user[user_id],
                    ratio=self._config.negative_downsample_ratio,
                    rng=rng,
                )
            )
        return kept_train + held_out

    def sample_uninteracted_candidates(
        self, user_id: str, strategy: str, n: int, seed: int
    ) -> list[str]:
        """Sample up to ``n`` uninteracted users as ranking distractors for ``user_id``.

        Returns users the rater has **never** interacted with (not explicit
        dislikes from ``load()``). ``"random"`` draws uniformly;
        ``"popularity_biased"`` draws proportional to how often each target is
        rated. Both are seeded by ``seed`` and independent of
        ``config.random_seed``.
        """
        self._ensure_loaded()
        assert self._stats is not None
        assert self._user_index is not None

        interacted = self._stats.interacted_by_user.get(user_id, set())
        candidates = [
            uid
            for uid in self._user_index.index_to_id
            if uid != user_id and uid not in interacted
        ]
        rng = np.random.default_rng(seed)
        if strategy == _RANDOM:
            return sample_random(candidates, n, rng)
        if strategy == _POPULARITY_BIASED:
            weights = [
                float(self._stats.target_popularity.get(uid, 0)) for uid in candidates
            ]
            return sample_popularity_biased(candidates, weights, n, rng)
        raise ValueError(
            f"unknown candidate-sampling strategy {strategy!r}; "
            f"expected {_RANDOM!r} or {_POPULARITY_BIASED!r}"
        )
