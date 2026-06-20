"""Candidate pools, distractor sampling, and leave-one-out fold generation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from core.ground_truth import EvaluationDataset, mutual_match_partners
from core.types import UserIndex

from eval.services.sampling import sample_popularity_biased, sample_random

_RANDOM = "random"
_POPULARITY_BIASED = "popularity_biased"


@dataclass(frozen=True)
class LeaveOneOutFold:
    """One NCF leave-one-out evaluation fold."""

    user_id: str
    held_out_partner: str
    fold_index: int


def build_uninteracted_pool(
    *,
    user_id: str,
    user_index: UserIndex,
    interacted_targets_by_user: dict[str, list[str]],
) -> list[str]:
    """Users in ``user_index`` the rater has never interacted with (except self)."""
    interacted = set(interacted_targets_by_user.get(user_id, []))
    return [
        uid
        for uid in user_index.index_to_id
        if uid != user_id and uid not in interacted
    ]


def target_popularity_from_graph(
    interacted_targets_by_user: dict[str, list[str]],
) -> dict[str, int]:
    """Count how many raters have interacted with each target id."""
    counts: Counter[str] = Counter()
    for targets in interacted_targets_by_user.values():
        for target_id in targets:
            counts[target_id] += 1
    return dict(counts)


def sample_distractors(
    candidates: list[str],
    *,
    n: int,
    strategy: str,
    seed: int,
    popularity: dict[str, int],
) -> list[str]:
    """Sample up to ``n`` uninteracted distractors using ``strategy``."""
    rng = np.random.default_rng(seed)
    if strategy == _RANDOM:
        return sample_random(candidates, n, rng)
    if strategy == _POPULARITY_BIASED:
        weights = [float(popularity.get(uid, 0)) for uid in candidates]
        return sample_popularity_biased(candidates, weights, n, rng)
    raise ValueError(
        f"unknown candidate-sampling strategy {strategy!r}; "
        f"expected {_RANDOM!r} or {_POPULARITY_BIASED!r}"
    )


def leave_one_out_folds(ground_truth: EvaluationDataset) -> list[LeaveOneOutFold]:
    """One fold per mutual-match partner in the eval split."""
    folds: list[LeaveOneOutFold] = []
    user_ids = sorted(
        {interaction.user_id for interaction in ground_truth.interactions}
    )
    for user_id in user_ids:
        partners = sorted(
            mutual_match_partners(ground_truth.interactions, user_id)
        )
        for fold_index, partner in enumerate(partners):
            folds.append(
                LeaveOneOutFold(
                    user_id=user_id,
                    held_out_partner=partner,
                    fold_index=fold_index,
                )
            )
    return folds


def derive_fold_seed(
    *, base_seed: int, user_id: str, partner: str, fold_index: int
) -> int:
    """Deterministic per-fold seed from run config and fold identity."""
    return hash((base_seed, user_id, partner, fold_index)) & 0xFFFFFFFF
