"""Indexing mechanics: build a UserIndex and aggregate interaction statistics.

These are pure reductions over the raw interactions. The statistics they
produce (who interacted with whom, how popular each target is) are what the
negative-sampling strategies consume; keeping them here means the loader never
re-derives them ad hoc.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from core.types import RawInteraction, UserIndex


@dataclass(frozen=True)
class InteractionStats:
    """Structured summary of the observed interactions.

    - ``interacted_by_user``: for each rater, the set of targets they rated
      (used to exclude already-seen targets from negative sampling).
    - ``target_popularity``: how often each id appears as a rating target
      (the sampling weight for the popularity-biased strategy).
    - ``all_targets``: every distinct target id, in first-seen order.
    """

    interacted_by_user: dict[str, set[str]] = field(default_factory=dict)
    target_popularity: dict[str, int] = field(default_factory=dict)
    all_targets: list[str] = field(default_factory=list)


def build_user_index(raws: Iterable[RawInteraction]) -> UserIndex:
    """Build a UserIndex over every id appearing as a rater or a target."""
    ids: list[str] = []
    for r in raws:
        ids.append(r.user_id)
        ids.append(r.target_id)
    return UserIndex.from_ids(ids)


def index_interactions(raws: Iterable[RawInteraction]) -> InteractionStats:
    """Reduce raw interactions to per-user targets and target popularity."""
    interacted: dict[str, set[str]] = {}
    popularity: Counter[str] = Counter()
    all_targets: list[str] = []
    seen_targets: set[str] = set()
    for r in raws:
        interacted.setdefault(r.user_id, set()).add(r.target_id)
        popularity[r.target_id] += 1
        if r.target_id not in seen_targets:
            seen_targets.add(r.target_id)
            all_targets.append(r.target_id)
    return InteractionStats(
        interacted_by_user=interacted,
        target_popularity=dict(popularity),
        all_targets=all_targets,
    )
