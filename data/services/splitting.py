"""Per-user stratified, seeded train/val/test splitting.

Libimseti carries no timestamps, so the holdout is a random per-user split
rather than a temporal cutoff. Two properties matter:

- Stratification: each user's interactions are split independently, and within
  a user the two label classes (like / dislike) are split independently too, so
  every split keeps a comparable class balance for that user.
- Coverage for small users: ``allocate_counts`` seeds one record into each split
  in priority order (train, then val, then test) before distributing the rest
  proportionally. So a stratum of size 1 lands in train, size 2 in train+val,
  size 3 in train+val+test, and larger strata approach the configured ratios.

Splitting is deterministic given the seed: strata are processed in a fixed
sorted order and shuffled with a single seeded numpy generator.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from core.types import ProcessedInteraction, Split


def allocate_counts(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Split ``n`` items into (train, val, test) counts.

    Coverage-first then proportional: seed one item per split in priority order
    (train > val > test) for tiny strata, then hand out the remainder to whichever
    split is furthest below its proportional target (ties broken by the same
    priority). The returned counts always sum to ``n``.
    """
    if n <= 0:
        return (0, 0, 0)
    counts = [0, 0, 0]
    priority = (0, 1, 2)  # train, val, test
    for slot in range(min(n, 3)):
        counts[priority[slot]] += 1
    targets = [ratio * n for ratio in ratios]
    for _ in range(n - sum(counts)):
        best = max(priority, key=lambda i: (targets[i] - counts[i], -i))
        counts[best] += 1
    return (counts[0], counts[1], counts[2])


def assign_splits(
    records: list[tuple[str, str, int]],
    ratios: tuple[float, float, float],
    seed: int,
) -> list[ProcessedInteraction]:
    """Assign a split to each ``(user_id, target_id, label)`` record.

    Records are grouped by user and then by label (the stratification key);
    each stratum is shuffled with the seeded generator and sliced by
    ``allocate_counts``. Every input record appears exactly once in the output.
    """
    rng = np.random.default_rng(seed)
    by_stratum: dict[tuple[str, int], list[tuple[str, str, int]]] = defaultdict(list)
    for record in records:
        user_id, _target_id, label = record
        by_stratum[(user_id, label)].append(record)

    out: list[ProcessedInteraction] = []
    for key in sorted(by_stratum):
        stratum = by_stratum[key]
        order = rng.permutation(len(stratum))
        n_train, n_val, _n_test = allocate_counts(len(stratum), ratios)
        for rank, position in enumerate(order):
            if rank < n_train:
                split: Split = "train"
            elif rank < n_train + n_val:
                split = "val"
            else:
                split = "test"
            user_id, target_id, label = stratum[position]
            out.append(
                ProcessedInteraction(
                    user_id=user_id, target_id=target_id, label=label, split=split
                )
            )
    return out
