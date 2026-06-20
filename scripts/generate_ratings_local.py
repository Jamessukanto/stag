"""One-off generator for data/ratings.local.dat (closed user subgraph).

Requires full Libimseti at data/ratings.dat. Safe to delete after use; steps are
documented in README.md.

Default: seed from up to 6,000 mutual-like pairs, then write every rating where
both endpoints lie in that user cohort (likes, dislikes, one-sided). Caps output
at 120,000 directed rows by reducing seed pairs when needed.

Usage:
    python scripts/generate_ratings_local.py
    python scripts/generate_ratings_local.py \\
        --source data/ratings.dat --output data/ratings.local.dat
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

LIKE_THRESHOLD = 7
DEFAULT_TARGET_PAIRS = 6_000
DEFAULT_MAX_DIRECTED_EDGES = 120_000
DEFAULT_SEED = 42


def _collect_mutual_pairs(source: Path) -> list[tuple[str, str]]:
    """Return undirected mutual-like pairs as (min_id, max_id) tuples."""
    likes: dict[str, set[str]] = defaultdict(set)
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            user_id, target_id, rating_raw = line.split(",")
            if int(rating_raw) >= LIKE_THRESHOLD:
                likes[user_id].add(target_id)

    pairs: list[tuple[str, str]] = []
    for user_id, targets in likes.items():
        for target_id in targets:
            if user_id < target_id and user_id in likes.get(target_id, set()):
                pairs.append((user_id, target_id))
    return pairs


def _cohort_from_pairs(chosen: list[tuple[str, str]]) -> set[str]:
    cohort: set[str] = set()
    for user_a, user_b in chosen:
        cohort.add(user_a)
        cohort.add(user_b)
    return cohort


def _count_cohort_edges(source: Path, cohort: set[str]) -> int:
    count = 0
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            user_id, target_id, _ = line.split(",")
            if user_id in cohort and target_id in cohort:
                count += 1
    return count


def _write_cohort_subgraph(source: Path, output: Path, cohort: set[str]) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    edge_count = 0
    with source.open(encoding="utf-8") as handle, output.open("w", encoding="utf-8") as out:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            user_id, target_id, rating_raw = line.split(",")
            if user_id in cohort and target_id in cohort:
                out.write(f"{user_id},{target_id},{rating_raw}\n")
                edge_count += 1
    return edge_count


def _stats_for_output(output: Path) -> dict[str, int]:
    likes: dict[str, set[str]] = defaultdict(set)
    users: set[str] = set()
    edges = 0
    likes_count = 0
    dislikes_count = 0
    with output.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            user_id, target_id, rating_raw = line.split(",")
            edges += 1
            users.add(user_id)
            users.add(target_id)
            if int(rating_raw) >= LIKE_THRESHOLD:
                likes[user_id].add(target_id)
                likes_count += 1
            else:
                dislikes_count += 1
    mutual_pairs = sum(
        1
        for user_id, targets in likes.items()
        for target_id in targets
        if user_id < target_id and user_id in likes.get(target_id, set())
    )
    return {
        "users": len(users),
        "edges": edges,
        "likes": likes_count,
        "dislikes": dislikes_count,
        "undirected_mutual_like_pairs": mutual_pairs,
    }


def _resolve_seed_pair_count(
    source: Path,
    mutual_pairs: list[tuple[str, str]],
    *,
    target_pairs: int,
    max_directed_edges: int,
    seed: int,
) -> int:
    """Seed pair count <= target_pairs whose closed subgraph fits max edges.

    Uses an empirical estimate plus at most a few full-file counts (not log-scale
    binary search over 17M rows).
    """
    cap = min(target_pairs, len(mutual_pairs))
    # ~56 directed rows per seed pair at Libimseti scale (81k / 1500 empirically).
    estimate = max(1, min(cap, max_directed_edges // 56))
    seed_pairs = estimate
    for _ in range(4):
        cohort = _cohort_from_pairs(random.Random(seed).sample(mutual_pairs, seed_pairs))
        edge_count = _count_cohort_edges(source, cohort)
        if edge_count <= max_directed_edges:
            break
        seed_pairs = max(1, int(seed_pairs * max_directed_edges / max(edge_count, 1) * 0.98))
    # If under cap with room, nudge up once.
    if seed_pairs < cap:
        trial = min(cap, seed_pairs + max(1, (cap - seed_pairs) // 2))
        cohort = _cohort_from_pairs(random.Random(seed).sample(mutual_pairs, trial))
        if _count_cohort_edges(source, cohort) <= max_directed_edges:
            seed_pairs = trial
    return seed_pairs


def generate_local_subsample(
    source: Path,
    output: Path,
    *,
    target_pairs: int = DEFAULT_TARGET_PAIRS,
    max_directed_edges: int = DEFAULT_MAX_DIRECTED_EDGES,
    seed: int = DEFAULT_SEED,
) -> dict[str, int]:
    """Seed a user cohort from mutual-like pairs; write the closed subgraph."""
    if target_pairs <= 0:
        raise ValueError(f"target_pairs must be positive, got {target_pairs}")

    mutual_pairs = _collect_mutual_pairs(source)
    if not mutual_pairs:
        raise ValueError("no mutual-like pairs found in source")

    seed_pairs = min(target_pairs, len(mutual_pairs))
    if max_directed_edges > 0:
        seed_pairs = _resolve_seed_pair_count(
            source,
            mutual_pairs,
            target_pairs=target_pairs,
            max_directed_edges=max_directed_edges,
            seed=seed,
        )

    rng = random.Random(seed)
    chosen = rng.sample(mutual_pairs, seed_pairs)
    cohort = _cohort_from_pairs(chosen)
    _write_cohort_subgraph(source, output, cohort)
    stats = _stats_for_output(output)
    stats["seed_mutual_pairs"] = seed_pairs
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate closed-subgraph local Libimseti slice (ratings.local.dat)"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/ratings.dat"),
        help="Full Libimseti ratings file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ratings.local.dat"),
        help="Output path for the local slice",
    )
    parser.add_argument(
        "--target-pairs",
        type=int,
        default=DEFAULT_TARGET_PAIRS,
        help="Mutual-like pairs to seed the user cohort (may be reduced to fit max edges)",
    )
    parser.add_argument(
        "--max-directed-edges",
        type=int,
        default=DEFAULT_MAX_DIRECTED_EDGES,
        help="Cap directed rows (0 = no cap; default 120k)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for pair sampling (match config.random_seed)",
    )
    args = parser.parse_args(argv)

    if not args.source.is_file():
        print(f"error: source file not found: {args.source}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Scanning {args.source} for mutual-like pairs...", file=sys.stderr, flush=True)
    stats = generate_local_subsample(
        args.source,
        args.output,
        target_pairs=args.target_pairs,
        max_directed_edges=args.max_directed_edges,
        seed=args.seed,
    )
    print(
        f"Wrote {args.output}: "
        f"{stats['users']:,} users, "
        f"{stats['edges']:,} directed ratings "
        f"({stats['likes']:,} likes, {stats['dislikes']:,} dislikes), "
        f"{stats['undirected_mutual_like_pairs']:,} undirected mutual-like pairs "
        f"(seeded from {stats['seed_mutual_pairs']:,} pairs)",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    main()
