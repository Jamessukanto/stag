"""Reciprocal score computation and candidate ranking."""

from __future__ import annotations

from collections.abc import Callable

from core.interfaces import Aggregator
from core.scoring import Scorer
from core.types import UserIndex


def reciprocal_score(
    *,
    scorer: Scorer,
    aggregator: Aggregator,
    user_index_u: int,
    user_index_v: int,
) -> float:
    """Fuse directional scores s(u->v) and s(v->u) into r(u, v)."""
    s_uv = scorer(user_index_u, user_index_v)
    s_vu = scorer(user_index_v, user_index_u)
    return aggregator.aggregate(s_uv, s_vu)


def rank_candidates(
    candidates: list[str],
    *,
    score_fn: Callable[[str], float],
) -> list[str]:
    """Rank ``candidates`` by ``score_fn`` descending; ties break by id."""
    return sorted(candidates, key=lambda cid: (-score_fn(cid), cid))


def rank_by_reciprocal_score(
    *,
    candidates: list[str],
    rater_id: str,
    user_index: UserIndex,
    scorer: Scorer,
    aggregator: Aggregator,
) -> list[str]:
    """Rank ``candidates`` for ``rater_id`` by reciprocal score."""
    u_idx = user_index.to_index(rater_id)

    def score_fn(candidate_id: str) -> float:
        v_idx = user_index.to_index(candidate_id)
        return reciprocal_score(
            scorer=scorer,
            aggregator=aggregator,
            user_index_u=u_idx,
            user_index_v=v_idx,
        )

    return rank_candidates(candidates, score_fn=score_fn)
