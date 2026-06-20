"""Tests for reciprocal ranking (eval.services.ranking)."""

from __future__ import annotations

from core.interfaces import Aggregator
from eval.services.ranking import rank_candidates, reciprocal_score


class _FixedAggregator:
    def __init__(self, value: float) -> None:
        self._value = value
        self.calls: list[tuple[float, float]] = []

    def aggregate(self, s_ab: float, s_ba: float) -> float:
        self.calls.append((s_ab, s_ba))
        return self._value


class TestReciprocalScore:
    def test_uses_both_directional_scores(self) -> None:
        scores = {(0, 1): 2.0, (1, 0): 3.0}

        def scorer(u: int, v: int) -> float:
            return scores[(u, v)]

        agg = _FixedAggregator(5.0)
        result = reciprocal_score(
            scorer=scorer,
            aggregator=agg,
            user_index_u=0,
            user_index_v=1,
        )
        assert result == 5.0
        assert agg.calls == [(2.0, 3.0)]


class TestRankCandidates:
    def test_sorts_descending_by_score(self) -> None:
        def score_fn(candidate: str) -> float:
            return {"a": 3.0, "b": 1.0, "c": 2.0}[candidate]

        ranked = rank_candidates(["b", "a", "c"], score_fn=score_fn)
        assert ranked == ["a", "c", "b"]

    def test_tie_breaks_by_candidate_id(self) -> None:
        def score_fn(candidate: str) -> float:
            return 1.0

        ranked = rank_candidates(["z", "a", "m"], score_fn=score_fn)
        assert ranked == ["a", "m", "z"]

    def test_accepts_aggregator_protocol(self) -> None:
        class _SumAggregator(Aggregator):
            def aggregate(self, s_ab: float, s_ba: float) -> float:
                return s_ab + s_ba

        _ = _SumAggregator()
