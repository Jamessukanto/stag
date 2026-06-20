"""Tests for vectorized batch reciprocal scoring."""

from __future__ import annotations

import numpy as np
import pytest
from core.serialization import ModelArtifact

from eval.services.aggregators import get_aggregator
from eval.services.batch_scoring import batch_reciprocal_scores, rank_by_reciprocal_score_batch
from eval.services.ranking import rank_by_reciprocal_score
from core.scoring import reconstruct_scorer


@pytest.fixture
def dot_artifact(sample_artifact: ModelArtifact) -> ModelArtifact:
    return sample_artifact.model_copy(update={"extra": {}})


class TestBatchReciprocalScores:
    def test_matches_slow_path_for_dot_product(
        self,
        dot_artifact: ModelArtifact,
    ) -> None:
        scorer = reconstruct_scorer(dot_artifact)
        aggregator = get_aggregator("product")
        user_ids = dot_artifact.user_index.index_to_id
        rater_idx = 0
        candidate_indices = np.array([1, 2, 3, 4], dtype=np.intp)
        candidate_ids = [user_ids[i] for i in candidate_indices]

        batch_scores = batch_reciprocal_scores(
            artifact=dot_artifact,
            aggregation="product",
            weighted_alpha=0.5,
            rater_idx=rater_idx,
            candidate_indices=candidate_indices,
        )
        slow_ranking = rank_by_reciprocal_score(
            candidates=candidate_ids,
            rater_id=user_ids[rater_idx],
            user_index=dot_artifact.user_index,
            scorer=scorer,
            aggregator=aggregator,
        )
        batch_ranking = rank_by_reciprocal_score_batch(
            candidate_ids=candidate_ids,
            candidate_indices=candidate_indices,
            rater_idx=rater_idx,
            artifact=dot_artifact,
            aggregation="product",
            weighted_alpha=0.5,
        )
        assert batch_ranking == slow_ranking

        for i, cid in enumerate(candidate_ids):
            v_idx = dot_artifact.user_index.to_index(cid)
            expected = aggregator.aggregate(
                scorer(rater_idx, v_idx),
                scorer(v_idx, rater_idx),
            )
            assert batch_scores[i] == pytest.approx(expected)

    def test_harmonic_aggregation(self, dot_artifact: ModelArtifact) -> None:
        scores = batch_reciprocal_scores(
            artifact=dot_artifact,
            aggregation="harmonic",
            weighted_alpha=0.5,
            rater_idx=0,
            candidate_indices=np.array([1], dtype=np.intp),
        )
        assert scores.shape == (1,)
        assert np.isfinite(scores[0])
