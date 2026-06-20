"""Evaluator orchestration: load artifact, rank, score metrics."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from core.config import Config
from core.ground_truth import EvaluationDataset, mutual_match_partners
from core.interfaces import Aggregator
from core.scoring import Scorer, reconstruct_scorer
from core.serialization import ModelArtifact
from core.types import EvaluationResult

from eval.services.aggregators import get_aggregator
from eval.services.candidates import (
    build_uninteracted_pool,
    derive_fold_seed,
    leave_one_out_folds,
    sample_distractors,
    target_popularity_from_graph,
)
from eval.services.batch_scoring import rank_by_reciprocal_score_batch
from eval.services.metrics import hr_at_k, mean_metric, ndcg_at_k, recall_at_k
from eval.services.ranking import rank_by_reciprocal_score


class ReciprocalEvaluator:
    """Scores a model from its on-disk artifact without importing model code."""

    def __init__(
        self,
        config: Config,
        *,
        ncf_distractors: int = 100,
        weighted_alpha: float = 0.5,
    ) -> None:
        self._config = config
        self._ncf_distractors = ncf_distractors
        self._weighted_alpha = weighted_alpha

    def evaluate(
        self,
        artifact_path: Path,
        ground_truth: EvaluationDataset,
        aggregation: str,
        k: int,
    ) -> EvaluationResult:
        artifact = ModelArtifact.load(artifact_path)
        scorer = reconstruct_scorer(artifact)
        aggregator = get_aggregator(aggregation, alpha=self._weighted_alpha)

        recall = _mean_recall_at_k(
            artifact=artifact,
            ground_truth=ground_truth,
            aggregation=aggregation,
            weighted_alpha=self._weighted_alpha,
            k=k,
        )
        hr, ndcg = _mean_ncf_metrics(
            artifact=artifact,
            ground_truth=ground_truth,
            scorer=scorer,
            aggregator=aggregator,
            k=k,
            ncf_distractors=self._ncf_distractors,
            base_seed=self._config.random_seed,
        )

        return EvaluationResult(
            model_name=artifact.model_name,
            aggregation=aggregation,
            k=k,
            recall_at_k=recall,
            hr_at_k=hr,
            ndcg_at_k=ndcg,
            evaluated_at=datetime.now(UTC).isoformat(),
        )


def _mean_recall_at_k(
    *,
    artifact: ModelArtifact,
    ground_truth: EvaluationDataset,
    aggregation: str,
    weighted_alpha: float,
    k: int,
) -> float:
    user_ids = sorted(artifact.user_index.index_to_id)
    id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    recalls: list[float] = []
    for user_id in user_ids:
        partners = mutual_match_partners(ground_truth.interactions, user_id)
        if not partners:
            continue
        candidate_ids = [uid for uid in user_ids if uid != user_id]
        candidate_indices = np.array(
            [id_to_idx[cid] for cid in candidate_ids],
            dtype=np.intp,
        )
        ranking = rank_by_reciprocal_score_batch(
            candidate_ids=candidate_ids,
            candidate_indices=candidate_indices,
            rater_idx=id_to_idx[user_id],
            artifact=artifact,
            aggregation=aggregation,
            weighted_alpha=weighted_alpha,
        )
        recalls.append(recall_at_k(ranking, partners, k=k))
    return mean_metric(recalls)


def _mean_ncf_metrics(
    *,
    artifact: ModelArtifact,
    ground_truth: EvaluationDataset,
    scorer: Scorer,
    aggregator: Aggregator,
    k: int,
    ncf_distractors: int,
    base_seed: int,
) -> tuple[float, float]:
    popularity = target_popularity_from_graph(ground_truth.interacted_targets_by_user)
    hr_values: list[float] = []
    ndcg_values: list[float] = []

    for fold in leave_one_out_folds(ground_truth):
        pool = build_uninteracted_pool(
            user_id=fold.user_id,
            user_index=artifact.user_index,
            interacted_targets_by_user=ground_truth.interacted_targets_by_user,
        )
        seed = derive_fold_seed(
            base_seed=base_seed,
            user_id=fold.user_id,
            partner=fold.held_out_partner,
            fold_index=fold.fold_index,
        )
        distractors = sample_distractors(
            pool,
            n=ncf_distractors,
            strategy=artifact.sampling_strategy,
            seed=seed,
            popularity=popularity,
        )
        candidates = [fold.held_out_partner, *distractors]
        ranking = rank_by_reciprocal_score(
            candidates=candidates,
            rater_id=fold.user_id,
            user_index=artifact.user_index,
            scorer=scorer,
            aggregator=aggregator,
        )
        hr_values.append(hr_at_k(ranking, fold.held_out_partner, k=k))
        ndcg_values.append(ndcg_at_k(ranking, fold.held_out_partner, k=k))

    return mean_metric(hr_values), mean_metric(ndcg_values)
