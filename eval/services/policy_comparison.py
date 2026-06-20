"""Compare engagement-only vs reciprocal ranking on mutual-match ground truth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from core.ground_truth import EvaluationDataset, mutual_match_partners
from core.serialization import ModelArtifact

from eval.services.batch_scoring import (
    rank_by_engagement_score_batch,
    rank_by_reciprocal_score_batch,
)
from eval.services.metrics import mean_metric, recall_at_k


@dataclass(frozen=True)
class PolicyUserResult:
    """Per-user recall under both ranking policies."""

    user_id: str
    recall_engagement: float
    recall_mutual: float
    engagement_top1: str
    mutual_top1: str
    buried_mutual: bool
    top1_differs: bool
    engagement_top1_is_mutual: bool


@dataclass(frozen=True)
class PolicyComparisonResult:
    """Cohort summary: engagement vs reciprocal ranking on test mutual matches."""

    model_name: str
    aggregation: str
    k: int
    users_with_test_mutuals: int
    mean_recall_engagement: float
    mean_recall_mutual: float
    mean_recall_delta: float
    buried_mutual_rate: float
    top1_trap_rate: float
    example_cases: list[dict[str, Any]]


def compare_ranking_policies(
    artifact: ModelArtifact,
    ground_truth: EvaluationDataset,
    *,
    k: int,
    aggregation: str = "harmonic",
    weighted_alpha: float = 0.5,
    max_examples: int = 3,
) -> PolicyComparisonResult:
    """Compare Recall@K under engagement vs reciprocal ranking for each eligible user."""
    user_ids = sorted(artifact.user_index.index_to_id)
    id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    per_user: list[PolicyUserResult] = []

    for user_id in user_ids:
        partners = mutual_match_partners(ground_truth.interactions, user_id)
        if not partners:
            continue

        candidate_ids = [uid for uid in user_ids if uid != user_id]
        candidate_indices = np.array(
            [id_to_idx[cid] for cid in candidate_ids],
            dtype=np.intp,
        )
        rater_idx = id_to_idx[user_id]

        engagement_ranking = rank_by_engagement_score_batch(
            candidate_ids=candidate_ids,
            candidate_indices=candidate_indices,
            rater_idx=rater_idx,
            artifact=artifact,
        )
        mutual_ranking = rank_by_reciprocal_score_batch(
            candidate_ids=candidate_ids,
            candidate_indices=candidate_indices,
            rater_idx=rater_idx,
            artifact=artifact,
            aggregation=aggregation,
            weighted_alpha=weighted_alpha,
        )

        recall_eng = recall_at_k(engagement_ranking, partners, k=k)
        recall_mut = recall_at_k(mutual_ranking, partners, k=k)
        top_k_eng = set(engagement_ranking[:k])
        top_k_mut = set(mutual_ranking[:k])
        buried = bool(partners & top_k_mut - top_k_eng)
        eng_top1 = engagement_ranking[0]
        mut_top1 = mutual_ranking[0]
        top1_differs = eng_top1 != mut_top1

        per_user.append(
            PolicyUserResult(
                user_id=user_id,
                recall_engagement=recall_eng,
                recall_mutual=recall_mut,
                engagement_top1=eng_top1,
                mutual_top1=mut_top1,
                buried_mutual=buried,
                top1_differs=top1_differs,
                engagement_top1_is_mutual=eng_top1 in partners,
            )
        )

    return _summarize_policy_results(
        artifact.model_name,
        aggregation,
        k,
        per_user,
        max_examples=max_examples,
    )


def compare_ranking_policies_from_path(
    artifact_path: Path,
    ground_truth: EvaluationDataset,
    *,
    k: int,
    aggregation: str = "harmonic",
    weighted_alpha: float = 0.5,
    max_examples: int = 3,
) -> PolicyComparisonResult:
    """Load artifact from disk and compare ranking policies."""
    artifact = ModelArtifact.load(artifact_path)
    return compare_ranking_policies(
        artifact,
        ground_truth,
        k=k,
        aggregation=aggregation,
        weighted_alpha=weighted_alpha,
        max_examples=max_examples,
    )


def _summarize_policy_results(
    model_name: str,
    aggregation: str,
    k: int,
    per_user: list[PolicyUserResult],
    *,
    max_examples: int,
) -> PolicyComparisonResult:
    if not per_user:
        return PolicyComparisonResult(
            model_name=model_name,
            aggregation=aggregation,
            k=k,
            users_with_test_mutuals=0,
            mean_recall_engagement=0.0,
            mean_recall_mutual=0.0,
            mean_recall_delta=0.0,
            buried_mutual_rate=0.0,
            top1_trap_rate=0.0,
            example_cases=[],
        )

    recalls_eng = [u.recall_engagement for u in per_user]
    recalls_mut = [u.recall_mutual for u in per_user]
    mean_eng = mean_metric(recalls_eng)
    mean_mut = mean_metric(recalls_mut)
    buried_rate = sum(1 for u in per_user if u.buried_mutual) / len(per_user)

    differing_top1 = [u for u in per_user if u.top1_differs]
    if differing_top1:
        trap_rate = sum(
            1 for u in differing_top1 if not u.engagement_top1_is_mutual
        ) / len(differing_top1)
    else:
        trap_rate = 0.0

    examples = _select_example_cases(per_user, max_examples=max_examples)

    return PolicyComparisonResult(
        model_name=model_name,
        aggregation=aggregation,
        k=k,
        users_with_test_mutuals=len(per_user),
        mean_recall_engagement=mean_eng,
        mean_recall_mutual=mean_mut,
        mean_recall_delta=mean_mut - mean_eng,
        buried_mutual_rate=buried_rate,
        top1_trap_rate=trap_rate,
        example_cases=examples,
    )


def _select_example_cases(
    per_user: list[PolicyUserResult],
    *,
    max_examples: int,
) -> list[dict[str, Any]]:
    """Pick users with largest recall gain or a buried mutual / rank flip."""
    ranked = sorted(
        per_user,
        key=lambda u: (
            -(u.recall_mutual - u.recall_engagement),
            u.buried_mutual,
            u.top1_differs,
            u.user_id,
        ),
    )
    cases: list[dict[str, Any]] = []
    for user in ranked[:max_examples]:
        if user.recall_mutual <= user.recall_engagement and not user.buried_mutual:
            continue
        cases.append(
            {
                "user_id": user.user_id,
                "recall_engagement": user.recall_engagement,
                "recall_mutual": user.recall_mutual,
                "engagement_top1": user.engagement_top1,
                "mutual_top1": user.mutual_top1,
                "engagement_top1_is_mutual": user.engagement_top1_is_mutual,
                "buried_mutual": user.buried_mutual,
            }
        )
    return cases


def policy_result_to_dict(result: PolicyComparisonResult) -> dict[str, Any]:
    """Serialize a policy comparison for JSON output."""
    return {
        "model_name": result.model_name,
        "aggregation": result.aggregation,
        "k": result.k,
        "users_with_test_mutuals": result.users_with_test_mutuals,
        "mean_recall_engagement": result.mean_recall_engagement,
        "mean_recall_mutual": result.mean_recall_mutual,
        "mean_recall_delta": result.mean_recall_delta,
        "buried_mutual_rate": result.buried_mutual_rate,
        "top1_trap_rate": result.top1_trap_rate,
        "example_cases": result.example_cases,
    }
