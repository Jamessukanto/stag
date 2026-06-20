"""Build evaluation datasets and sweep metrics across aggregations and k."""

from __future__ import annotations

import sys
from pathlib import Path

from core.ground_truth import EvaluationDataset
from core.types import EvaluationResult, ProcessedInteraction, Split
from eval import ReciprocalEvaluator

from experiments.config import ExperimentRunConfig


def build_eval_dataset(
    interactions: list[ProcessedInteraction],
    *,
    split: Split,
) -> EvaluationDataset:
    """Filter to eval split; attach full interaction graph for distractor sampling."""
    return EvaluationDataset.from_interactions(interactions, split=split)


def evaluate_sweep(
    config: ExperimentRunConfig,
    artifact_paths: list[Path],
    ground_truth: EvaluationDataset,
) -> list[EvaluationResult]:
    """Evaluate each artifact across all aggregations and k values."""
    evaluator = ReciprocalEvaluator(
        config.base,
        ncf_distractors=config.ncf_distractors,
        weighted_alpha=config.weighted_alpha,
    )
    results: list[EvaluationResult] = []
    for path in artifact_paths:
        for aggregation in config.aggregations:
            for k in config.base.k_values:
                print(
                    f"  {path.stem} | {aggregation} | k={k}...",
                    file=sys.stderr,
                    flush=True,
                )
                results.append(
                    evaluator.evaluate(
                        path,
                        ground_truth,
                        aggregation=aggregation,
                        k=k,
                    )
                )
    return results
