"""Persist engagement vs reciprocal ranking policy comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.types import ProcessedInteraction
from eval.services.policy_comparison import (
    compare_ranking_policies_from_path,
    policy_result_to_dict,
)

from experiments.config import ExperimentRunConfig
from experiments.services.evaluation import build_eval_dataset


def run_policy_analysis(
    config: ExperimentRunConfig,
    interactions: list[ProcessedInteraction],
    artifact_paths: list[Path],
) -> Path:
    """Compare ranking policies for each artifact; write ``policy_tradeoff.json``."""
    ground_truth = build_eval_dataset(interactions, split=config.eval_split)
    rows: list[dict[str, Any]] = []
    for path in artifact_paths:
        result = compare_ranking_policies_from_path(
            path,
            ground_truth,
            k=config.policy_k,
            aggregation=config.policy_aggregation,
            weighted_alpha=config.weighted_alpha,
        )
        rows.append(policy_result_to_dict(result))

    config.results_dir.mkdir(parents=True, exist_ok=True)
    out = config.results_dir / "policy_tradeoff.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return out
