"""End-to-end experiment pipeline orchestration."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from experiments.config import ExperimentRunConfig
from experiments.services.evaluation import build_eval_dataset, evaluate_sweep
from experiments.services.loading import load_interactions
from experiments.services.results import (
    ComparisonRow,
    collect_comparison_table,
    persist_results,
)
from experiments.services.training import train_mf, train_neumf


@dataclass(frozen=True)
class PipelineResult:
    table: list[ComparisonRow]
    mf_artifact: Path
    neumf_artifact: Path
    comparison_json: Path
    comparison_csv: Path
    resolved_config: Path


def run_pipeline(config: ExperimentRunConfig) -> PipelineResult:
    """Load data, train both models, evaluate, and persist comparison results."""
    print("Loading interactions...", file=sys.stderr, flush=True)
    interactions = load_interactions(config)
    print(f"Loaded {len(interactions):,} interactions.", file=sys.stderr, flush=True)

    print(f"Training MF...", file=sys.stderr, flush=True)
    mf_path = train_mf(config, interactions)

    print("Training NeuMF...", file=sys.stderr, flush=True)
    neumf_path = train_neumf(config, interactions)
    
    print("Evaluating...", file=sys.stderr, flush=True)

    ground_truth = build_eval_dataset(interactions, split=config.eval_split)
    results = evaluate_sweep(config, [mf_path, neumf_path], ground_truth)
    table = collect_comparison_table(results)
    paths = persist_results(config, table)
    print("Done.", file=sys.stderr, flush=True)
    return PipelineResult(
        table=table,
        mf_artifact=mf_path,
        neumf_artifact=neumf_path,
        comparison_json=paths.json_path,
        comparison_csv=paths.csv_path,
        resolved_config=paths.resolved_config,
    )
