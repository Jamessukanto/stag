"""Collect, persist, and render comparison tables."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from core.types import EvaluationResult

from experiments.config import ExperimentRunConfig, persist_run_config


@dataclass(frozen=True)
class ComparisonRow:
    model_name: str
    aggregation: str
    k: int
    recall_at_k: float
    hr_at_k: float
    ndcg_at_k: float


@dataclass(frozen=True)
class PersistedResults:
    json_path: Path
    csv_path: Path
    resolved_config: Path


def collect_comparison_table(results: list[EvaluationResult]) -> list[ComparisonRow]:
    """Convert EvaluationResult records into stable comparison rows."""
    return [
        ComparisonRow(
            model_name=r.model_name,
            aggregation=r.aggregation,
            k=r.k,
            recall_at_k=r.recall_at_k,
            hr_at_k=r.hr_at_k,
            ndcg_at_k=r.ndcg_at_k,
        )
        for r in results
    ]


def persist_results(
    config: ExperimentRunConfig,
    table: list[ComparisonRow],
) -> PersistedResults:
    """Write comparison JSON/CSV and resolved config to results_dir."""
    config.results_dir.mkdir(parents=True, exist_ok=True)
    json_path = config.results_dir / "comparison.json"
    csv_path = config.results_dir / "comparison.csv"
    rows = [row.__dict__ for row in table]
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fieldnames = [
        "model_name",
        "aggregation",
        "k",
        "recall_at_k",
        "hr_at_k",
        "ndcg_at_k",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    resolved = persist_run_config(config)
    return PersistedResults(
        json_path=json_path,
        csv_path=csv_path,
        resolved_config=resolved,
    )


def render_markdown_table(table: list[ComparisonRow]) -> str:
    """Render a readable markdown comparison table."""
    headers = [
        "model_name",
        "aggregation",
        "k",
        "recall_at_k",
        "hr_at_k",
        "ndcg_at_k",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in table:
        cells = [
            row.model_name,
            row.aggregation,
            str(row.k),
            f"{row.recall_at_k:.4f}",
            f"{row.hr_at_k:.4f}",
            f"{row.ndcg_at_k:.4f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
