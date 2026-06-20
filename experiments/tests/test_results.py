"""Tests for comparison table collection and persistence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from core.types import EvaluationResult
from experiments.config import ExperimentRunConfig
from experiments.services.results import (
    collect_comparison_table,
    persist_results,
    render_markdown_table,
)


def _sample_results() -> list[EvaluationResult]:
    rows: list[EvaluationResult] = []
    for model in ("mf", "neumf"):
        for agg in ("product", "harmonic", "weighted"):
            for k in (2, 3):
                rows.append(
                    EvaluationResult(
                        model_name=model,
                        aggregation=agg,
                        k=k,
                        recall_at_k=0.1,
                        hr_at_k=0.2,
                        ndcg_at_k=0.3,
                        evaluated_at="2026-01-01T00:00:00+00:00",
                    )
                )
    return rows


class TestResults:
    def test_collect_comparison_table_fills_all_cells(self) -> None:
        results = _sample_results()
        table = collect_comparison_table(results)
        assert len(table) == 12
        keys = {(r.model_name, r.aggregation, r.k) for r in table}
        assert len(keys) == 12

    def test_persist_results_writes_json_and_csv(
        self,
        tmp_path: Path,
        pipeline_config: ExperimentRunConfig,
    ) -> None:
        pipeline_config = pipeline_config.model_copy(
            update={"results_dir": tmp_path / "results"}
        )
        results = _sample_results()
        table = collect_comparison_table(results)
        paths = persist_results(pipeline_config, table)
        assert paths.json_path.is_file()
        assert paths.csv_path.is_file()
        json_rows = json.loads(paths.json_path.read_text(encoding="utf-8"))
        assert len(json_rows) == 12
        with paths.csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)
        assert len(csv_rows) == 12
        assert set(csv_rows[0].keys()) == {
            "model_name",
            "aggregation",
            "k",
            "recall_at_k",
            "hr_at_k",
            "ndcg_at_k",
        }

    def test_render_markdown_table_non_empty(self) -> None:
        table = collect_comparison_table(_sample_results())
        md = render_markdown_table(table)
        assert "| model_name |" in md
        assert "mf" in md
