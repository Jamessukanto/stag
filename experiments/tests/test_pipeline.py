"""End-to-end pipeline and reproducibility tests."""

from __future__ import annotations

import pytest
from experiments.config import ExperimentRunConfig
from experiments.pipeline import run_pipeline


class TestPipeline:
    def test_run_pipeline_end_to_end(
        self,
        pipeline_config: ExperimentRunConfig,
    ) -> None:
        result = run_pipeline(pipeline_config)
        expected_rows = (
            2 * len(pipeline_config.aggregations) * len(pipeline_config.base.k_values)
        )
        assert len(result.table) == expected_rows
        assert result.comparison_json.is_file()
        assert result.comparison_csv.is_file()
        assert result.resolved_config.is_file()
        assert result.mf_artifact.is_file()
        assert result.neumf_artifact.is_file()
        for row in result.table:
            assert 0.0 <= row.recall_at_k <= 1.0
            assert 0.0 <= row.hr_at_k <= 1.0
            assert 0.0 <= row.ndcg_at_k <= 1.0

    def test_run_pipeline_reproducible(
        self,
        pipeline_config: ExperimentRunConfig,
    ) -> None:
        first = run_pipeline(pipeline_config)
        second = run_pipeline(pipeline_config)
        assert len(first.table) == len(second.table)
        for a, b in zip(first.table, second.table, strict=True):
            assert a.model_name == b.model_name
            assert a.aggregation == b.aggregation
            assert a.k == b.k
            assert a.recall_at_k == pytest.approx(b.recall_at_k, rel=1e-5)
            assert a.hr_at_k == pytest.approx(b.hr_at_k, rel=1e-5)
            assert a.ndcg_at_k == pytest.approx(b.ndcg_at_k, rel=1e-5)
