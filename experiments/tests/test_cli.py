"""Tests for the experiments CLI and checked-in run configs."""

from __future__ import annotations

from pathlib import Path

import pytest
from experiments.cli import main
from experiments.config import load_run_config
from experiments.pipeline import run_pipeline

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROTOTYPE_CONFIG = _REPO_ROOT / "experiments" / "configs" / "prototype.json"
_BENCHMARK_CONFIG = _REPO_ROOT / "experiments" / "configs" / "benchmark.json"


class TestCli:
    def test_cli_requires_config(self) -> None:
        with pytest.raises(SystemExit):
            main([])


class TestCheckedInConfigs:
    def test_prototype_config_loads(self) -> None:
        cfg = load_run_config(_PROTOTYPE_CONFIG)
        assert cfg.base.data_path == Path("data/ratings.local.dat")
        assert cfg.base.artifact_dir == Path("artifacts/prototype")
        assert cfg.base.embedding_dim == 4
        assert cfg.base.epochs == 3
        assert cfg.base.k_values == [2, 5]
        assert cfg.aggregations == ["product", "harmonic"]
        assert cfg.ncf_distractors == 4
        assert cfg.results_dir == Path("experiments/results/prototype")

    def test_benchmark_config_loads(self) -> None:
        cfg = load_run_config(_BENCHMARK_CONFIG)
        assert cfg.base.data_path == Path("data/ratings.dat")
        assert cfg.base.artifact_dir == Path("artifacts/benchmark")
        assert cfg.base.embedding_dim == 32
        assert cfg.base.epochs == 20
        assert cfg.base.k_values == [5, 10, 20]
        assert cfg.ncf_distractors == 100
        assert cfg.results_dir == Path("experiments/results/benchmark")


class TestPrototypeConfigEndToEnd:
    def test_prototype_config_runs_end_to_end(
        self,
        tmp_path: Path,
        ratings_file: Path,
    ) -> None:
        loaded = load_run_config(_PROTOTYPE_CONFIG)
        config = loaded.model_copy(
            update={
                "base": loaded.base.model_copy(
                    update={
                        "data_path": ratings_file,
                        "artifact_dir": tmp_path / "artifacts",
                    }
                ),
                "results_dir": tmp_path / "results",
            }
        )
        result = run_pipeline(config)
        expected_rows = 2 * len(config.aggregations) * len(config.base.k_values)
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
