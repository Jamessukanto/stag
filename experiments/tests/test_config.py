"""Tests for experiment run configuration."""

from __future__ import annotations

import json
from pathlib import Path

from core.config import Config
from experiments.config import ExperimentRunConfig, load_run_config, persist_run_config


class TestExperimentRunConfig:
    def test_defaults_validate(self) -> None:
        cfg = ExperimentRunConfig(base=Config())
        assert cfg.eval_split == "test"
        assert cfg.sampling_strategy == "random"
        assert cfg.aggregations == ["product", "harmonic", "weighted"]
        assert cfg.ncf_distractors == 100
        assert cfg.weighted_alpha == 0.5

    def test_json_round_trip_nested(self, tmp_path: Path) -> None:
        original = ExperimentRunConfig(
            base=Config(random_seed=7, k_values=[1, 2]),
            eval_split="val",
            sampling_strategy="popularity_biased",
        )
        path = tmp_path / "run.json"
        path.write_text(original.model_dump_json(indent=2), encoding="utf-8")
        loaded = load_run_config(path)
        assert loaded == original

    def test_json_round_trip_flat(self, tmp_path: Path) -> None:
        flat = {
            "data_path": "data/ratings.dat",
            "random_seed": 99,
            "k_values": [5],
            "eval_split": "test",
            "sampling_strategy": "random",
        }
        path = tmp_path / "flat.json"
        path.write_text(json.dumps(flat), encoding="utf-8")
        loaded = load_run_config(path)
        assert loaded.base.random_seed == 99
        assert loaded.base.k_values == [5]
        assert loaded.eval_split == "test"

    def test_persist_resolved_config(self, tmp_path: Path) -> None:
        cfg = ExperimentRunConfig(
            base=Config(random_seed=1),
            results_dir=tmp_path / "results",
        )
        out = persist_run_config(cfg)
        assert out == tmp_path / "results" / "resolved_config.json"
        assert out.is_file()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["base"]["random_seed"] == 1
        assert data["eval_split"] == "test"
