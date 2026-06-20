"""Tests for policy analysis orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from core.serialization import ModelArtifact
from core.types import ProcessedInteraction
from experiments.config import ExperimentRunConfig
from experiments.services.policy_analysis import run_policy_analysis


class TestPolicyAnalysis:
    def test_writes_policy_tradeoff_json(
        self,
        pipeline_config: ExperimentRunConfig,
        processed_interactions: list[ProcessedInteraction],
        sample_artifact: ModelArtifact,
        tmp_path: Path,
    ) -> None:
        interactions = processed_interactions
        mf_path = tmp_path / "mf.json"
        sample_artifact.save(mf_path)
        pipeline_config.results_dir.mkdir(parents=True, exist_ok=True)

        out = run_policy_analysis(
            pipeline_config,
            interactions,
            [mf_path],
        )

        assert out == pipeline_config.results_dir / "policy_tradeoff.json"
        assert out.is_file()
        rows = json.loads(out.read_text(encoding="utf-8"))
        assert len(rows) == 1
        assert rows[0]["model_name"] == "mf"
        assert "mean_recall_engagement" in rows[0]
        assert "mean_recall_mutual" in rows[0]
