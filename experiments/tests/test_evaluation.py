"""Tests for evaluation dataset construction and metric sweep."""

from __future__ import annotations

from experiments.config import ExperimentRunConfig
from experiments.services.evaluation import build_eval_dataset, evaluate_sweep
from experiments.services.loading import load_interactions
from experiments.services.training import train_mf, train_neumf


class TestEvaluation:
    def test_build_eval_dataset_filters_split(
        self,
        pipeline_config: ExperimentRunConfig,
    ) -> None:
        interactions = load_interactions(pipeline_config)
        dataset = build_eval_dataset(interactions, split=pipeline_config.eval_split)
        assert dataset.split == pipeline_config.eval_split
        assert all(i.split == pipeline_config.eval_split for i in dataset.interactions)
        assert len(dataset.interacted_targets_by_user) > 0

    def test_evaluate_sweep_returns_all_cells(
        self,
        pipeline_config: ExperimentRunConfig,
    ) -> None:
        interactions = load_interactions(pipeline_config)
        mf_path = train_mf(pipeline_config, interactions)
        neumf_path = train_neumf(pipeline_config, interactions)
        dataset = build_eval_dataset(interactions, split=pipeline_config.eval_split)
        artifact_paths = [mf_path, neumf_path]
        results = evaluate_sweep(pipeline_config, artifact_paths, dataset)
        expected = 2 * len(pipeline_config.aggregations) * len(pipeline_config.base.k_values)
        assert len(results) == expected
