"""Reproducible orchestration: load data, train models, evaluate, compare."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from experiments.config import ExperimentRunConfig
    from experiments.pipeline import run_pipeline

__all__ = ["ExperimentRunConfig", "run_pipeline"]


def __getattr__(name: str) -> Any:
    if name == "ExperimentRunConfig":
        from experiments.config import ExperimentRunConfig

        return ExperimentRunConfig
    if name == "run_pipeline":
        from experiments.pipeline import run_pipeline

        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
