"""Experiment run configuration wrapping the frozen core.Config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.config import Config
from core.types import Split
from pydantic import BaseModel, Field

DEFAULT_AGGREGATIONS: list[str] = ["product", "harmonic", "weighted"]


class ExperimentRunConfig(BaseModel):
    """Orchestration settings: shared Config plus experiments-only fields."""

    base: Config = Field(default_factory=Config)
    eval_split: Split = "test"
    sampling_strategy: str = "random"
    aggregations: list[str] = Field(default_factory=lambda: list(DEFAULT_AGGREGATIONS))
    ncf_distractors: int = 100
    weighted_alpha: float = 0.5
    results_dir: Path = Path("experiments/results")


def load_run_config(path: Path) -> ExperimentRunConfig:
    """Load run config from JSON (nested ``base`` or flat core.Config fields)."""
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if "base" in raw:
        return ExperimentRunConfig.model_validate(raw)
    base_fields = set(Config.model_fields)
    base_data = {k: v for k, v in raw.items() if k in base_fields}
    run_data = {k: v for k, v in raw.items() if k not in base_fields}
    return ExperimentRunConfig(base=Config.model_validate(base_data), **run_data)


def persist_run_config(config: ExperimentRunConfig) -> Path:
    """Write resolved config to ``results_dir/resolved_config.json``."""
    config.results_dir.mkdir(parents=True, exist_ok=True)
    out = config.results_dir / "resolved_config.json"
    out.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    return out
