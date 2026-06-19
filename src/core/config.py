"""The single configuration schema shared across all modules.

Model-specific hyperparameters (e.g. a NeuMF tower depth) are NOT added here;
they ride in ``ModelArtifact.hyperparameters`` so this contract stays stable.
"""

from __future__ import annotations

import math
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class Config(BaseModel):
    """End-to-end run configuration with sensible, validated defaults."""

    data_path: Path = Path("data/ratings.dat")
    artifact_dir: Path = Path("artifacts")

    train_ratio: float = Field(default=0.7, gt=0.0, lt=1.0)
    val_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
    test_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)

    embedding_dim: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=0.01, gt=0.0)
    epochs: int = Field(default=20, gt=0)

    k_values: list[int] = Field(default_factory=lambda: [5, 10, 20])
    negative_downsample_ratio: float = Field(default=1.0, gt=0.0)
    random_seed: int = 42

    @model_validator(mode="after")
    def _ratios_sum_to_one(self) -> Config:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(
                f"train/val/test ratios must sum to 1.0, got {total}"
            )
        return self
