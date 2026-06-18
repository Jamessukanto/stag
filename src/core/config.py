"""Project-wide runtime configuration.

A single settings object that every module reads from. No module should hardcode
paths or hyperparameters - they consume them from ``Config``. Determinism for
the whole pipeline flows from ``random_seed``.
"""

from __future__ import annotations

import math
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_RATIO_SUM_TOLERANCE = 1e-6


class Config(BaseSettings):
    """Runtime configuration, overridable via env vars prefixed ``RECREC_``."""

    model_config = SettingsConfigDict(
        env_prefix="RECREC_",
        env_file=".env",
        extra="ignore",
    )

    data_path: Path = Path("data/libimseti/ratings.dat")
    artifact_dir: Path = Path("artifacts")

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    embedding_dim: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=0.01, gt=0)
    epochs: int = Field(default=20, gt=0)
    k_values: list[int] = Field(default_factory=lambda: [5, 10, 20])
    random_seed: int = 42

    @field_validator("train_ratio", "val_ratio", "test_ratio")
    @classmethod
    def _ratio_in_unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"split ratios must be in [0, 1], got {value}")
        return value

    @field_validator("k_values")
    @classmethod
    def _k_values_positive(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("k_values must not be empty")
        if any(k <= 0 for k in value):
            raise ValueError(f"all k_values must be positive, got {value}")
        return value

    @model_validator(mode="after")
    def _ratios_sum_to_one(self) -> Config:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not math.isclose(total, 1.0, abs_tol=_RATIO_SUM_TOLERANCE):
            raise ValueError(
                "train_ratio + val_ratio + test_ratio must sum to 1.0, "
                f"got {total}"
            )
        return self
