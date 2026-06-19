"""Contract tests for the Config schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.config import Config


class TestConfigDefaults:
    def test_parses_with_defaults(self) -> None:
        cfg = Config()
        assert cfg.train_ratio + cfg.val_ratio + cfg.test_ratio == pytest.approx(1.0)
        assert isinstance(cfg.k_values, list)
        assert all(isinstance(k, int) for k in cfg.k_values)
        assert cfg.embedding_dim > 0
        assert cfg.epochs > 0
        assert isinstance(cfg.random_seed, int)

    def test_exposes_all_documented_fields(self) -> None:
        cfg = Config()
        for field in (
            "data_path",
            "artifact_dir",
            "train_ratio",
            "val_ratio",
            "test_ratio",
            "embedding_dim",
            "learning_rate",
            "epochs",
            "k_values",
            "negative_downsample_ratio",
            "random_seed",
        ):
            assert hasattr(cfg, field)


class TestConfigValidation:
    def test_rejects_ratios_that_do_not_sum_to_one(self) -> None:
        with pytest.raises(ValidationError):
            Config(train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    def test_accepts_custom_ratios_that_sum_to_one(self) -> None:
        cfg = Config(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        assert cfg.train_ratio == 0.6

    def test_rejects_negative_embedding_dim(self) -> None:
        with pytest.raises(ValidationError):
            Config(embedding_dim=0)
