"""Load interactions via the data module public surface."""

from __future__ import annotations

from core.types import ProcessedInteraction
from data import LibimsetiDataLoader

from experiments.config import ExperimentRunConfig


def load_interactions(config: ExperimentRunConfig) -> list[ProcessedInteraction]:
    """Load and split data with train-negative downsampling enabled."""
    loader = LibimsetiDataLoader(config.base, downsample=True)
    return loader.load()
