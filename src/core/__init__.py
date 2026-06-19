"""Frozen shared contracts for the reciprocal-rec project.

Everything downstream modules (data, models, eval) are allowed to depend on is
re-exported here. These contracts are stable: downstream chats import them and
must not redefine them.
"""

from __future__ import annotations

from core.config import Config
from core.interfaces import Aggregator, DataLoader, Evaluator, PreferenceModel
from core.scoring import Scorer, reconstruct_scorer
from core.serialization import ModelArtifact
from core.types import (
    EvaluationResult,
    ProcessedInteraction,
    RawInteraction,
    Split,
    UserIndex,
)

__all__ = [
    "Aggregator",
    "Config",
    "DataLoader",
    "EvaluationResult",
    "Evaluator",
    "ModelArtifact",
    "PreferenceModel",
    "ProcessedInteraction",
    "RawInteraction",
    "Scorer",
    "Split",
    "UserIndex",
    "reconstruct_scorer",
]
