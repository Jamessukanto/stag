"""Frozen shared contracts for the reciprocal-rec project.

Everything downstream modules (data, models, eval) are allowed to depend on is
re-exported here. These contracts are stable: downstream chats import them and
must not redefine them.
"""

from __future__ import annotations

from core.config import Config
from core.ground_truth import (
    EvaluationDataset,
    build_interacted_targets_by_user,
    filter_split,
    mutual_match_partners,
)
from core.interfaces import Aggregator, DataLoader, Evaluator, PreferenceModel
from core.scoring import Scorer, reconstruct_scorer, verify_scorer_matches_directional
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
    "EvaluationDataset",
    "EvaluationResult",
    "Evaluator",
    "ModelArtifact",
    "PreferenceModel",
    "ProcessedInteraction",
    "RawInteraction",
    "Scorer",
    "Split",
    "UserIndex",
    "build_interacted_targets_by_user",
    "filter_split",
    "mutual_match_partners",
    "reconstruct_scorer",
    "verify_scorer_matches_directional",
]
