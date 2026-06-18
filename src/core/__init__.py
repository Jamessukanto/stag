"""src.core - the stable shared foundation.

Contracts only: types, interfaces, serialization, and config. Contains no
business logic. Downstream modules import from here and must not redefine these.
"""

from src.core.config import Config
from src.core.interfaces import DataLoader, Evaluator, ReciprocityModel
from src.core.serialization import CURRENT_SCHEMA_VERSION, ModelArtifact
from src.core.types import (
    EvaluationResult,
    ModelName,
    ProcessedInteraction,
    RawInteraction,
    SamplingStrategy,
    SplitLabel,
    UserIndex,
)

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "Config",
    "DataLoader",
    "EvaluationResult",
    "Evaluator",
    "ModelArtifact",
    "ModelName",
    "ProcessedInteraction",
    "RawInteraction",
    "ReciprocityModel",
    "SamplingStrategy",
    "SplitLabel",
    "UserIndex",
]
