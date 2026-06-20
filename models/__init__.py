"""Preference-model plug-ins (MF, NeuMF) behind ``core.PreferenceModel``."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.mf.model import MatrixFactorizationModel
    from models.neumf.model import NeuMFModel

__all__ = ["MatrixFactorizationModel", "NeuMFModel"]


def __getattr__(name: str) -> type:
    if name == "MatrixFactorizationModel":
        from models.mf.model import MatrixFactorizationModel

        return MatrixFactorizationModel
    if name == "NeuMFModel":
        from models.neumf.model import NeuMFModel

        return NeuMFModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
