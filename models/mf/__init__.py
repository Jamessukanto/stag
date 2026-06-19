"""Matrix-factorization preference model behind ``core.PreferenceModel``."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.mf.model import MatrixFactorizationModel

__all__ = ["MatrixFactorizationModel"]


def __getattr__(name: str) -> type[MatrixFactorizationModel]:
    if name == "MatrixFactorizationModel":
        from models.mf.model import MatrixFactorizationModel

        return MatrixFactorizationModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
