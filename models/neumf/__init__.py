"""NeuMF preference model behind ``core.PreferenceModel``."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.neumf.model import NeuMFModel

__all__ = ["NeuMFModel"]


def __getattr__(name: str) -> type[NeuMFModel]:
    if name == "NeuMFModel":
        from models.neumf.model import NeuMFModel

        return NeuMFModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
