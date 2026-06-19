"""The data module: raw Libimseti ratings -> model-ready supervision.

Owns loading, binarization, per-user stratified splitting, and negative
sampling. Communicates with other modules only through the frozen ``core``
contracts; it knows nothing about how scores are computed or evaluated.
"""

from __future__ import annotations

from data.loader import LibimsetiDataLoader

__all__ = ["LibimsetiDataLoader"]
