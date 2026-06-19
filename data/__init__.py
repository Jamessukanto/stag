"""The data module: raw Libimseti ratings -> model-ready supervision.

Owns loading, binarization, per-user stratified splitting, train-negative
downsampling, and uninteracted-candidate sampling for eval ranking.
Communicates with other modules only through the frozen ``core`` contracts.
"""

from __future__ import annotations

from data.loader import LibimsetiDataLoader

__all__ = ["LibimsetiDataLoader"]
