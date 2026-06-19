"""Raw Libimseti ratings parsing and binarization mechanics.

The Libimseti ``ratings.dat`` file is comma-separated ``user_id,target_id,rating``
with ratings on a 1-10 integer scale. ``parse_ratings`` streams the file one line
at a time so arbitrarily large inputs never need to be fully materialized; the
public interface (an iterator of ``RawInteraction``) is unchanged regardless of
file size.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from core.types import RawInteraction

LIKE_THRESHOLD = 7


def binarize(rating: int, threshold: int = LIKE_THRESHOLD) -> int:
    """Map a 1-10 rating to a binary label: ``rating >= threshold`` is a like."""
    return 1 if rating >= threshold else 0


def parse_ratings(path: Path) -> Iterator[RawInteraction]:
    """Yield ``RawInteraction`` records from a Libimseti ratings file.

    Lazily reads the file line by line. Blank lines are skipped; any other
    malformed line (wrong field count, non-integer rating, out-of-range rating)
    raises ``ValueError`` rather than being silently dropped.
    """
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split(",")
            if len(fields) != 3:
                raise ValueError(
                    f"{path}:{lineno}: expected 'user,target,rating', got {raw_line!r}"
                )
            user_id, target_id, rating_str = (field.strip() for field in fields)
            try:
                rating = int(rating_str)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{lineno}: rating is not an integer: {rating_str!r}"
                ) from exc
            yield RawInteraction(user_id=user_id, target_id=target_id, rating=rating)
