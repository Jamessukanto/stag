"""Tests for the parsing/binarization service mechanics."""

from __future__ import annotations

from pathlib import Path

import pytest
from core.types import RawInteraction

from data.services.parsing import binarize, parse_ratings


class TestBinarize:
    @pytest.mark.parametrize(
        ("rating", "label"),
        [(1, 0), (5, 0), (6, 0), (7, 1), (8, 1), (10, 1)],
    )
    def test_threshold_at_seven(self, rating: int, label: int) -> None:
        assert binarize(rating) == label


class TestParseRatings:
    def test_yields_raw_interactions(self, ratings_file: Path) -> None:
        parsed = list(parse_ratings(ratings_file))
        assert all(isinstance(r, RawInteraction) for r in parsed)
        assert len(parsed) == 12

    def test_round_trips_the_written_records(
        self, ratings_file: Path, raw_interactions: list[RawInteraction]
    ) -> None:
        parsed = list(parse_ratings(ratings_file))
        assert parsed == raw_interactions

    def test_is_streaming_generator(self, ratings_file: Path) -> None:
        import types

        gen = parse_ratings(ratings_file)
        assert isinstance(gen, types.GeneratorType)

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "with_blanks.dat"
        path.write_text("u1,u2,9\n\n  \nu2,u1,3\n", encoding="utf-8")
        parsed = list(parse_ratings(path))
        assert len(parsed) == 2

    def test_rejects_malformed_line(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.dat"
        path.write_text("u1,u2\n", encoding="utf-8")
        with pytest.raises(ValueError):
            list(parse_ratings(path))

    def test_rejects_out_of_range_rating(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_rating.dat"
        path.write_text("u1,u2,99\n", encoding="utf-8")
        with pytest.raises(ValueError):
            list(parse_ratings(path))
