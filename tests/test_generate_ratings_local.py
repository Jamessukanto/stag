"""Tests for scripts/generate_ratings_local.py closed-subgraph generation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_ratings_local.py"
_spec = importlib.util.spec_from_file_location("generate_ratings_local", _MODULE_PATH)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
generate_local_subsample = _mod.generate_local_subsample


def _write_source(path: Path) -> None:
    """Users 1-3: mutual 1-2 and 2-3; one-sided/dislikes between 1 and 3."""
    rows = [
        "1,2,9",
        "2,1,8",
        "2,3,7",
        "3,2,8",
        "1,3,10",
        "3,1,2",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


class TestGenerateLocalSubsample:
    def test_closed_subgraph_includes_dislikes(self, tmp_path: Path) -> None:
        source = tmp_path / "source.dat"
        output = tmp_path / "local.dat"
        _write_source(source)
        stats = generate_local_subsample(
            source,
            output,
            target_pairs=2,
            max_directed_edges=0,
            seed=42,
        )
        assert stats["edges"] == 6
        assert stats["likes"] == 4
        assert stats["dislikes"] == 2
        assert stats["undirected_mutual_like_pairs"] == 2
        assert stats["seed_mutual_pairs"] == 2
        assert stats["users"] == 3

    def test_single_seed_pair_smaller_subgraph(self, tmp_path: Path) -> None:
        source = tmp_path / "source.dat"
        output = tmp_path / "local.dat"
        _write_source(source)
        stats = generate_local_subsample(
            source,
            output,
            target_pairs=1,
            max_directed_edges=0,
            seed=42,
        )
        assert stats["edges"] == 2
        assert stats["seed_mutual_pairs"] == 1

    def test_respects_max_directed_edges(self, tmp_path: Path) -> None:
        source = tmp_path / "source.dat"
        output = tmp_path / "local.dat"
        _write_source(source)
        stats = generate_local_subsample(
            source,
            output,
            target_pairs=2,
            max_directed_edges=2,
            seed=42,
        )
        assert stats["edges"] <= 2
        assert stats["seed_mutual_pairs"] == 1

    def test_raises_when_no_mutual_pairs(self, tmp_path: Path) -> None:
        source = tmp_path / "source.dat"
        source.write_text("1,2,8\n2,1,3\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no mutual-like pairs"):
            generate_local_subsample(
                source,
                tmp_path / "out.dat",
                target_pairs=1,
                max_directed_edges=0,
                seed=42,
            )
