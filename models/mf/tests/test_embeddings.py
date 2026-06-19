"""Tests for MF embedding initialization and directional dot product."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from models.mf.services.embeddings import directional_dot, init_embeddings


class TestInitEmbeddings:
    def test_shape(self) -> None:
        rng = np.random.default_rng(0)
        source, target = init_embeddings(n_users=5, dim=4, rng=rng)
        assert source.shape == (5, 4)
        assert target.shape == (5, 4)

    def test_deterministic_for_same_seed(self) -> None:
        source_a, target_a = init_embeddings(
            n_users=3, dim=2, rng=np.random.default_rng(7)
        )
        source_b, target_b = init_embeddings(
            n_users=3, dim=2, rng=np.random.default_rng(7)
        )
        np.testing.assert_array_equal(source_a, source_b)
        np.testing.assert_array_equal(target_a, target_b)


class TestDirectionalDot:
    @pytest.fixture
    def tiny_embeddings(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        source = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
        target = np.array([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        return source, target

    def test_matches_hand_computed_dot_product(
        self, tiny_embeddings: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ) -> None:
        source, target = tiny_embeddings
        assert directional_dot(source, target, u_idx=0, v_idx=0) == pytest.approx(0.5)
        assert directional_dot(source, target, u_idx=2, v_idx=2) == pytest.approx(1.0)
        assert directional_dot(source, target, u_idx=1, v_idx=0) == pytest.approx(0.5)
