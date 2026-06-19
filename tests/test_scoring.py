"""Contract tests for the artifact-driven scoring interpreter (core.scoring).

These tests verify the central architectural invariant: directional scores are
reconstructed from a ModelArtifact alone, never by importing any model module.
"""

from __future__ import annotations

import sys

import numpy as np
import numpy.typing as npt
import pytest
from core.scoring import reconstruct_scorer
from core.serialization import ModelArtifact
from core.types import UserIndex


def _sigmoid(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float64)


class TestImplicitDotDefault:
    def test_mf_artifact_scores_as_dot_product(self, sample_artifact: ModelArtifact) -> None:
        scorer = reconstruct_scorer(sample_artifact)
        src = np.asarray(sample_artifact.source_embeddings)
        tgt = np.asarray(sample_artifact.target_embeddings)
        for u in range(len(sample_artifact.user_index)):
            for v in range(len(sample_artifact.user_index)):
                expected = float(np.dot(src[u], tgt[v]))
                assert scorer(u, v) == pytest.approx(expected)

    def test_dot_matches_hand_computed_value(self) -> None:
        art = ModelArtifact(
            model_name="mf",
            sampling_strategy="random",
            hyperparameters={},
            user_index=UserIndex.from_ids(["a", "b"]),
            source_embeddings=[[1.0, 2.0], [3.0, 4.0]],
            target_embeddings=[[5.0, 6.0], [7.0, 8.0]],
            created_at="2026-01-01T00:00:00+00:00",
        )
        scorer = reconstruct_scorer(art)
        # p_a . q_b = 1*7 + 2*8 = 23
        assert scorer(0, 1) == pytest.approx(23.0)
        # p_b . q_a = 3*5 + 4*6 = 39
        assert scorer(1, 0) == pytest.approx(39.0)


class TestProgramInterpreter:
    def test_neumf_program_reproduces_inline_forward_pass(
        self, sample_neumf_artifact: ModelArtifact
    ) -> None:
        art = sample_neumf_artifact
        scorer = reconstruct_scorer(art)

        src = np.asarray(art.source_embeddings)
        tgt = np.asarray(art.target_embeddings)
        mlp_src = np.asarray(art.extra["mlp_source"])
        mlp_tgt = np.asarray(art.extra["mlp_target"])
        w1 = np.asarray(art.extra["W1"])
        b1 = np.asarray(art.extra["b1"])
        wout = np.asarray(art.extra["Wout"])
        bout = np.asarray(art.extra["bout"])

        for u in range(len(art.user_index)):
            for v in range(len(art.user_index)):
                gmf = src[u] * tgt[v]
                mlp_in = np.concatenate([mlp_src[u], mlp_tgt[v]])
                mlp_h = np.maximum(mlp_in @ w1 + b1, 0.0)
                fused = np.concatenate([gmf, mlp_h])
                logit = fused @ wout + bout
                expected = float(_sigmoid(logit).reshape(-1)[0])
                assert scorer(u, v) == pytest.approx(expected)

    def test_neumf_scores_are_in_unit_interval(
        self, sample_neumf_artifact: ModelArtifact
    ) -> None:
        scorer = reconstruct_scorer(sample_neumf_artifact)
        n = len(sample_neumf_artifact.user_index)
        for u in range(n):
            for v in range(n):
                s = scorer(u, v)
                assert 0.0 <= s <= 1.0

    def test_rejects_unknown_op(self) -> None:
        art = ModelArtifact(
            model_name="mystery",
            sampling_strategy="random",
            hyperparameters={},
            user_index=UserIndex.from_ids(["a"]),
            source_embeddings=[[1.0]],
            target_embeddings=[[1.0]],
            extra={"score_program": [{"op": "quantum_entangle", "out": "s"}]},
            created_at="2026-01-01T00:00:00+00:00",
        )
        with pytest.raises(ValueError):
            reconstruct_scorer(art)


class TestNoModelImportRequired:
    def test_scoring_does_not_import_any_model_module(
        self, sample_neumf_artifact: ModelArtifact
    ) -> None:
        # Drop any already-imported model modules, then score and assert none
        # were imported as a side effect of reconstruction.
        for name in list(sys.modules):
            if name == "models" or name.startswith("models."):
                del sys.modules[name]
        scorer = reconstruct_scorer(sample_neumf_artifact)
        _ = scorer(0, 1)
        offending = [n for n in sys.modules if n == "models" or n.startswith("models.")]
        assert offending == []
