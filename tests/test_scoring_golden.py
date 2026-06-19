"""Golden tests: artifact score_program must match reference forward passes.

These are the contract tests model chats run after save/load to prove
``reconstruct_scorer(artifact)`` agrees with ``directional_score(u, v)``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from core.scoring import reconstruct_scorer, verify_scorer_matches_directional
from core.serialization import ModelArtifact


def _sigmoid(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float64)


def _mf_reference_forward(artifact: ModelArtifact, user_u: str, target_v: str) -> float:
    """Reference MF forward: dot product of source and target embeddings."""
    idx = artifact.user_index
    src = np.asarray(artifact.source_embeddings, dtype=np.float64)
    tgt = np.asarray(artifact.target_embeddings, dtype=np.float64)
    return float(np.dot(src[idx.to_index(user_u)], tgt[idx.to_index(target_v)]))


def _neumf_reference_forward(artifact: ModelArtifact, user_u: str, target_v: str) -> float:
    """Reference NeuMF forward matching the sample_neumf_artifact score_program."""
    idx = artifact.user_index
    u, v = idx.to_index(user_u), idx.to_index(target_v)
    src = np.asarray(artifact.source_embeddings, dtype=np.float64)
    tgt = np.asarray(artifact.target_embeddings, dtype=np.float64)
    mlp_src = np.asarray(artifact.extra["mlp_source"], dtype=np.float64)
    mlp_tgt = np.asarray(artifact.extra["mlp_target"], dtype=np.float64)
    w1 = np.asarray(artifact.extra["W1"], dtype=np.float64)
    b1 = np.asarray(artifact.extra["b1"], dtype=np.float64)
    wout = np.asarray(artifact.extra["Wout"], dtype=np.float64)
    bout = np.asarray(artifact.extra["bout"], dtype=np.float64)

    gmf = src[u] * tgt[v]
    mlp_in = np.concatenate([mlp_src[u], mlp_tgt[v]])
    mlp_h = np.maximum(mlp_in @ w1 + b1, 0.0)
    fused = np.concatenate([gmf, mlp_h])
    logit = fused @ wout + bout
    return float(_sigmoid(logit).reshape(-1)[0])


# Hand-picked golden pairs — spot-check, not exhaustive grid search.
_MF_GOLDEN_PAIRS: list[tuple[str, str]] = [
    ("u1", "u2"),
    ("u2", "u1"),
    ("u3", "u4"),
    ("u5", "u1"),
]

_NEUMF_GOLDEN_PAIRS: list[tuple[str, str]] = [
    ("u1", "u1"),
    ("u1", "u3"),
    ("u4", "u2"),
    ("u5", "u5"),
]


class TestGoldenScoringContract:
    def test_mf_golden_pairs_match_reference_forward(
        self, sample_artifact: ModelArtifact
    ) -> None:
        verify_scorer_matches_directional(
            sample_artifact,
            lambda u, v: _mf_reference_forward(sample_artifact, u, v),
            _MF_GOLDEN_PAIRS,
        )

    def test_neumf_golden_pairs_match_reference_forward(
        self, sample_neumf_artifact: ModelArtifact
    ) -> None:
        verify_scorer_matches_directional(
            sample_neumf_artifact,
            lambda u, v: _neumf_reference_forward(sample_neumf_artifact, u, v),
            _NEUMF_GOLDEN_PAIRS,
        )

    def test_verify_raises_on_mismatch(self, sample_artifact: ModelArtifact) -> None:
        def wrong_score(_u: str, _v: str) -> float:
            return 999.0

        with pytest.raises(AssertionError, match="score mismatch"):
            verify_scorer_matches_directional(
                sample_artifact, wrong_score, [("u1", "u2")]
            )

    def test_full_grid_still_matches_for_toy_artifacts(
        self, sample_artifact: ModelArtifact, sample_neumf_artifact: ModelArtifact
    ) -> None:
        for artifact, forward in (
            (sample_artifact, _mf_reference_forward),
            (sample_neumf_artifact, _neumf_reference_forward),
        ):
            scorer = reconstruct_scorer(artifact)
            idx = artifact.user_index
            for user_u in idx.index_to_id:
                for target_v in idx.index_to_id:
                    expected = forward(artifact, user_u, target_v)
                    actual = scorer(
                        idx.to_index(user_u), idx.to_index(target_v)
                    )
                    assert actual == pytest.approx(expected)
