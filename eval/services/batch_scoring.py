"""Vectorized reciprocal scores for large candidate pools."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from core.serialization import ModelArtifact


def _resolve_tensor(artifact: ModelArtifact, name: str) -> npt.NDArray[np.float64]:
    if name == "source_embeddings":
        return np.asarray(artifact.source_embeddings, dtype=np.float64)
    if name == "target_embeddings":
        return np.asarray(artifact.target_embeddings, dtype=np.float64)
    if name in artifact.extra:
        return np.asarray(artifact.extra[name], dtype=np.float64)
    raise KeyError(f"tensor {name!r} not found in artifact source/target/extra")


def _batch_directional_scores(
    artifact: ModelArtifact,
    source_indices: npt.NDArray[np.intp],
    target_indices: npt.NDArray[np.intp],
) -> npt.NDArray[np.float64]:
    """Directional scores s(source_i -> target_i) for aligned index pairs."""
    if source_indices.shape != target_indices.shape:
        raise ValueError("source_indices and target_indices must have the same shape")
    batch = source_indices.shape[0]
    program = artifact.extra.get("score_program") if artifact.extra else None
    if program is None:
        source = _resolve_tensor(artifact, "source_embeddings")
        target = _resolve_tensor(artifact, "target_embeddings")
        return np.asarray(
            np.sum(source[source_indices] * target[target_indices], axis=1),
            dtype=np.float64,
        )

    env: dict[str, npt.NDArray[np.float64]] = {}
    last: str | None = None
    for node in program:
        op = node["op"]
        out = str(node["out"])
        if op == "lookup":
            table = _resolve_tensor(artifact, str(node["table"]))
            indices = source_indices if node["index"] == "u" else target_indices
            env[out] = table[indices]
        elif op == "dot":
            env[out] = np.sum(env[str(node["a"])] * env[str(node["b"])], axis=1)
        elif op == "multiply":
            env[out] = env[str(node["a"])] * env[str(node["b"])]
        elif op == "add":
            env[out] = env[str(node["a"])] + env[str(node["b"])]
        elif op == "concat":
            parts = [env[str(name)] for name in node["inputs"]]
            env[out] = np.concatenate(parts, axis=1)
        elif op == "dense":
            weight = _resolve_tensor(artifact, str(node["weight"]))
            result = env[str(node["input"])] @ weight
            if node.get("bias") is not None:
                result = result + _resolve_tensor(artifact, str(node["bias"]))
            env[out] = result
        elif op == "relu":
            env[out] = np.maximum(env[str(node["input"])], 0.0)
        elif op == "tanh":
            env[out] = np.tanh(env[str(node["input"])])
        elif op == "sigmoid":
            env[out] = 1.0 / (1.0 + np.exp(-env[str(node["input"])]))
        else:
            raise ValueError(f"unknown score_program op: {op!r}")
        last = out
    if last is None:
        raise ValueError("score_program is empty")
    scores = np.asarray(env[last], dtype=np.float64).reshape(-1)
    if scores.shape[0] != batch:
        raise ValueError("batched score_program must return one score per candidate")
    return scores


def _aggregate_vectorized(
    s_ab: npt.NDArray[np.float64],
    s_ba: npt.NDArray[np.float64],
    aggregation: str,
    *,
    weighted_alpha: float,
) -> npt.NDArray[np.float64]:
    if aggregation == "product":
        return s_ab * s_ba
    if aggregation == "harmonic":
        total = s_ab + s_ba
        return np.divide(
            2.0 * s_ab * s_ba,
            total,
            out=np.zeros_like(s_ab),
            where=total != 0.0,
        )
    if aggregation == "weighted":
        return weighted_alpha * s_ab + (1.0 - weighted_alpha) * s_ba
    raise ValueError(f"unknown aggregation {aggregation!r}")


def batch_reciprocal_scores(
    *,
    artifact: ModelArtifact,
    aggregation: str,
    weighted_alpha: float,
    rater_idx: int,
    candidate_indices: npt.NDArray[np.intp],
) -> npt.NDArray[np.float64]:
    """Reciprocal scores r(rater, v) for many candidate row indices."""
    if candidate_indices.size == 0:
        return np.array([], dtype=np.float64)
    n = candidate_indices.shape[0]
    rater = np.full(n, rater_idx, dtype=np.intp)
    s_uv = _batch_directional_scores(artifact, rater, candidate_indices)
    s_vu = _batch_directional_scores(artifact, candidate_indices, rater)
    return _aggregate_vectorized(
        s_uv,
        s_vu,
        aggregation,
        weighted_alpha=weighted_alpha,
    )


def rank_by_engagement_score_batch(
    *,
    candidate_ids: list[str],
    candidate_indices: npt.NDArray[np.intp],
    rater_idx: int,
    artifact: ModelArtifact,
) -> list[str]:
    """Rank candidates by directional score s(rater -> candidate) only."""
    if candidate_indices.size == 0:
        return []
    rater = np.full(candidate_indices.shape[0], rater_idx, dtype=np.intp)
    scores = _batch_directional_scores(artifact, rater, candidate_indices)
    order = sorted(
        range(len(candidate_ids)),
        key=lambda i: (-float(scores[i]), candidate_ids[i]),
    )
    return [candidate_ids[i] for i in order]


def rank_by_reciprocal_score_batch(
    *,
    candidate_ids: list[str],
    candidate_indices: npt.NDArray[np.intp],
    rater_idx: int,
    artifact: ModelArtifact,
    aggregation: str,
    weighted_alpha: float,
) -> list[str]:
    """Rank candidates by reciprocal score; ties break by candidate id (matches slow path)."""
    scores = batch_reciprocal_scores(
        artifact=artifact,
        aggregation=aggregation,
        weighted_alpha=weighted_alpha,
        rater_idx=rater_idx,
        candidate_indices=candidate_indices,
    )
    order = sorted(
        range(len(candidate_ids)),
        key=lambda i: (-float(scores[i]), candidate_ids[i]),
    )
    return [candidate_ids[i] for i in order]
