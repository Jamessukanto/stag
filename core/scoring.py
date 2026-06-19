"""Model-agnostic reconstruction of directional scores from a ModelArtifact.

This is the architectural seam that lets ``eval/`` score any model without
importing ``models/`` and without branching on ``model_name``. The artifact is
self-describing: ``extra["score_program"]`` carries a small declarative compute
graph over a fixed primitive vocabulary, which this numpy-only interpreter
evaluates. When no program is present, the score defaults to the dot product
``p_u . q_v`` (matrix factorization).

A program is an ordered list of nodes. Each node is a dict ``{"op", "out", ...}``
that binds intermediate value ``out`` from named inputs. The two special inputs
are the source-user row index ``u`` and target-user row index ``v``. Tensor
names resolve to ``source_embeddings``, ``target_embeddings``, or any key in
``extra``. The value bound by the final node is the scalar directional score.

Frozen op vocabulary:
- ``lookup``  {table, index in {"u","v"}}            -> row vector
- ``dot``     {a, b}                                 -> scalar
- ``multiply``{a, b}  (element-wise)                 -> vector
- ``add``     {a, b}  (element-wise)                 -> vector
- ``concat``  {inputs: [...]}                        -> vector
- ``dense``   {input, weight, bias?}  (x @ W + b)    -> vector (W is [in, out])
- ``relu``    {input}                                -> vector
- ``tanh``    {input}                                -> vector
- ``sigmoid`` {input}                                -> vector

Adding a model that needs only these primitives requires no change here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from core.serialization import ModelArtifact

Scorer = Callable[[int, int], float]
Program = list[dict[str, object]]

_OPS: frozenset[str] = frozenset(
    {"lookup", "dot", "multiply", "add", "concat", "dense", "relu", "tanh", "sigmoid"}
)


def _resolve_tensor(artifact: ModelArtifact, name: str) -> npt.NDArray[np.float64]:
    if name == "source_embeddings":
        return np.asarray(artifact.source_embeddings, dtype=np.float64)
    if name == "target_embeddings":
        return np.asarray(artifact.target_embeddings, dtype=np.float64)
    if name in artifact.extra:
        return np.asarray(artifact.extra[name], dtype=np.float64)
    raise KeyError(f"tensor {name!r} not found in artifact source/target/extra")


def _validate_program(program: Program) -> None:
    for node in program:
        op = node.get("op")
        if op not in _OPS:
            raise ValueError(f"unknown score_program op: {op!r}")


def _to_scalar(value: npt.NDArray[np.float64]) -> float:
    arr = np.asarray(value, dtype=np.float64)
    if arr.size != 1:
        raise ValueError("score_program final node must produce a scalar")
    return float(arr.reshape(-1)[0])


def reconstruct_scorer(artifact: ModelArtifact) -> Scorer:
    """Return a callable ``score(u, v) -> float`` reconstructed from the artifact.

    ``u`` and ``v`` are integer row indices (from the artifact's ``user_index``).
    No model module is imported; everything is computed from the artifact alone.
    """
    program = artifact.extra.get("score_program") if artifact.extra else None

    if program is None:
        source = np.asarray(artifact.source_embeddings, dtype=np.float64)
        target = np.asarray(artifact.target_embeddings, dtype=np.float64)

        def _dot_scorer(u: int, v: int) -> float:
            return float(np.dot(source[u], target[v]))

        return _dot_scorer

    _validate_program(program)

    def _program_scorer(u: int, v: int) -> float:
        env: dict[str, npt.NDArray[np.float64]] = {}
        last: str | None = None
        for node in program:
            op = node["op"]
            out = str(node["out"])
            if op == "lookup":
                table = _resolve_tensor(artifact, str(node["table"]))
                env[out] = table[u if node["index"] == "u" else v]
            elif op == "dot":
                env[out] = np.dot(env[str(node["a"])], env[str(node["b"])])
            elif op == "multiply":
                env[out] = env[str(node["a"])] * env[str(node["b"])]
            elif op == "add":
                env[out] = env[str(node["a"])] + env[str(node["b"])]
            elif op == "concat":
                parts = [env[str(name)] for name in node["inputs"]]
                env[out] = np.concatenate(parts)
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
            last = out
        if last is None:
            raise ValueError("score_program is empty")
        return _to_scalar(env[last])

    return _program_scorer


def verify_scorer_matches_directional(
    artifact: ModelArtifact,
    directional_score: Callable[[str, str], float],
    pairs: Sequence[tuple[str, str]],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """Assert the artifact interpreter matches a model's ``directional_score``.

    Models call this in their save/load round-trip tests so training forward
    passes and ``core.scoring.reconstruct_scorer`` never silently diverge.
    Raises ``AssertionError`` on the first mismatched pair.
    """
    scorer = reconstruct_scorer(artifact)
    user_index = artifact.user_index
    for user_u, target_v in pairs:
        expected = directional_score(user_u, target_v)
        actual = scorer(user_index.to_index(user_u), user_index.to_index(target_v))
        if not np.isclose(actual, expected, rtol=rtol, atol=atol):
            raise AssertionError(
                f"score mismatch for ({user_u!r}, {target_v!r}): "
                f"interpreter={actual}, model={expected}"
            )
