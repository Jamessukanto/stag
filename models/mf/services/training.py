"""SGD/Adam training step and epoch loss for matrix factorization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from core.types import ProcessedInteraction, UserIndex

from models.mf.services.embeddings import directional_dot
from models.mf.services.loss import l2_grad, squared_loss_and_grad

_BETA1 = 0.9
_BETA2 = 0.999
_EPS = 1e-8


@dataclass
class AdamState:
    """Per-parameter Adam moment estimates."""

    m: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    v: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    step: int = 0


def adam_update(
    param: npt.NDArray[np.float64],
    grad: npt.NDArray[np.float64],
    state: AdamState,
    learning_rate: float,
) -> None:
    """In-place Adam update for a single parameter vector."""
    if state.m.shape != param.shape:
        state.m = np.zeros_like(param)
        state.v = np.zeros_like(param)
        state.step = 0
    state.step += 1
    state.m = _BETA1 * state.m + (1.0 - _BETA1) * grad
    state.v = _BETA2 * state.v + (1.0 - _BETA2) * (grad * grad)
    m_hat = state.m / (1.0 - _BETA1**state.step)
    v_hat = state.v / (1.0 - _BETA2**state.step)
    param -= learning_rate * m_hat / (np.sqrt(v_hat) + _EPS)


def compute_epoch_loss(
    interactions: list[ProcessedInteraction],
    source: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    user_index: UserIndex,
    *,
    l2_weight: float,
) -> float:
    """Mean squared loss plus L2 penalty over all train interactions."""
    if not interactions:
        return 0.0
    total = 0.0
    for interaction in interactions:
        u_idx = user_index.to_index(interaction.user_id)
        v_idx = user_index.to_index(interaction.target_id)
        score = directional_dot(source, target, u_idx, v_idx)
        loss, _ = squared_loss_and_grad(score, float(interaction.label))
        total += loss
        total += 0.5 * l2_weight * (
            float(np.dot(source[u_idx], source[u_idx]))
            + float(np.dot(target[v_idx], target[v_idx]))
        )
    return total / len(interactions)


def train_epoch(
    interactions: list[ProcessedInteraction],
    source: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    user_index: UserIndex,
    *,
    learning_rate: float,
    l2_weight: float,
    rng: np.random.Generator,
    adam_states: dict[tuple[str, int], AdamState],
) -> float:
    """Run one shuffled training epoch; return mean loss."""
    if not interactions:
        return 0.0
    order = rng.permutation(len(interactions))
    total_loss = 0.0
    for idx in order:
        interaction = interactions[int(idx)]
        u_idx = user_index.to_index(interaction.user_id)
        v_idx = user_index.to_index(interaction.target_id)
        score = directional_dot(source, target, u_idx, v_idx)
        loss, d_score = squared_loss_and_grad(score, float(interaction.label))
        total_loss += loss

        grad_p = d_score * target[v_idx] + l2_grad(source[u_idx], l2_weight)
        grad_q = d_score * source[u_idx] + l2_grad(target[v_idx], l2_weight)

        p_state = adam_states.setdefault(("source", u_idx), AdamState())
        q_state = adam_states.setdefault(("target", v_idx), AdamState())
        adam_update(source[u_idx], grad_p, p_state, learning_rate)
        adam_update(target[v_idx], grad_q, q_state, learning_rate)

    return total_loss / len(interactions)
