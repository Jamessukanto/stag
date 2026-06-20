"""Training loop, seeding, and early-stopping helpers for NeuMF."""

from __future__ import annotations

import random

import numpy as np
import torch
from core.types import ProcessedInteraction, UserIndex
from torch import nn
from torch.optim import Optimizer

from models.neumf.services.loss import bce_loss


def seed_all(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_split_bce(
    network: nn.Module,
    interactions: list[ProcessedInteraction],
    user_index: UserIndex,
) -> float:
    """Mean BCE over interactions without gradient updates."""
    if not interactions:
        return 0.0
    network.eval()
    total = 0.0
    with torch.no_grad():
        for interaction in interactions:
            u_idx = torch.tensor(user_index.to_index(interaction.user_id), dtype=torch.long)
            v_idx = torch.tensor(user_index.to_index(interaction.target_id), dtype=torch.long)
            pred = network.forward_pair(u_idx, v_idx)
            label = torch.tensor(float(interaction.label))
            total += float(bce_loss(pred, label))
    return total / len(interactions)


def train_epoch(
    network: nn.Module,
    interactions: list[ProcessedInteraction],
    user_index: UserIndex,
    *,
    optimizer: Optimizer,
    seed: int,
) -> float:
    """Run one shuffled training epoch; return mean BCE loss."""
    if not interactions:
        return 0.0
    network.train()
    order = np.random.default_rng(seed).permutation(len(interactions))
    total_loss = 0.0
    for idx in order:
        interaction = interactions[int(idx)]
        optimizer.zero_grad()
        u_idx = torch.tensor(user_index.to_index(interaction.user_id), dtype=torch.long)
        v_idx = torch.tensor(user_index.to_index(interaction.target_id), dtype=torch.long)
        pred = network.forward_pair(u_idx, v_idx)
        label = torch.tensor(float(interaction.label))
        loss = bce_loss(pred, label)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        total_loss += float(loss.detach())
    return total_loss / len(interactions)


def should_stop_early(
    *,
    best_val_loss: float,
    current_val_loss: float,
    epochs_without_improvement: int,
    patience: int,
) -> tuple[bool, float, int]:
    """Return whether to stop, updated best loss, and stale-epoch count."""
    if current_val_loss < best_val_loss:
        return False, current_val_loss, 0
    stale = epochs_without_improvement + 1
    return stale >= patience, best_val_loss, stale
