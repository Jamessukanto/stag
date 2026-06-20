"""Tests for NeuMF network forward pass."""

from __future__ import annotations

import torch
from models.neumf.services.network import NeuMFNetwork, default_mlp_layers


class TestNeuMFNetworkForward:
    def test_forward_returns_scalar_in_unit_interval(self) -> None:
        emb_dim = 4
        n_users = 5
        network = NeuMFNetwork(
            n_users=n_users,
            embedding_dim=emb_dim,
            mlp_layers=default_mlp_layers(emb_dim),
        )
        score = network.forward_pair(
            torch.tensor(0, dtype=torch.long),
            torch.tensor(1, dtype=torch.long),
        )
        assert score.shape == ()
        assert torch.isfinite(score)
        assert 0.0 < float(score) < 1.0

    def test_branch_hidden_dims(self) -> None:
        emb_dim = 8
        n_users = 3
        mlp_layers = default_mlp_layers(emb_dim)
        network = NeuMFNetwork(
            n_users=n_users,
            embedding_dim=emb_dim,
            mlp_layers=mlp_layers,
        )
        u_idx = torch.tensor(0, dtype=torch.long)
        v_idx = torch.tensor(1, dtype=torch.long)

        gmf_hidden = network.gmf_hidden(u_idx, v_idx)
        mlp_hidden = network.mlp_hidden(u_idx, v_idx)

        assert gmf_hidden.shape == (emb_dim,)
        assert mlp_hidden.shape == (emb_dim // 2,)
        assert network.fusion_input_dim == emb_dim + emb_dim // 2

    def test_default_mlp_layers_shape(self) -> None:
        emb_dim = 6
        assert default_mlp_layers(emb_dim) == [12, 6, 3]
