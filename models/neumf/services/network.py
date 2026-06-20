"""NeuMF GMF + MLP + fusion network."""

from __future__ import annotations

import torch
from torch import nn


def default_mlp_layers(embedding_dim: int) -> list[int]:
    """Return the default two-hidden MLP tower sizes."""
    return [embedding_dim * 2, embedding_dim, embedding_dim // 2]


class NeuMFNetwork(nn.Module):
    """NeuMF directional scorer: GMF branch + MLP branch fused through sigmoid."""

    def __init__(
        self,
        *,
        n_users: int,
        embedding_dim: int,
        mlp_layers: list[int],
    ) -> None:
        super().__init__()
        if len(mlp_layers) < 2:
            raise ValueError("mlp_layers must contain at least two hidden sizes")
        if mlp_layers[0] != embedding_dim * 2:
            raise ValueError("mlp_layers[0] must equal 2 * embedding_dim")

        self.embedding_dim = embedding_dim
        self.mlp_layers = list(mlp_layers)
        self.mlp_last_hidden_dim = mlp_layers[-1]
        self.fusion_input_dim = embedding_dim + self.mlp_last_hidden_dim

        self.gmf_user = nn.Embedding(n_users, embedding_dim)
        self.gmf_item = nn.Embedding(n_users, embedding_dim)
        self.mlp_user = nn.Embedding(n_users, embedding_dim)
        self.mlp_item = nn.Embedding(n_users, embedding_dim)

        mlp_modules: list[nn.Module] = []
        in_dim = mlp_layers[0]
        for out_dim in mlp_layers[1:]:
            mlp_modules.append(nn.Linear(in_dim, out_dim))
            mlp_modules.append(nn.ReLU())
            in_dim = out_dim
        self.mlp_tower = nn.Sequential(*mlp_modules)
        self.fusion_head = nn.Linear(self.fusion_input_dim, 1)

    def gmf_hidden(self, u_idx: torch.Tensor, v_idx: torch.Tensor) -> torch.Tensor:
        """Element-wise product of GMF embeddings."""
        gmf: torch.Tensor = self.gmf_user(u_idx) * self.gmf_item(v_idx)
        return gmf

    def mlp_hidden(self, u_idx: torch.Tensor, v_idx: torch.Tensor) -> torch.Tensor:
        """Last hidden layer of the MLP branch after ReLU."""
        mlp_in = torch.cat([self.mlp_user(u_idx), self.mlp_item(v_idx)], dim=-1)
        mlp: torch.Tensor = self.mlp_tower(mlp_in)
        return mlp

    def forward_pair(self, u_idx: torch.Tensor, v_idx: torch.Tensor) -> torch.Tensor:
        """Return scalar s(u->v) in (0, 1)."""
        gmf = self.gmf_hidden(u_idx, v_idx)
        mlp = self.mlp_hidden(u_idx, v_idx)
        fused = torch.cat([gmf, mlp], dim=-1)
        logit = self.fusion_head(fused).squeeze(-1)
        score: torch.Tensor = torch.sigmoid(logit)
        return score
