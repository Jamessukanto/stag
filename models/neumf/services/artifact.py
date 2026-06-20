"""Build and restore ModelArtifact payloads for NeuMF."""

from __future__ import annotations

from typing import Any

import torch
from core.config import Config
from core.serialization import ModelArtifact
from core.types import UserIndex

from models.neumf.services.network import NeuMFNetwork, default_mlp_layers


def build_score_program(*, mlp_layer_count: int) -> list[dict[str, Any]]:
    """Build a score_program matching the two-hidden MLP NeuMF layout."""
    if mlp_layer_count != 2:
        raise ValueError("only two-hidden MLP towers are supported")

    program: list[dict[str, Any]] = [
        {"op": "lookup", "out": "g_u", "table": "source_embeddings", "index": "u"},
        {"op": "lookup", "out": "g_v", "table": "target_embeddings", "index": "v"},
        {"op": "multiply", "out": "gmf", "a": "g_u", "b": "g_v"},
        {"op": "lookup", "out": "m_u", "table": "mlp_source", "index": "u"},
        {"op": "lookup", "out": "m_v", "table": "mlp_target", "index": "v"},
        {"op": "concat", "out": "mlp_in", "inputs": ["m_u", "m_v"]},
        {"op": "dense", "out": "mlp_h1", "input": "mlp_in", "weight": "W1", "bias": "b1"},
        {"op": "relu", "out": "mlp_a1", "input": "mlp_h1"},
        {"op": "dense", "out": "mlp_h2", "input": "mlp_a1", "weight": "W2", "bias": "b2"},
        {"op": "relu", "out": "mlp_a2", "input": "mlp_h2"},
        {"op": "concat", "out": "fused", "inputs": ["gmf", "mlp_a2"]},
        {"op": "dense", "out": "logit", "input": "fused", "weight": "Wout", "bias": "bout"},
        {"op": "sigmoid", "out": "score", "input": "logit"},
    ]
    return program


def _linear_to_lists(linear: torch.nn.Linear) -> tuple[list[list[float]], list[float]]:
    weight = linear.weight.detach().cpu().T.tolist()
    bias = linear.bias.detach().cpu().tolist()
    return weight, bias


def _embedding_to_lists(embedding: torch.nn.Embedding) -> list[list[float]]:
    return embedding.weight.detach().cpu().tolist()


def _extract_mlp_weights(network: NeuMFNetwork) -> dict[str, Any]:
    linear_layers = [module for module in network.mlp_tower if isinstance(module, torch.nn.Linear)]
    if len(linear_layers) != 2:
        raise ValueError("expected exactly two MLP linear layers")

    w1, b1 = _linear_to_lists(linear_layers[0])
    w2, b2 = _linear_to_lists(linear_layers[1])
    wout, bout = _linear_to_lists(network.fusion_head)
    return {
        "W1": w1,
        "b1": b1,
        "W2": w2,
        "b2": b2,
        "Wout": wout,
        "bout": bout,
    }


def build_artifact(
    *,
    config: Config,
    sampling_strategy: str,
    mlp_layers: list[int],
    early_stopping_patience: int,
    user_index: UserIndex,
    network: NeuMFNetwork,
) -> ModelArtifact:
    """Serialize trained NeuMF weights into a standard ModelArtifact."""
    mlp_weights = _extract_mlp_weights(network)
    extra: dict[str, Any] = {
        "mlp_source": _embedding_to_lists(network.mlp_user),
        "mlp_target": _embedding_to_lists(network.mlp_item),
        **mlp_weights,
        "score_program": build_score_program(mlp_layer_count=len(mlp_layers) - 1),
    }
    return ModelArtifact(
        model_name="neumf",
        sampling_strategy=sampling_strategy,
        hyperparameters={
            "embedding_dim": config.embedding_dim,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "negative_downsample_ratio": config.negative_downsample_ratio,
            "mlp_layers": mlp_layers,
            "optimizer": "adam",
            "early_stopping_patience": early_stopping_patience,
            "loss": "bce",
        },
        user_index=user_index,
        source_embeddings=_embedding_to_lists(network.gmf_user),
        target_embeddings=_embedding_to_lists(network.gmf_item),
        extra=extra,
        trained_on_split="train",
    )


def restore_network(artifact: ModelArtifact) -> tuple[NeuMFNetwork, UserIndex]:
    """Rebuild a NeuMFNetwork from a saved artifact."""
    if artifact.model_name != "neumf":
        raise ValueError(f"expected model_name 'neumf', got {artifact.model_name!r}")

    hyper = artifact.hyperparameters
    embedding_dim = int(hyper["embedding_dim"])
    mlp_layers = list(hyper.get("mlp_layers", default_mlp_layers(embedding_dim)))
    n_users = len(artifact.user_index)

    network = NeuMFNetwork(
        n_users=n_users,
        embedding_dim=embedding_dim,
        mlp_layers=mlp_layers,
    )
    network.gmf_user.weight.data = torch.tensor(artifact.source_embeddings, dtype=torch.float32)
    network.gmf_item.weight.data = torch.tensor(artifact.target_embeddings, dtype=torch.float32)
    network.mlp_user.weight.data = torch.tensor(artifact.extra["mlp_source"], dtype=torch.float32)
    network.mlp_item.weight.data = torch.tensor(artifact.extra["mlp_target"], dtype=torch.float32)

    linear_layers = [module for module in network.mlp_tower if isinstance(module, torch.nn.Linear)]
    for layer, weight_key, bias_key in (
        (linear_layers[0], "W1", "b1"),
        (linear_layers[1], "W2", "b2"),
    ):
        layer.weight.data = torch.tensor(artifact.extra[weight_key], dtype=torch.float32).T
        layer.bias.data = torch.tensor(artifact.extra[bias_key], dtype=torch.float32)

    network.fusion_head.weight.data = torch.tensor(artifact.extra["Wout"], dtype=torch.float32).T
    network.fusion_head.bias.data = torch.tensor(artifact.extra["bout"], dtype=torch.float32)

    return network, artifact.user_index
