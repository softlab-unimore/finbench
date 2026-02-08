"""Dynamic Graph Diffusion Neural Network building blocks."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

if __package__ in (None, ""):
    from ggd import GeneralizedGraphDiffusion  # type: ignore
    from catattn import CatMultiAttn  # type: ignore
else:  # pragma: no cover - executed only when imported as a package
    from .ggd import GeneralizedGraphDiffusion
    from .catattn import CatMultiAttn

__all__ = ["DGDNN"]


class DGDNN(nn.Module):
    """Decoupled Graph Diffusion Neural Network.

    The network alternates between diffusion layers, which propagate information
    over learned diffusion bases, and attention layers, which fuse the diffused
    representation with a persistent skip connection.  The output is a per-node
    classification logit vector.
    """

    def __init__(
        self,
        diffusion_size: Sequence[int],
        embedding_size: Sequence[int],
        embedding_hidden_size: int,
        embedding_output_size: int,
        raw_feature_size: int,
        classes: int,
        layers: int,
        num_nodes: int,
        expansion_step: int,
        num_heads: int,
        active: Sequence[bool],
    ) -> None:
        super().__init__()

        if len(diffusion_size) - 1 != layers:
            raise ValueError("diffusion_size length must be equal to layers + 1")
        if len(embedding_size) != layers:
            raise ValueError("embedding_size length must equal the number of layers")
        if len(active) != layers:
            raise ValueError("active mask must specify one flag per layer")

        self.layers = layers
        self.expansion_step = expansion_step

        # Transition matrices (diffusion bases) and convex combination weights.
        self.T = nn.Parameter(torch.empty(layers, expansion_step, num_nodes, num_nodes))
        self.theta = nn.Parameter(torch.empty(layers, expansion_step))

        # Graph diffusion stages.
        self.diffusion_layers = nn.ModuleList(
            [
                GeneralizedGraphDiffusion(
                    input_dim=diffusion_size[i],
                    output_dim=diffusion_size[i + 1],
                    active=active[i],
                )
                for i in range(layers)
            ]
        )

        # Self-attention fusion after each diffusion step.
        self.cat_attn_layers = nn.ModuleList(
            [
                CatMultiAttn(
                    input_time=embedding_size[i],
                    num_heads=num_heads,
                    hidden_dim=embedding_hidden_size,
                    output_dim=embedding_output_size,
                    use_activation=active[i],
                )
                for i in range(layers)
            ]
        )

        # Project raw node descriptors so that the first attention block can fuse
        # them with the diffused representation.
        self.raw_h_prime = nn.Linear(diffusion_size[0], raw_feature_size)

        # Final classifier operating on the fused embedding.
        self.linear = nn.Linear(embedding_output_size, classes)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize learnable parameters with sensible defaults."""

        nn.init.xavier_uniform_(self.T)
        nn.init.constant_(self.theta, 1.0 / float(self.theta.size(-1)))
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.raw_h_prime.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
        if self.raw_h_prime.bias is not None:
            nn.init.zeros_(self.raw_h_prime.bias)

    def forward(self, X: Tensor, A: Tensor) -> Tensor:
        """Run the Dynamic Graph Diffusion Neural Network forward pass."""

        if X.dim() != 2 or A.dim() != 2:
            raise ValueError("X and A must both be 2-D tensors")
        if A.size(0) != A.size(1):
            raise ValueError("Adjacency matrix A must be square")
        if A.size(0) != X.size(0):
            raise ValueError("Adjacency size must match the number of nodes in X")

        theta_soft = F.softmax(self.theta, dim=-1)

        h = X
        h_prime = X

        for layer_idx in range(self.layers):
            diffusion = self.diffusion_layers[layer_idx]
            attn = self.cat_attn_layers[layer_idx]

            coefficients = theta_soft[layer_idx]
            bases = self.T[layer_idx]
            h = diffusion(coefficients, bases, h, A)

            if layer_idx == 0:
                h_prime = attn(h, self.raw_h_prime(X))
            else:
                h_prime = h_prime + attn(h, h_prime)

        return self.linear(h_prime)



## For those who use fast implementation version.

# class DGDNN(nn.Module):
#     def __init__(
#         self,
#         diffusion_size: list,
#         embedding_size: list,
#         embedding_hidden_size: int,
#         embedding_output_size: int,
#         raw_feature_size: int,
#         classes: int,
#         layers: int,
#         num_heads: int,
#         active: list
#     ):
#         super().__init__()
#         assert len(diffusion_size) - 1 == layers, "Mismatch in diffusion layers"
#         assert len(embedding_size) == layers, "Mismatch in attention layers"

#         self.layers = layers

#         self.diffusion_layers = nn.ModuleList([
#             GeneralizedGraphDiffusion(diffusion_size[i], diffusion_size[i + 1], active[i])
#             for i in range(layers)
#         ])

#         self.cat_attn_layers = nn.ModuleList([
#             CatMultiAttn(
#                 input_time=embedding_size[i],        # e.g., input = concat[h, h_prime] dim
#                 num_heads=num_heads,
#                 hidden_dim=embedding_hidden_size,
#                 output_dim=embedding_output_size,
#                 use_activation=active[i]
#             )
#             for i in range(len(embedding_size))
#         ])
#         # Transform raw features to be divisible by num_heads
#         self.raw_h = nn.Linear(diffusion_size[0], raw_feature_size)

#         self.linear = nn.Linear(embedding_output_size, classes)

#     def forward(self, X: torch.Tensor, A: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             X: [N, F_in]         - node features
#             A: [2, E]            - adjacency (sparse index)
#             W: [E]               - edge weights (if using sparse edge_index)

#         Returns:
#             logits: [N, classes]
#         """
#         z = X
#         h = X

#         for l in range(self.layers):
#             z = self.diffusion_layers[l](z, A, W)  # GeneralizedGraphDiffusion (e.g. GCNConv)
#             if l == 0:
#                 h = self.cat_attn_layers[l](z, self.raw_h(h))
#             else:
#                 h = h + self.cat_attn_layers[l](z, h)

#         return self.linear(h)  # [N, classes]