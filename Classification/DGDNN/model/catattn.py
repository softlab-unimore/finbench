"""Concatenation-based multi-head attention utilities."""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["CatMultiAttn"]


class CatMultiAttn(nn.Module):
    """Applies multi-head self-attention over concatenated feature streams.

    The module is intentionally lightweight: each node (or sample) is treated as a
    token whose embedding corresponds to the concatenation of two temporal
    descriptors.  The attention mechanism mixes information across nodes while the
    projection network optionally applies a non-linearity and adjusts the channel
    dimensionality.
    """

    def __init__(
        self,
        input_time: int,
        num_heads: int,
        hidden_dim: int,
        output_dim: int,
        use_activation: bool,
    ) -> None:
        """Construct the concatenation attention block.

        Args:
            input_time: Combined temporal dimension after concatenation ``T1 + T2``.
            num_heads: Number of attention heads.  Must divide ``input_time``.
            hidden_dim: Hidden size of the projection multilayer perceptron.
            output_dim: Output dimensionality returned by the block.
            use_activation: Whether to apply a :class:`~torch.nn.GELU` activation
                between the projection layers.
        """

        super().__init__()
        if input_time % num_heads != 0:
            raise ValueError(
                "input_time must be divisible by num_heads for MultiheadAttention"
            )

        # Multi-head attention treats every node as a token; the actual batch size
        # is handled by temporarily unsqueezing the sequence tensor.
        self.attn = nn.MultiheadAttention(
            embed_dim=input_time, num_heads=num_heads, batch_first=False
        )
        self.norm = nn.LayerNorm(input_time)

        # Projection from the concatenated representation to the downstream space.
        self.proj = nn.Sequential(
            nn.Linear(input_time, hidden_dim),
            nn.GELU() if use_activation else nn.Identity(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _prepare_inputs(self, h: Tensor, h_prime: Tensor) -> Tensor:
        """Validate and concatenate the two feature streams."""

        if h.shape[0] != h_prime.shape[0]:
            raise ValueError("Both inputs must contain the same number of nodes")
        if h.dim() != 2 or h_prime.dim() != 2:
            raise ValueError("Inputs to CatMultiAttn must be two-dimensional tensors")

        concatenated = torch.cat([h, h_prime], dim=1)
        return concatenated.unsqueeze(1)  # (N, 1, input_time)

    def forward(self, h: Tensor, h_prime: Tensor) -> Tensor:
        """Fuse the provided representations.

        Args:
            h: A ``[N, T1]`` tensor representing the primary features for each node.
            h_prime: A ``[N, T2]`` tensor containing auxiliary features.

        Returns:
            ``[N, output_dim]`` tensor describing the refined node representations.
        """

        sequence = self._prepare_inputs(h, h_prime)

        # Run self-attention on the concatenated representation.  The same tensor is
        # used as query/key/value because we only need self-attention here.
        attended, _ = self.attn(sequence, sequence, sequence)
        attended = self.norm(attended)

        # Remove the artificial batch dimension introduced earlier and project.
        fused = attended.squeeze(1)
        return self.proj(fused)