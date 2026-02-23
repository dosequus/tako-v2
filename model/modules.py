"""HRM module components: Input Network, L-module, H-module."""

import torch
import torch.nn as nn
from model.transformer import TransformerBlock, RotaryEmbedding


class InputNetwork(nn.Module):
    """Input network: tokenized state → dense embeddings with RoPE.

    Args:
        vocab_size: Size of token vocabulary
        d_model: Model dimension
        max_seq_len: Maximum sequence length
    """

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int = 2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

        # Scaling factor (common in transformers)
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed input tokens.

        Args:
            x: Token indices [batch, seq_len]

        Returns:
            Embeddings [batch, seq_len, d_model]
        """
        # Embed and scale
        emb = self.embedding(x) * self.scale
        return emb


class LModule(nn.Module):
    """L-module: fast tactical reasoning, updates every timestep.

    Inputs are combined via element-wise addition:
        input = z_L_prev + z_H + x_emb

    Then passed through multiple stacked transformer blocks.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        n_layers: Number of transformer layers to stack
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        z_L_prev: torch.Tensor,
        z_H: torch.Tensor,
        x_emb: torch.Tensor
    ) -> torch.Tensor:
        """Update L-module state.

        Args:
            z_L_prev: Previous L-state [batch, seq_len, d_model]
            z_H: Current H-state [batch, seq_len, d_model]
            x_emb: Input embeddings [batch, seq_len, d_model]

        Returns:
            Updated z_L [batch, seq_len, d_model]
        """
        # Element-wise addition of inputs
        combined = z_L_prev + z_H + x_emb

        # Pass through stacked transformer layers
        z_L = combined
        for layer in self.layers:
            z_L = layer(z_L)

        return z_L


class HModule(nn.Module):
    """H-module: slow strategic reasoning, updates every T timesteps.

    Inputs are combined via element-wise addition:
        input = z_H_prev + z_L

    Then passed through multiple stacked transformer blocks (identical architecture to L-module).

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        n_layers: Number of transformer layers to stack
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        z_H_prev: torch.Tensor,
        z_L: torch.Tensor
    ) -> torch.Tensor:
        """Update H-module state.

        Args:
            z_H_prev: Previous H-state [batch, seq_len, d_model]
            z_L: Current L-state [batch, seq_len, d_model]

        Returns:
            Updated z_H [batch, seq_len, d_model]
        """
        # Element-wise addition of inputs
        combined = z_H_prev + z_L

        # Pass through stacked transformer layers
        z_H = combined
        for layer in self.layers:
            z_H = layer(z_H)

        return z_H


import math
