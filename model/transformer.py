"""Transformer building blocks with RoPE, FlashAttention, and SwiGLU."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Enable FlashAttention if available
_flash_attn_enabled = False
try:
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
        _flash_attn_enabled = True
        print("✅ FlashAttention (SDPA) enabled")
except Exception as e:
    print(f"⚠️  FlashAttention not available: {e}")
    _flash_attn_enabled = False


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Args:
        dim: Model dimension
        eps: Epsilon for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm.

        Args:
            x: Input tensor [..., dim]

        Returns:
            Normalized tensor [..., dim]
        """
        # RMS = sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary positional embeddings.

    Args:
        x: Input tensor [batch, seq_len, n_heads, head_dim]
        freqs_cos: Cosine frequencies [seq_len, head_dim]
        freqs_sin: Sine frequencies [seq_len, head_dim]

    Returns:
        Tensor with RoPE applied [batch, seq_len, n_heads, head_dim]
    """
    # Split into even and odd dimensions
    x1, x2 = x[..., ::2], x[..., 1::2]

    # Reshape freqs for broadcasting: [1, seq_len, 1, head_dim//2]
    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(2)
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(2)

    # Apply rotation
    rotated_x1 = x1 * freqs_cos - x2 * freqs_sin
    rotated_x2 = x1 * freqs_sin + x2 * freqs_cos

    # Interleave back
    rotated_x = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)

    return rotated_x


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE).

    Args:
        dim: Head dimension (should be even)
        max_seq_len: Maximum sequence length
        base: Base for frequency computation
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin for max_seq_len
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("freqs_cos", freqs.cos(), persistent=False)
        self.register_buffer("freqs_sin", freqs.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cos/sin frequencies for given sequence length.

        Args:
            seq_len: Sequence length

        Returns:
            Tuple of (freqs_cos, freqs_sin), each [seq_len, dim]
        """
        return self.freqs_cos[:seq_len], self.freqs_sin[:seq_len]


class TransformerBlock(nn.Module):
    """Transformer block with RoPE, FlashAttention, SwiGLU, and RMSNorm.

    Architecture:
        - Multi-head self-attention with RoPE
        - SwiGLU feed-forward network
        - RMSNorm (post-norm)
        - No biases in linear layers

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability
        max_seq_len: Maximum sequence length for RoPE
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        max_seq_len: int = 2048
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Attention layers (no bias)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # RoPE
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

        # Feed-forward (SwiGLU): no bias
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

        # Normalization
        self.attn_norm = RMSNorm(d_model)
        self.ff_norm = RMSNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dropout_p = dropout

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            attn_mask: Optional attention mask [batch, seq_len, seq_len]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Self-attention with RoPE
        residual = x

        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Apply RoPE to Q and K
        freqs_cos, freqs_sin = self.rope(seq_len)
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)

        # Reshape for attention: [batch, n_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # FlashAttention (uses flash attention 2/3 if available in PyTorch 2.0+)
        # Use scaled_dot_product_attention if available, otherwise fallback
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False
            )
        else:
            # Fallback for older PyTorch versions
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if attn_mask is not None:
                attn_scores = attn_scores + attn_mask
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            attn_output = torch.matmul(attn_probs, v)

        # Reshape and project: [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.o_proj(attn_output)
        attn_output = self.dropout(attn_output)

        # Post-norm and residual
        x = self.attn_norm(residual + attn_output)

        # Feed-forward (SwiGLU)
        residual = x
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        ff_output = self.down_proj(F.silu(gate) * up)  # SwiGLU activation
        ff_output = self.dropout(ff_output)

        # Post-norm and residual
        x = self.ff_norm(residual + ff_output)

        return x
