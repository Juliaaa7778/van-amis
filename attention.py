"""
Attention mechanisms for autoregressive models.

This module contains FlashAttention-based multi-head attention and
transformer layer implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn.flash_attn_interface import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class FAMultiHeadAttention(nn.Module):
    """
    FlashAttention-based Multi-Head Attention.

    This implementation uses FlashAttention for efficient attention computation
    on GPUs with causal masking for autoregressive models.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        seqlen: Maximum sequence length.
        dropout: Dropout rate (default: 0.0).
    """

    def __init__(self, d_model: int, n_heads: int, seqlen: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.dim_head = d_model // n_heads
        self.d_model = d_model

        # Combined QKV projection
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, L, d_model].

        Returns:
            Output tensor of shape [B, L, d_model].
        """
        B, L, _ = x.size()

        # QKV projection
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for attention: [B, L, H, dim_head]
        q = q.view(B, L, self.n_heads, self.dim_head)
        k = k.view(B, L, self.n_heads, self.dim_head)
        v = v.view(B, L, self.n_heads, self.dim_head)

        if FLASH_ATTN_AVAILABLE:
            # Use FlashAttention
            out = flash_attn_func(
                q=q.half(),
                k=k.half(),
                v=v.half(),
                causal=True,
            ).float()
        else:
            # Fallback to standard attention
            out = self._standard_attention(q, k, v)

        # Merge heads: [B, L, d_model]
        out = out.reshape(B, L, self.n_heads * self.dim_head)
        return out

    def _standard_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Standard scaled dot-product attention with causal mask."""
        B, L, H, D = q.shape

        # Reshape for batched matrix multiply
        q = q.transpose(1, 2)  # [B, H, L, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scale = D ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask
        mask = torch.triu(torch.ones(L, L, device=q.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape back: [B, L, H, D]
        out = out.transpose(1, 2)
        return out


class FeedForward(nn.Module):
    """
    Feed-forward network for transformer layers.

    Args:
        d_model: Model dimension.
        d_ff: Hidden dimension (typically 4 * d_model).
        dropout: Dropout rate (default: 0.0).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)


class TransformerLayer(nn.Module):
    """
    Single transformer layer with pre-norm architecture.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
        seqlen: Maximum sequence length.
        dropout: Dropout rate (default: 0.0).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        seqlen: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = FAMultiHeadAttention(d_model, n_heads, seqlen, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        h = self.norm1(x)
        a = self.attn(h)
        x = x + a
        x = self.norm2(x)
        x = x + self.ff(x)
        return x


class TransformerStack(nn.Module):
    """
    Stack of transformer layers.

    Args:
        n_layers: Number of transformer layers.
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
        seqlen: Maximum sequence length.
        dropout: Dropout rate (default: 0.0).
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        seqlen: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff, seqlen, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x
