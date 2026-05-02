"""
FlashAttention-based Autoregressive Model base class.

This module provides a base class for autoregressive models used in
variational neural annealing for statistical mechanics problems.
"""

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import TransformerStack

try:
    from flash_attn import flash_attn_with_kvcache
    FLASH_ATTN_KV_AVAILABLE = True
except ImportError:
    FLASH_ATTN_KV_AVAILABLE = False


class FlashARModelBase(nn.Module, ABC):
    """
    Base class for FlashAttention-based autoregressive models.

    This base class provides common functionality for autoregressive models
    used in statistical mechanics applications (SK, EA, etc.).

    Subclasses should implement:
        - _get_num_sites(): Return the number of spin sites
        - Any problem-specific methods

    Args:
        n_sites: Number of spin sites (sequence length).
        M: Number of categories for embeddings (default: 2 for binary spins).
        token_dim: Dimension of token embeddings.
        pos_dim: Dimension of positional embeddings.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads.
        device: Device to place the model on.
        flash_dtype: Data type for FlashAttention ('float16' or 'bfloat16').
    """

    def __init__(
        self,
        n_sites: int,
        M: int = 2,
        token_dim: int = 2,
        pos_dim: int = 30,
        n_layers: int = 1,
        n_heads: int = 1,
        device: str = "cuda:0",
        flash_dtype: str = "float16",
    ):
        super().__init__()

        self.n = n_sites
        self.M = M
        self.token_dim = token_dim
        self.pos_dim = pos_dim
        self.d_model = token_dim + pos_dim
        self.d_ff = self.d_model * 4
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.device = device
        self.dtype = getattr(torch, flash_dtype)

        # Embeddings
        self.fc_in = nn.Embedding(M, token_dim)
        self.pos_emb = nn.Embedding(n_sites, pos_dim)

        # Transformer backbone
        self.transformer = TransformerStack(
            n_layers, self.d_model, n_heads, self.d_ff, n_sites, dropout=0.0
        )

        # Output projection
        self.fc_out = nn.Linear(self.d_model, M)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with small random values."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing log probabilities for each position.

        Args:
            x: Input tensor of shape [B, n] with values in {0, 1, ..., M-1}.

        Returns:
            Log probabilities of shape [B, n, M].
        """
        B, T = x.size()

        # Shift input for autoregressive prediction (prepend start token)
        x = torch.cat(
            (torch.ones(B, 1, device=self.device, dtype=torch.long), x[:, :-1]),
            dim=1,
        )

        # Positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=self.device)
        pos_emb = self.pos_emb(pos).unsqueeze(0).expand(B, -1, -1)

        # Token embeddings
        x = self.fc_in(x)

        # Concatenate token and position embeddings
        h_cat = torch.cat((x, pos_emb), dim=-1)

        # Transformer
        h = self.transformer(h_cat)

        # Output projection
        logits = self.fc_out(h)
        logits = F.log_softmax(logits, dim=-1)

        return logits

    def log_prob(self, spins: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of spin configurations.

        Args:
            spins: Spin configurations of shape [B, n] with values in {-1, +1}.

        Returns:
            Log probabilities of shape [B].
        """
        B = spins.size(0)

        # Convert {-1, +1} to {0, 1}
        x01 = ((spins.view(B, self.n) + 1) // 2).long()

        # Get log probabilities
        x_hat = self.forward(x01)  # [B, n, M]

        # Gather log prob for each position
        log_prob = x_hat.gather(2, x01.unsqueeze(-1)).squeeze(-1)  # [B, n]
        log_prob = log_prob.sum(dim=1)  # [B]

        return log_prob

    @torch.no_grad()
    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Generate samples autoregressively with KV-cache.

        Args:
            batch_size: Number of samples to generate.

        Returns:
            Sampled spin configurations of shape [B, n] with values in {-1, +1}.
        """
        B = batch_size
        n = self.n
        H = self.n_heads
        D = self.d_model // self.n_heads

        samples = torch.ones((B, n), dtype=torch.long, device=self.device)

        if FLASH_ATTN_KV_AVAILABLE:
            return self._sample_with_kv_cache(B, n, H, D, samples)
        else:
            return self._sample_naive(B, n, samples)

    def _sample_with_kv_cache(
        self,
        B: int,
        n: int,
        H: int,
        D: int,
        samples: torch.Tensor,
    ) -> torch.Tensor:
        """Sample using FlashAttention KV-cache for efficiency."""
        # Initialize KV caches for each layer
        k_caches = [
            torch.zeros(B, n, H, D, device=self.device, dtype=self.dtype)
            for _ in range(self.n_layers)
        ]
        v_caches = [torch.zeros_like(k) for k in k_caches]
        cache_seqlens = torch.zeros(B, dtype=torch.int32, device=self.device)

        for i in range(n):
            # Get previous token
            if i > 0:
                prev = samples[:, i - 1].unsqueeze(1)
            else:
                prev = torch.ones(B, 1, device=self.device, dtype=torch.long)

            # Embeddings
            h = self.fc_in(prev.long())
            pos = torch.tensor([i], dtype=torch.long, device=self.device)
            pos_emb = self.pos_emb(pos).unsqueeze(0).expand(B, -1, -1)
            h = torch.cat((h, pos_emb), dim=-1)

            # Process through transformer layers with KV-cache
            for lyr, layer in enumerate(self.transformer.layers):
                qkv = F.linear(
                    layer.norm1(h),
                    layer.attn.in_proj_weight,
                    layer.attn.in_proj_bias,
                )
                q, k_new, v_new = qkv.chunk(3, dim=-1)

                q = q.view(B, 1, H, D).to(self.dtype)
                k_new = k_new.view(B, 1, H, D).to(self.dtype)
                v_new = v_new.view(B, 1, H, D).to(self.dtype)

                out = flash_attn_with_kvcache(
                    q=q,
                    k_cache=k_caches[lyr],
                    v_cache=v_caches[lyr],
                    k=k_new,
                    v=v_new,
                    cache_seqlens=cache_seqlens,
                    causal=True,
                ).float()

                out = out.view(B, 1, self.d_model)
                h = h + out
                h = layer.norm2(h)
                h = h + layer.ff(h)

            cache_seqlens += 1

            # Sample from output distribution
            logits = self.fc_out(h).squeeze(1)
            probs = F.softmax(logits, dim=-1)
            samples[:, i] = torch.multinomial(probs, num_samples=1).squeeze(1)

        # Convert {0, 1} to {-1, +1}
        samples = samples * 2 - 1
        return samples.float()

    def _sample_naive(self, B: int, n: int, samples: torch.Tensor) -> torch.Tensor:
        """Sample without KV-cache (slower but doesn't require flash_attn)."""
        for i in range(n):
            # Build sequence so far
            if i > 0:
                seq = samples[:, :i]
                x01 = seq.clone()
            else:
                x01 = torch.ones(B, 1, device=self.device, dtype=torch.long)

            # Get predictions
            logits = self.forward(x01)[:, -1, :]  # [B, M]
            probs = F.softmax(logits, dim=-1)
            samples[:, i] = torch.multinomial(probs, num_samples=1).squeeze(1)

        # Convert {0, 1} to {-1, +1}
        samples = samples * 2 - 1
        return samples.float()
