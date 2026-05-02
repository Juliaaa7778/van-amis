"""
Ising model specific FlashAR model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_ar import FlashARModelBase

try:
    from flash_attn import flash_attn_with_kvcache
    FLASH_ATTN_KV_AVAILABLE = True
except ImportError:
    FLASH_ATTN_KV_AVAILABLE = False

def unpatch(patched_samples: torch.Tensor, L: int, patch_size: int = 2) -> torch.Tensor:

    B, n = patched_samples.shape
    p = patch_size
    ph = pw = L // p
    bits = p * p

    # Decode each patch token into p*p bits using bit shifting
    shifts = torch.arange(bits - 1, -1, -1, device=patched_samples.device)
    bits_tensor = ((patched_samples.unsqueeze(-1) >> shifts) & 1)  # [B, n, bits] ∈ {0,1}

    # Reshape bits to spatial patches [B, ph, pw, p, p]
    patches = bits_tensor.view(B, ph, pw, p, p)

    # Rearrange patches to full grid [B, L, L]
    grid = patches.permute(0, 1, 3, 2, 4).contiguous().view(B, L, L)

    return grid.view(B, L * L).float()

class IsingFlashARModelPatchMultiTemp(FlashARModelBase):
    """
    Multi-temperature patch-based model for an L x L Ising lattice.

    Identical structure to IsingFlashARModelMultiTemp but operates on
    patch tokens (vocab size M^(patch_size^2)) instead of individual spins.

    Sequence seen by the Transformer (total length P = (L/patch_size)^2):
        input:  [beta,    patch_0, patch_1, ..., patch_{P-2}]
        target: [patch_0, patch_1, patch_2, ..., patch_{P-1}]
    """
    def __init__(
        self,
        L: int,
        beta_bins: int = 10,
        M: int = 2,
        patch_size: int = 2,
        token_dim: int = 2,
        pos_dim: int = 30,
        n_layers: int = 1,
        n_heads: int = 1,
        device: str = "cuda:0",
        flash_dtype: str = "float16",
        **kwargs,
    ):
        self.L = L
        self.patch_size = patch_size
        self.base_M = M
        self.token_dim  = token_dim

        # Calculate number of patches and effective vocabulary size
        n_patches = (L // patch_size) ** 2
        M_patched = M ** (patch_size ** 2)

        super().__init__(
            n_sites=n_patches,
            M=M_patched,
            token_dim=token_dim,
            pos_dim=pos_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            device=device,
            flash_dtype=flash_dtype,
        )

        self.wbe = nn.Embedding(beta_bins, token_dim).to(device)
        self.pos_emb = nn.Embedding(n_patches + 1, pos_dim).to(device)  # +1 for beta pos

    # --- _forward (same structure as the single-site MultiTemp) ---
    def _forward(self, x: torch.Tensor, beta_idx: torch.Tensor, target=None):
        B   = x.size(0)
        T_x = x.size(1)
        T   = T_x + 1

        if beta_idx.dim() == 2:
            beta_idx = beta_idx.squeeze(1)

        pos         = torch.arange(T, dtype=torch.long, device=self.device)
        pos_emb_all = self.pos_emb(pos)

        # beta
        beta_tok = self.wbe(beta_idx)
        beta_pos = pos_emb_all[0].unsqueeze(0).expand(B, -1)
        beta_h   = torch.cat([beta_tok, beta_pos], dim=-1).unsqueeze(1)  # [B, 1, d_model]

        # patches
        if T_x > 0:
            site_tok = self.fc_in(x)
            site_pos = pos_emb_all[1:T].unsqueeze(0).expand(B, -1, -1)
            site_h   = torch.cat([site_tok, site_pos], dim=-1)
            h = torch.cat([beta_h, site_h], dim=1)
        else:
            h = beta_h

        for layer in self.transformer.layers:
            h = layer(h)

        logits = self.fc_out(h)

        if target is not None:
            loss = F.cross_entropy(logits.transpose(1, 2), target, reduction="none")
        else:
            loss = None

        return logits, loss

    # --- log_prob ---
    def log_prob(self, spins: torch.Tensor, beta_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute log P(spins | beta).   spins are patch tokens [B, n_patches].
        """
        x_input = spins[:, :-1]
        target  = spins
        _, loss = self._forward(x_input, beta_idx, target)
        return -loss.sum(dim=1)
    
    # --- sample with KV-cache ---
    @torch.no_grad()
    def sample(self, batch_size: int, beta_idx: torch.Tensor) -> torch.Tensor:
        """
        Autoregressive sampling with KV-cache.
        Returns patch tokens [B, n_patches];  use unpatch() to get full spins.
        """
        device = self.device
        B = batch_size
        N = self.n          # n_patches
        H = self.n_heads
        D = self.d_model // H

        if isinstance(beta_idx, int):
            beta_idx = torch.full((B,), beta_idx, dtype=torch.long, device=device)
        elif not isinstance(beta_idx, torch.Tensor):
            beta_idx = torch.tensor(beta_idx, dtype=torch.long, device=device)
        if beta_idx.dim() == 0:
            beta_idx = beta_idx.expand(B)
        beta_idx = beta_idx.to(device)

        samples = torch.zeros((B, N), dtype=torch.long, device=device)

        k_caches = [
            torch.zeros(B, N + 1, H, D, device=device, dtype=self.dtype)
            for _ in range(self.n_layers)
        ]
        v_caches      = [torch.zeros_like(k) for k in k_caches]
        cache_seqlens = torch.zeros(B, dtype=torch.int32, device=device)

        # step 0 — beta token
        beta_tok = self.wbe(beta_idx)
        beta_pos = self.pos_emb(torch.tensor([0], device=device))
        h = torch.cat([beta_tok, beta_pos.expand(B, -1)], dim=-1).unsqueeze(1)

        h = self._process_layers_with_cache(h, k_caches, v_caches, cache_seqlens)
        cache_seqlens += 1

        logits = self.fc_out(h).squeeze(1)
        samples[:, 0] = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(1)

        # steps 1 .. N-1
        for i in range(1, N):
            prev_tok = samples[:, i - 1].unsqueeze(1)
            site_tok = self.fc_in(prev_tok)
            site_pos = self.pos_emb(torch.tensor([i], device=device))
            h = torch.cat([site_tok, site_pos.unsqueeze(0).expand(B, 1, -1)], dim=-1)

            h = self._process_layers_with_cache(h, k_caches, v_caches, cache_seqlens)
            cache_seqlens += 1

            logits = self.fc_out(h).squeeze(1)
            samples[:, i] = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(1)

        return samples
    
    
    # --- KV-cache helper ---
    def _process_layers_with_cache(self, h, k_caches, v_caches, cache_seqlens):
        B = h.shape[0]
        H = self.n_heads
        D = self.d_model // H

        for lyr, layer in enumerate(self.transformer.layers):
            qkv = F.linear(
                layer.norm1(h),
                layer.attn.in_proj_weight,
                layer.attn.in_proj_bias,
            )
            q, k_new, v_new = qkv.chunk(3, dim=-1)

            q     = q.view(B, 1, H, D).to(self.dtype)
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

        return h

    @classmethod
    def from_args(cls, args) -> "IsingFlashARModelPatchMultiTemp":
        """
        Create model from argument namespace.

        Args:
            args: Argument namespace with model configuration.

        Returns:
            Configured IsingFlashARModelPatch instance.
        """
        return cls(
            L=args.L,
            beta_bins=getattr(args, "beta_bins", 10),
            M=args.M,
            patch_size=getattr(args, "patch_size", 2),
            token_dim=args.token_dim,
            pos_dim=args.pos_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            device=args.device,
            flash_dtype=getattr(args, "flash_dtype", "float16"),
        )
