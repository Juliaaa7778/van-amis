"""
Build sample pool from a checkpoint trained with train_ddp.py.
Supports Z2 symmetry for models trained with --z2 flag.

Usage as script:
    # With Z2 symmetry (must match training)
    python build_sample_pool_ddp.py \
        --model_path out/.../L32_OBC_ddp_seed7_final.pt \
        --device cuda:0 --samples_per_beta 10000 --z2

    # Without Z2 (default, backward compatible)
    python build_sample_pool_ddp.py \
        --model_path out/.../L32_OBC_ddp_seed7_final.pt \
        --device cuda:0 --samples_per_beta 10000
"""
import os
import math
import argparse
import torch
import numpy as np

from model_ising import IsingFlashARModelPatchMultiTemp, unpatch
from hamiltonian_ising import IsingHamiltonian


class SamplePoolBuilder:

    def __init__(self, model_path, device="cuda:0", boundary="OBC",
                 z2=False, patch_size_override=None, **model_kwargs):
        """
        Parameters
        ----------
        z2 : bool
            If True, apply Z2 symmetry: random spin flips during sampling,
            and log_prob = logsumexp(log q(s), log q(-s)) - log(2).
            Must match the --z2 flag used during training.
        """
        self.model_path = model_path
        self.device = torch.device(device)
        self.boundary = boundary
        self.model_kwargs = model_kwargs
        self.z2 = z2
        self.patch_size_override = patch_size_override
        self.model = None
        self.betas = None
        self.cfg = None
        self.max_token = None  # set after loading model if z2

    def load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.betas = checkpoint['betas']
        num_temps = len(self.betas)

        if 'model_config' in checkpoint:
            self.cfg = checkpoint['model_config']
        elif self.model_kwargs:
            self.cfg = {
                'L': self.model_kwargs.get('L', 32),
                'patch_size': self.model_kwargs.get('patch_size', 2),
                'token_dim': self.model_kwargs.get('token_dim', 64),
                'pos_dim': self.model_kwargs.get('pos_dim', 256),
                'n_layers': self.model_kwargs.get('n_layers', 2),
                'n_heads': self.model_kwargs.get('n_heads', 4),
            }
            print(f"[!] No model_config in checkpoint, using fallback args")
        else:
            raise RuntimeError(
                "No model_config in checkpoint. Pass L, patch_size, etc. to constructor."
            )

        self.cfg['boundary'] = checkpoint.get('boundary', self.boundary)

        if self.patch_size_override is not None:
            self.cfg['patch_size'] = self.patch_size_override

        self.model = IsingFlashARModelPatchMultiTemp(
            L=self.cfg['L'],
            beta_bins=num_temps,
            M=2,
            patch_size=self.cfg['patch_size'],
            token_dim=self.cfg['token_dim'],
            pos_dim=self.cfg['pos_dim'],
            n_layers=self.cfg['n_layers'],
            n_heads=self.cfg['n_heads'],
            device=str(self.device),
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Z2 setup
        if self.z2:
            self.max_token = 2 ** (self.cfg['patch_size'] ** 2) - 1
            print(f"[Z2] Enabled: max_token={self.max_token}, "
                  f"patch_size={self.cfg['patch_size']}")

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded: L={self.cfg['L']}, patch={self.cfg['patch_size']}, "
              f"layers={self.cfg['n_layers']}, heads={self.cfg['n_heads']}, "
              f"params={n_params:,}, boundary={self.cfg['boundary']}, z2={self.z2}")
        print(f"Betas: {self.betas}")
        return self.model, self.cfg

    def _z2_sample(self, samples, batch_size):
        """Apply random Z2 flip to each sample with 50% probability."""
        flip = torch.randint(2, (batch_size, 1), device=samples.device, dtype=samples.dtype)
        return samples * (1 - flip) + (self.max_token - samples) * flip

    def _z2_log_prob(self, samples, beta_indices):
        """log q_z2(s) = logsumexp(log q(s), log q(-s)) - log(2)."""
        log_p = self.model.log_prob(samples, beta_indices)
        samples_inv = self.max_token - samples
        log_p_inv = self.model.log_prob(samples_inv, beta_indices)
        return torch.logsumexp(torch.stack([log_p, log_p_inv]), dim=0) - math.log(2)

    @torch.no_grad()
    def build_sample_pool(self, samples_per_beta=10000, batch_size=1000):
        if self.model is None:
            self.load_model()

        L = self.cfg['L']
        patch_size = self.cfg['patch_size']
        betas = self.betas
        K = len(betas)
        total_samples = K * samples_per_beta

        hamiltonian = IsingHamiltonian(
            L=L, device=self.device, boundary=self.cfg['boundary']
        )

        print(f"Building sample pool: {K} temps x {samples_per_beta} = {total_samples} samples"
              f" (z2={self.z2})")

        all_energies = np.zeros(total_samples, dtype=np.float32)
        all_log_probs_matrix = np.zeros((total_samples, K), dtype=np.float32)
        all_configs = np.zeros((total_samples, L, L), dtype=np.int8)

        n_full_batches = samples_per_beta // batch_size
        remainder = samples_per_beta % batch_size
        global_idx = 0

        for b_idx, beta in enumerate(betas):
            print(f"  beta={beta:.4f}: ", end="", flush=True)
            count = 0

            for batch_i in range(n_full_batches + (1 if remainder > 0 else 0)):
                bs = batch_size if batch_i < n_full_batches else remainder
                if bs == 0:
                    continue

                beta_indices = torch.full((bs,), b_idx, dtype=torch.long, device=self.device)
                samples = self.model.sample(bs, beta_indices)

                # Z2: random flip during sampling
                if self.z2:
                    samples = self._z2_sample(samples, bs)

                # Unpack to spin configs
                if patch_size > 1:
                    cfgs = unpatch(samples, L, patch_size=patch_size).view(bs, L, L)
                else:
                    cfgs = samples.view(bs, L, L)

                spins = cfgs.float() * 2.0 - 1.0
                energy = hamiltonian.energy(spins)

                all_energies[global_idx:global_idx + bs] = energy.cpu().numpy()
                all_configs[global_idx:global_idx + bs] = cfgs.cpu().numpy().astype(np.int8)

                # Cross-temperature log probs (Z2 symmetric if enabled)
                for k in range(K):
                    k_indices = torch.full((bs,), k, dtype=torch.long, device=self.device)
                    if self.z2:
                        log_p_k = self._z2_log_prob(samples, k_indices)
                    else:
                        log_p_k = self.model.log_prob(samples, k_indices)
                    all_log_probs_matrix[global_idx:global_idx + bs, k] = log_p_k.cpu().numpy()

                global_idx += bs
                count += bs

            print(f"{count} samples")

        pool = {
            'energies': all_energies,
            'betas': np.array(betas, dtype=np.float64),
            'log_probs_matrix': all_log_probs_matrix,
            'configs': all_configs,
        }
        print(f"Done: {global_idx} total samples")
        return pool

    def save_sample_pool(self, pool, save_path=None):
        if save_path is None:
            out_dir = os.path.dirname(self.model_path) or "."
            save_path = os.path.join(out_dir, "sample_pool.npz")
        np.savez_compressed(save_path, **pool)
        size_mb = os.path.getsize(save_path) / 1024 / 1024
        print(f"Saved to {save_path} ({size_mb:.1f} MB)")

    @staticmethod
    def load_sample_pool(load_path):
        data = np.load(load_path)
        pool = {k: data[k] for k in data.files}
        print(f"Loaded {load_path}, {len(pool['energies'])} samples")
        return pool


def parse_args():
    p = argparse.ArgumentParser(description="Build sample pool from train_ddp.py checkpoint")
    p.add_argument("--model_path", type=str, required=True, help="Path to *_final.pt")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--samples_per_beta", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=1000)
    p.add_argument("--output", type=str, default=None, help="Output .npz path")
    p.add_argument("--boundary", type=str, default="OBC", choices=["OBC", "PBC"])
    p.add_argument("--z2", action="store_true",
                    help="Enable Z2 symmetry (must match training)")

    g = p.add_argument_group("Architecture (only needed for old checkpoints)")
    g.add_argument("--L", type=int, default=32)
    g.add_argument("--patch_size", type=int, default=2)
    g.add_argument("--token_dim", type=int, default=64)
    g.add_argument("--pos_dim", type=int, default=256)
    g.add_argument("--n_layers", type=int, default=2)
    g.add_argument("--n_heads", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()

    builder = SamplePoolBuilder(
        model_path=args.model_path,
        device=args.device,
        boundary=args.boundary,
        z2=args.z2,
        L=args.L,
        patch_size=args.patch_size,
        token_dim=args.token_dim,
        pos_dim=args.pos_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    )

    pool = builder.build_sample_pool(
        samples_per_beta=args.samples_per_beta,
        batch_size=args.batch_size,
    )

    builder.save_sample_pool(pool, save_path=args.output)


if __name__ == "__main__":
    main()