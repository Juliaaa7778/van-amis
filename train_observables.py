#!/usr/bin/env python3
"""
Train FlashARModel on 2D Ising model to learn the Boltzmann distribution.
(Multi-temperature version with improved LINEAR annealing)

Changes from original:
- Z2 symmetry enforcement via LogProbWrapper: q(s) = q(-s)
- Observable statistics: E/N, |M|, Cv evaluated alongside free energy
- total_batch_size argument replaces samples_per_temp (auto-divided by num_temps)
- Divisibility check and warning for total_batch_size
- Online accumulation statistics in evaluate_vfe (O(1) memory)
- Adaptive eval_batch_size based on L
- Checkpoint saves model_config and boundary
"""
import argparse
import time
from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from hamiltonian_ising import IsingHamiltonian
from model_ising import IsingFlashARModelPatchMultiTemp, unpatch
from muon import get_optimizer
from utils import (
    Logger,
    CheckpointManager,
    count_parameters,
    ensure_dir,
    get_device,
    setup_seed,
)

# Default beta values (can be overridden by command line arguments)
DEFAULT_BETAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# ============================================================
# Z2 Symmetry Wrapper
# ============================================================

class LogProbWrapper(nn.Module):
    """
    Wraps IsingFlashARModelPatchMultiTemp to enforce Z2 symmetry:
        log q(s) = log( [q_raw(s) + q_raw(-s)] / 2 )

    All other methods (sample, parameters, etc.) are delegated to the
    underlying model unchanged.  The wrapper is transparent to the
    optimizer and checkpoint machinery.
    """

    def __init__(self, base_model: IsingFlashARModelPatchMultiTemp):
        super().__init__()
        self.model = base_model

    # ── forward: delegate to base model ──────────────────────────────
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # ── sampling: unchanged (samples from raw model) ─────────────────
    def sample(self, batch_size: int, beta_indices: torch.Tensor) -> torch.Tensor:
        return self.model.sample(batch_size, beta_indices)

    # ── log_prob: symmetrised ─────────────────────────────────────────
    def log_prob(self, samples: torch.Tensor, beta_indices: torch.Tensor) -> torch.Tensor:
        """
        Returns log( [q_raw(s) + q_raw(-s)] / 2 ) for each sample.

        For patch tokens: Z2 flip = bitwise XOR with all-ones mask (M_patched - 1).
        e.g. patch_size=2 → M_patched=16, mask=15 (0b1111)
        This correctly flips every spin bit packed inside the token.
        """
        M_patched    = self.model.base_M ** (self.model.patch_size ** 2)
        flipped      = (M_patched - 1) ^ samples          # bitwise XOR flip
        log_q_s      = self.model.log_prob(samples, beta_indices)
        log_q_s_flip = self.model.log_prob(flipped, beta_indices)

        # log-sum-exp trick for numerical stability
        # log( (exp(a) + exp(b)) / 2 ) = log(exp(a) + exp(b)) - log(2)
        log_q_sym = torch.logaddexp(log_q_s, log_q_s_flip) - np.log(2.0)
        return log_q_sym

    # ── expose underlying state_dict / parameters ────────────────────
    def state_dict(self, **kwargs):
        return self.model.state_dict(**kwargs)

    def load_state_dict(self, state_dict, **kwargs):
        return self.model.load_state_dict(state_dict, **kwargs)


# ============================================================
# Adaptive eval batch size
# ============================================================

def get_eval_params(L: int, eval_samples: int, eval_batch_size_hint: int) -> int:
    """
    Return an appropriate eval batch size based on lattice size L.
    Prevents OOM for large lattices without chunking.
    """
    if L <= 32:
        batch_size = 200
    elif L <= 64:
        batch_size = 100
    elif L <= 128:
        batch_size = 25
    else:
        batch_size = 10

    # Respect the user-provided hint if it is smaller
    batch_size = min(batch_size, eval_batch_size_hint)

    # Ensure divisibility into eval_samples
    if eval_samples % batch_size != 0:
        # Find largest divisor of eval_samples that is <= batch_size
        for b in range(batch_size, 0, -1):
            if eval_samples % b == 0:
                batch_size = b
                break

    return batch_size


# ============================================================
# Annealing (unchanged from original)
# ============================================================

def beta_to_idx(beta: float, betas: List[float]) -> int:
    return betas.index(beta)

def idx_to_beta(idx: int, betas: List[float]) -> float:
    return betas[idx]

def get_focused_critical_annealing_factor(step: int, total_steps: int) -> float:
    """
    Three-phase annealing strategy optimized for critical region near β_c ≈ 0.44.

    Phase 1 (0–20%):  factor 0.00 → 0.35
    Phase 2 (20–70%): factor 0.35 → 0.55  (critical region)
    Phase 3 (70–100%):factor 0.55 → 1.00
    """
    if total_steps <= 0:
        return 1.0

    progress = min(step / total_steps, 1.0)

    if progress < 0.2:
        return 0.35 * (progress / 0.2)
    elif progress < 0.7:
        ratio = (progress - 0.2) / 0.5
        return 0.35 + (0.55 - 0.35) * ratio
    else:
        ratio = (progress - 0.7) / 0.3
        return 0.55 + (1.0 - 0.55) * ratio


# ============================================================
# Argument parsing
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FlashARModel on Ising model for Boltzmann distribution (Multi-temperature)"
    )

    # System parameters
    parser.add_argument("--L", type=int, default=64, help="Number of spins (linear size)")
    parser.add_argument("--boundary", type=str, default="OBC", choices=["OBC", "PBC"],
                        help="Boundary condition (OBC: open, PBC: periodic)")
    parser.add_argument("--betas", type=float, nargs='+', default=None,
                        help="List of inverse temperatures (default: 0.1 to 1.0)")
    parser.add_argument("--betas_file", type=str, default=None,
                        help="Path to a text file containing beta values (one per line, "
                             "lines starting with '#' are ignored). "
                             "Takes priority over --betas and DEFAULT_BETAS.")
    parser.add_argument("--annealing_steps", type=int, default=15000,
                        help="Steps for THREE-PHASE CRITICAL annealing (0 to disable)")
    parser.add_argument("--seed", type=int, default=11, help="Random seed (0 for random)")

    # Model parameters
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--token_dim",  type=int, default=64)
    parser.add_argument("--pos_dim",    type=int, default=256)
    parser.add_argument("--n_layers",   type=int, default=2)
    parser.add_argument("--n_heads",    type=int, default=4)

    # Training parameters  ← CHANGED: samples_per_temp → total_batch_size
    parser.add_argument("--total_batch_size", type=int, default=2000,
                        help="Total samples per step (auto-divided by num_temps). "
                             "Must be divisible by num_temps; adjusted with warning if not.")
    parser.add_argument("--num_steps",   type=int,   default=20000)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--optimizer",   type=str,   default="muon",
                        choices=["adam", "adamw", "muon"])
    parser.add_argument("--clip_grad",   type=float, default=1.0)

    # Scheduler parameters
    parser.add_argument("--scheduler",     action="store_true")
    parser.add_argument("--warmup_steps",  type=int,   default=2000)
    parser.add_argument("--eta_min",       type=float, default=1e-5)

    # Evaluation parameters
    parser.add_argument("--eval_interval", type=int, default=0)
    parser.add_argument("--eval_samples",  type=int, default=100000)

    # Output parameters
    parser.add_argument("--print_step", type=int, default=100)
    parser.add_argument("--save_step",  type=int, default=1000)
    parser.add_argument("--out_dir",    type=str,
                        default="out/L16_observables_2")
    parser.add_argument("--verbose",    action="store_true")

    # Device
    parser.add_argument("--cuda", type=int, default=5)

    return parser.parse_args()


# ============================================================
# Training step (only samples_per_temp naming changed internally)
# ============================================================

def train_step_multitemp(
    model,                          # LogProbWrapper or raw model
    hamiltonian: IsingHamiltonian,
    optimizer: torch.optim.Optimizer,
    betas: List[float],
    samples_per_temp: int,
    L: int,
    patch_size: int,
    annealing_factor: float = 1.0,
    clip_grad: float = 0.0,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Tuple[float, dict]:
    """Single training step — identical logic to original, Z2 is handled by wrapper."""

    model.train()
    device = next(model.parameters()).device
    n = hamiltonian.n
    num_temps = len(betas)
    total_batch_size = samples_per_temp * num_temps

    beta_indices = torch.arange(num_temps, device=device).repeat_interleave(samples_per_temp)

    betas_tensor = torch.tensor(betas, dtype=torch.float32, device=device)
    current_betas = betas_tensor[beta_indices] * annealing_factor

    with torch.no_grad():
        samples = model.sample(total_batch_size, beta_indices)

    log_prob = model.log_prob(samples, beta_indices)   # Z2-symmetrised if wrapped

    with torch.no_grad():
        if patch_size > 1:
            spins  = unpatch(samples, L, patch_size=patch_size) * 2.0 - 1.0
            energy = hamiltonian.energy(spins)
        else:
            spins  = samples * 2.0 - 1.0
            energy = hamiltonian.energy(spins)

        per_sample_loss = current_betas * energy + log_prob

        reshaped_loss = per_sample_loss.view(num_temps, samples_per_temp)
        baseline      = reshaped_loss.mean(dim=1, keepdim=True)
        advantages    = per_sample_loss - baseline.repeat_interleave(samples_per_temp)

    loss_per_temp  = (log_prob * advantages).view(num_temps, samples_per_temp).mean(dim=1)
    loss_reinforce = loss_per_temp.sum()

    optimizer.zero_grad()
    loss_reinforce.backward()
    if clip_grad > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    with torch.no_grad():
        stats = {}
        for i, beta in enumerate(betas):
            temp_mask        = beta_indices == i
            temp_energy      = energy[temp_mask].mean().item() / n
            temp_loss        = per_sample_loss[temp_mask].mean().item()
            temp_free_energy = (temp_loss / (beta * annealing_factor) / n
                                if annealing_factor > 1e-6 else float('nan'))
            stats[f"E_{beta:.2f}"]            = temp_energy
            stats[f"F_{beta:.2f}"]            = temp_free_energy
            stats[f"current_beta_{beta:.2f}"] = beta * annealing_factor

    return loss_reinforce.item(), stats


# ============================================================
# evaluate_vfe  — online accumulation (O(1) memory)
# ============================================================

@torch.no_grad()
def evaluate_vfe(
    model,
    hamiltonian: IsingHamiltonian,
    L: int,
    patch_size: int,
    beta: float,
    beta_idx: int,
    batch_size: int,
    n_samples: int = 100000,
) -> Tuple[float, float]:
    """
    Variational free energy via online accumulation.

    Uses Welford-style running sums instead of storing all samples,
    so memory cost is O(1) regardless of n_samples.

    Returns:
        (mean_F_per_spin, std_F_per_spin)   — std is population std-dev
    """
    model.eval()
    device = next(model.parameters()).device
    n      = hamiltonian.n

    assert n_samples % batch_size == 0, \
        f"n_samples ({n_samples}) must be divisible by batch_size ({batch_size})"

    # Online accumulators (float64 for numerical stability)
    local_sum    = torch.tensor(0.0, dtype=torch.float64, device=device)
    local_sum_sq = torch.tensor(0.0, dtype=torch.float64, device=device)
    local_count  = 0

    for _ in range(n_samples // batch_size):
        beta_idx_tensor = torch.full((batch_size,), beta_idx,
                                     dtype=torch.long, device=device)

        samples  = model.sample(batch_size, beta_idx_tensor)
        log_prob = model.log_prob(samples, beta_idx_tensor)

        if patch_size > 1:
            spins  = unpatch(samples, L, patch_size=patch_size) * 2.0 - 1.0
            energy = hamiltonian.energy(spins)
        else:
            spins  = samples * 2.0 - 1.0
            energy = hamiltonian.energy(spins)

        # Free energy per spin: F/n = E/n + (1/β) * log q
        free_energy = (energy + log_prob / beta) / n   # shape [batch_size]
        fe64 = free_energy.to(torch.float64)

        local_sum    += fe64.sum()
        local_sum_sq += (fe64 ** 2).sum()
        local_count  += batch_size

    mean_F   = (local_sum    / local_count).item()
    mean_sq  = (local_sum_sq / local_count).item()
    variance = max(mean_sq - mean_F ** 2, 0.0)
    std_F    = variance ** 0.5

    return mean_F, std_F


# ============================================================
# evaluate_observables  — E/N, |M|, Cv
# ============================================================

@torch.no_grad()
def evaluate_observables(
    model,
    hamiltonian: IsingHamiltonian,
    L: int,
    patch_size: int,
    beta: float,
    beta_idx: int,
    batch_size: int,
    n_samples: int,
) -> dict:
    """
    Estimate thermodynamic observables via importance-weight-free MC
    (direct sampling from q):

        <E>/N       — mean energy per spin
        <|M|>       — mean absolute magnetisation per spin
        Cv/N        — specific heat per spin = β² * (Var[E]) / N
                      (note: this is Cv from energy fluctuations of q,
                       not the true equilibrium Cv unless q = p_Boltzmann)

    Returns:
        dict with keys: 'E_per_spin', 'M_per_spin', 'Cv_per_spin',
                        'std_E', 'std_M'
    """
    model.eval()
    device = next(model.parameters()).device
    n = hamiltonian.n

    assert n_samples % batch_size == 0

    # Online accumulators
    sum_E    = torch.tensor(0.0, dtype=torch.float64, device=device)
    sum_E2   = torch.tensor(0.0, dtype=torch.float64, device=device)
    sum_M    = torch.tensor(0.0, dtype=torch.float64, device=device)
    sum_M2   = torch.tensor(0.0, dtype=torch.float64, device=device)
    count    = 0

    for _ in range(n_samples // batch_size):
        beta_idx_tensor = torch.full((batch_size,), beta_idx,
                                     dtype=torch.long, device=device)
        samples = model.sample(batch_size, beta_idx_tensor)

        if patch_size > 1:
            spins = unpatch(samples, L, patch_size=patch_size) * 2.0 - 1.0
        else:
            spins = samples * 2.0 - 1.0         # shape [B, n]

        energy = hamiltonian.energy(spins)       # shape [B]
        # magnetisation per spin: mean of all spins in each sample
        mag = spins.mean(dim=-1).abs()           # shape [B], already per spin

        e64  = energy.to(torch.float64)
        m64  = mag.to(torch.float64)

        sum_E  += e64.sum()
        sum_E2 += (e64 ** 2).sum()
        sum_M  += m64.sum()
        sum_M2 += (m64 ** 2).sum()
        count  += batch_size

    mean_E   = (sum_E  / count).item()
    mean_E2  = (sum_E2 / count).item()
    mean_M   = (sum_M  / count).item()
    mean_M2  = (sum_M2 / count).item()

    var_E    = max(mean_E2 - mean_E ** 2, 0.0)
    std_E    = max(mean_E2 / count - (mean_E / count) ** 2, 0.0) ** 0.5   # std of mean
    std_M    = max(mean_M2 / count - (mean_M / count) ** 2, 0.0) ** 0.5

    # Cv per spin = β² * Var[E] / N
    Cv_per_spin = beta ** 2 * var_E / n

    return {
        'E_per_spin':  mean_E / n,
        'M_per_spin':  mean_M,          # already per spin (mean of |s_i|)
        'Cv_per_spin': Cv_per_spin,
        'std_E':       std_E / n,       # std of mean E/N
        'std_M':       std_M,           # std of mean |M|
    }


# ============================================================
# evaluate_all_temps  — VFE + observables
# ============================================================

@torch.no_grad()
def evaluate_all_temps(
    model,
    hamiltonian: IsingHamiltonian,
    L: int,
    patch_size: int,
    betas: List[float],
    eval_samples: int,
    eval_batch_size_hint: int,
    logger: Logger,
    out_filename: str,
) -> List[Tuple[float, float, float]]:
    """
    Evaluate VFE and observables at all temperatures.

    Returns:
        List of (beta, mean_F, std_F) tuples.
    """
    logger.log("\n" + "=" * 70)
    logger.log("Evaluating all temperatures (VFE + observables)...")
    logger.log("=" * 70)

    results       = []
    obs_results   = []

    for idx, beta in enumerate(betas):
        # Adaptive batch size
        batch_size = get_eval_params(L, eval_samples, eval_batch_size_hint)

        # --- VFE ---
        mean_F, std_F = evaluate_vfe(
            model=model,
            hamiltonian=hamiltonian,
            L=L,
            patch_size=patch_size,
            beta=beta,
            beta_idx=idx,
            batch_size=batch_size,
            n_samples=eval_samples,
        )

        # --- Observables ---
        obs = evaluate_observables(
            model=model,
            hamiltonian=hamiltonian,
            L=L,
            patch_size=patch_size,
            beta=beta,
            beta_idx=idx,
            batch_size=batch_size,
            n_samples=eval_samples,
        )

        logger.log(
            f"beta={beta:.2f}  "
            f"F/n={mean_F:.6f}±{std_F:.6f}  "
            f"E/n={obs['E_per_spin']:.6f}±{obs['std_E']:.6f}  "
            f"|M|={obs['M_per_spin']:.6f}±{obs['std_M']:.6f}  "
            f"Cv/n={obs['Cv_per_spin']:.6f}"
        )
        results.append((beta, mean_F, std_F))
        obs_results.append((beta, obs))

    # Save VFE results
    vfe_filename = f"{out_filename}_vfe_multitemp.txt"
    with open(vfe_filename, "w") as f:
        f.write("# beta free_energy_per_spin free_energy_std\n")
        for beta, f_mean, f_std in results:
            f.write(f"{beta:.6f} {f_mean:.8f} {f_std:.8f}\n")

    # Save observables
    obs_filename = f"{out_filename}_observables.txt"
    with open(obs_filename, "w") as f:
        f.write("# beta E_per_spin std_E M_per_spin std_M Cv_per_spin\n")
        for beta, obs in obs_results:
            f.write(
                f"{beta:.6f} "
                f"{obs['E_per_spin']:.8f} {obs['std_E']:.8f} "
                f"{obs['M_per_spin']:.8f} {obs['std_M']:.8f} "
                f"{obs['Cv_per_spin']:.8f}\n"
            )

    logger.log(f"VFE saved  → {vfe_filename}")
    logger.log(f"Obs saved  → {obs_filename}")
    logger.log("=" * 70 + "\n")
    return results


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    seed   = setup_seed(args.seed)
    device = get_device(args.cuda)
    n      = args.L * args.L

    # Beta values — priority: --betas_file > --betas > DEFAULT_BETAS
    if args.betas_file is not None:
        with open(args.betas_file, "r") as f:
            betas = [
                float(line.strip())
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
        betas = sorted(betas)
        print(f"Loaded {len(betas)} beta values from {args.betas_file}")
    elif args.betas is not None:
        betas = sorted(args.betas)
    else:
        betas = DEFAULT_BETAS
    num_temps = len(betas)

    # ── total_batch_size → samples_per_temp ──────────────────────────
    samples_per_temp = args.total_batch_size // num_temps
    if args.total_batch_size % num_temps != 0:
        adjusted_total = samples_per_temp * num_temps
        print(
            f"[Warning] total_batch_size={args.total_batch_size} is not divisible "
            f"by num_temps={num_temps}. "
            f"Adjusted to {adjusted_total} (samples_per_temp={samples_per_temp})."
        )
    # ─────────────────────────────────────────────────────────────────

    out_filename = f"{args.out_dir}/L{args.L}_{args.boundary}_multitemp_seed{seed}"
    ensure_dir(out_filename + ".log")

    logger             = Logger(out_filename)
    checkpoint_manager = CheckpointManager(out_filename, args.save_step)

    if args.verbose:
        logger.log("Ising Model Multi-Temperature Free Energy Training (with Observables)")
        logger.log("=" * 70)
    logger.log(f"L={args.L}, N={n}, boundary={args.boundary}, seed={seed}")
    logger.log(f"Training {num_temps} temperatures: {betas}")
    logger.log(f"total_batch_size={args.total_batch_size}  →  "
               f"samples_per_temp={samples_per_temp}")
    logger.log(f"Annealing strategy: THREE-PHASE CRITICAL over {args.annealing_steps} steps")
    logger.log(f"  Phase 1 (0-20%):  factor 0.00 -> 0.35 [fast warmup]")
    logger.log(f"  Phase 2 (20-70%): factor 0.35 -> 0.55 [critical wandering β_c≈0.44]")
    logger.log(f"  Phase 3 (70-100%):factor 0.55 -> 1.00 [low-T hardening]")
    logger.log(f"Z2 symmetry: ENABLED (LogProbWrapper)")
    if args.verbose:
        logger.log(f"Device: {device}")

    # Hamiltonian
    hamiltonian = IsingHamiltonian(L=args.L, device=device, boundary=args.boundary)
    logger.log(f"Hamiltonian: {hamiltonian}")
    logger.log("")

    # Base model
    base_model = IsingFlashARModelPatchMultiTemp(
        L=args.L,
        beta_bins=num_temps,
        M=2,
        patch_size=args.patch_size,
        token_dim=args.token_dim,
        pos_dim=args.pos_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        device=str(device),
    ).to(device)

    # Wrap with Z2 symmetry
    model = LogProbWrapper(base_model)

    logger.log(f"Model: patched AR, patch_size={args.patch_size}, Z2-symmetric")
    n_params = count_parameters(base_model)
    logger.log(f"Parameters: {n_params:,}")
    logger.log("")

    # Optimizer (operates on base_model parameters via wrapper)
    optimizer = get_optimizer(args.optimizer, model, lr=args.lr)
    logger.log(f"Optimizer: {args.optimizer}, lr={args.lr}")

    # Scheduler
    scheduler = None
    if args.scheduler:
        warmup       = LinearLR(optimizer, start_factor=1e-2, total_iters=args.warmup_steps)
        cosine_steps = args.num_steps - args.warmup_steps
        cosine       = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=args.eta_min)
        scheduler    = SequentialLR(optimizer,
                                    schedulers=[warmup, cosine],
                                    milestones=[args.warmup_steps])
        logger.log(f"Scheduler: warmup({args.warmup_steps}) + cosine(eta_min={args.eta_min})")
    logger.log("")

    # ── Training loop (unchanged logic) ──────────────────────────────
    logger.log("Starting training...")
    logger.log("-" * 70)
    start_time = time.time()

    for step in range(1, args.num_steps + 1):

        annealing_factor = (
            get_focused_critical_annealing_factor(step, args.annealing_steps)
            if args.annealing_steps > 0 else 1.0
        )

        loss, stats = train_step_multitemp(
            model=model,
            hamiltonian=hamiltonian,
            optimizer=optimizer,
            betas=betas,
            samples_per_temp=samples_per_temp,
            L=args.L,
            patch_size=args.patch_size,
            annealing_factor=annealing_factor,
            clip_grad=args.clip_grad,
            scheduler=scheduler,
        )

        if step % args.print_step == 0:
            elapsed    = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']

            if args.annealing_steps > 0:
                progress = step / args.annealing_steps
                phase_str = ("P1:warmup"   if progress < 0.2 else
                             "P2:critical" if progress < 0.7 else
                             "P3:harden"   if progress < 1.0 else "done")
            else:
                phase_str = "off"

            logger.log(
                f"Step {step}/{args.num_steps}: "
                f"annealing={annealing_factor:.4f} [{phase_str}], "
                f"lr={current_lr:.2e}, time={elapsed:.1f}s"
            )
            if args.verbose:
                for beta in betas:
                    E_key   = f"E_{beta:.2f}"
                    F_key   = f"F_{beta:.2f}"
                    cb_key  = f"current_beta_{beta:.2f}"
                    if E_key in stats and F_key in stats:
                        logger.log(
                            f"  β={beta:.2f} (current={stats[cb_key]:.4f}): "
                            f"E/n={stats[E_key]:.6f}, F/n={stats[F_key]:.6f}"
                        )

        # Mid-training evaluation
        if args.eval_interval > 0 and step % args.eval_interval == 0:
            logger.log(f"\nEvaluating at step {step}...")
            eval_batch_hint = samples_per_temp * num_temps
            for idx, beta in enumerate(betas):
                batch_size = get_eval_params(args.L, args.eval_samples, eval_batch_hint)
                mean_F, std_F = evaluate_vfe(
                    model=model,
                    hamiltonian=hamiltonian,
                    L=args.L,
                    patch_size=args.patch_size,
                    beta=beta,
                    beta_idx=idx,
                    batch_size=batch_size,
                    n_samples=args.eval_samples,
                )
                logger.log(f"[Eval@{step}] β={beta:.2f}, F/n={mean_F:.6f}±{std_F:.6f}")

        # Checkpoint
        if args.save_step > 0 and step % args.save_step == 0:
            checkpoint_manager.save(step, model, optimizer, extra={'betas': betas})
            if args.verbose:
                logger.log(f"Checkpoint saved at step {step}\n")

    # ── Final evaluation ──────────────────────────────────────────────
    logger.log("")
    logger.log("Training completed")
    logger.log("")

    eval_batch_hint = samples_per_temp * num_temps
    results = evaluate_all_temps(
        model=model,
        hamiltonian=hamiltonian,
        L=args.L,
        patch_size=args.patch_size,
        betas=betas,
        eval_samples=args.eval_samples,
        eval_batch_size_hint=eval_batch_hint,
        logger=logger,
        out_filename=out_filename,
    )

    # ── Save final model (with model_config + boundary) ───────────────
    final_path = f"{out_filename}_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'betas': betas,
        'step': args.num_steps,
        'annealing_steps': args.annealing_steps,
        # ↓ NEW: model_config for downstream scripts
        'model_config': {
            'L':          args.L,
            'patch_size': args.patch_size,
            'token_dim':  args.token_dim,
            'pos_dim':    args.pos_dim,
            'n_layers':   args.n_layers,
            'n_heads':    args.n_heads,
        },
        'boundary': args.boundary,   # ↓ NEW
    }, final_path)
    logger.log(f"Final model saved to {final_path}")

    # ── Summary file ──────────────────────────────────────────────────
    summary_path = f"{out_filename}_summary.txt"
    with open(summary_path, "w") as f:
        f.write("2D Ising Model Multi-Temperature Free Energy Training Summary\n")
        f.write("=" * 70 + "\n")
        f.write(f"L = {args.L}\n")
        f.write(f"N = {n}\n")
        f.write(f"Boundary = {args.boundary}\n")
        f.write(f"Number of temperatures = {num_temps}\n")
        f.write(f"Temperatures (beta) = {betas}\n")
        f.write(f"total_batch_size = {args.total_batch_size}  "
                f"(samples_per_temp = {samples_per_temp})\n")
        f.write(f"Annealing strategy = THREE-PHASE CRITICAL\n")
        f.write(f"  Phase 1 (0-20%):   factor 0.00 -> 0.35 [fast warmup]\n")
        f.write(f"  Phase 2 (20-70%):  factor 0.35 -> 0.55 [critical wandering]\n")
        f.write(f"  Phase 3 (70-100%): factor 0.55 -> 1.00 [low-T hardening]\n")
        f.write(f"Annealing steps = {args.annealing_steps}\n")
        f.write(f"Warmup steps = {args.warmup_steps}\n")
        f.write(f"Seed = {seed}\n")
        f.write(f"Training steps = {args.num_steps}\n")
        f.write(f"Model parameters = {n_params}\n")
        f.write(f"Z2 symmetry = ENABLED\n")
        f.write("\n")
        f.write("Final Free Energy Results:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Beta':<10} {'F/n':<15} {'std(F/n)':<15}\n")
        f.write("-" * 70 + "\n")
        for beta, f_mean, f_std in results:
            f.write(f"{beta:<10.4f} {f_mean:<15.8f} {f_std:<15.8f}\n")

    logger.log(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()