"""
MCMC resampling analysis and ESS estimation from a sample pool.
Outputs only normalized_ess_results.txt and normalized_ess_results_training.txt.
"""

import os
import argparse
import numpy as np
from scipy.special import logsumexp

from build_sample_pool import SamplePoolBuilder

# ============================================================
#  Core Computation Functions
# ============================================================

def compute_mcmc_alpha(pool, target_beta, n_mcmc_steps):
    energies         = pool['energies']
    log_probs_matrix = pool['log_probs_matrix']
    K                = log_probs_matrix.shape[1]
    N_pool           = len(energies)
    samples_per_beta = N_pool // K

    log_q_mix = logsumexp(log_probs_matrix - np.log(K), axis=1)
    log_w     = -target_beta * energies - log_q_mix

    current_idx = np.random.randint(N_pool)
    log_rand    = np.log(np.random.rand(n_mcmc_steps))
    proposals   = np.random.randint(0, N_pool, size=n_mcmc_steps)
    recorded    = np.empty(n_mcmc_steps, dtype=np.int32)

    for i in range(n_mcmc_steps):
        prop      = proposals[i]
        log_ratio = log_w[prop] - log_w[current_idx]
        if log_ratio >= 0 or log_rand[i] < log_ratio:
            current_idx = prop
        recorded[i] = current_idx

    k_labels = np.clip(recorded // samples_per_beta, 0, K - 1)
    N_acc_k  = np.bincount(k_labels, minlength=K)
    alpha    = N_acc_k / n_mcmc_steps

    return alpha, n_mcmc_steps, N_acc_k

def build_reweighted_pool(pool, N_acc_k):
    energies         = pool['energies']
    log_probs_matrix = pool['log_probs_matrix']
    K                = log_probs_matrix.shape[1]
    N_pool           = len(energies)
    samples_per_beta = N_pool // K

    new_e, new_lp = [], []
    for k in range(K):
        n_draw = int(N_acc_k[k])
        if n_draw == 0:
            continue
        start   = k * samples_per_beta
        replace = n_draw > samples_per_beta
        chosen  = np.random.choice(samples_per_beta, size=n_draw, replace=replace)
        idx     = start + chosen
        new_e.append(energies[idx])
        new_lp.append(log_probs_matrix[idx])

    return {
        'energies':         np.concatenate(new_e,  axis=0),
        'log_probs_matrix': np.concatenate(new_lp, axis=0),
    }

# ============================================================
#  Metric Calculation
# ============================================================

def compute_ess_for_beta(pool, target_beta, n_mcmc_steps):
    """Computes ESS only."""
    # 1. MCMC -> alpha_k, N_acc_k
    alpha, _, N_acc_k = compute_mcmc_alpha(pool, target_beta, n_mcmc_steps)

    # 2. Build reweighted pool
    pool_rw = build_reweighted_pool(pool, N_acc_k)

    # 3. Calculate ESS
    energies_rw  = pool_rw['energies']
    log_probs_rw = pool_rw['log_probs_matrix']
    log_alpha = np.log(np.clip(alpha, 1e-300, None))
    log_q_mix = logsumexp(log_probs_rw + log_alpha[np.newaxis, :], axis=1)

    log_w = -target_beta * energies_rw - log_q_mix
    log_w_tilde = log_w - logsumexp(log_w)
    ess = np.exp(-logsumexp(2.0 * log_w_tilde))

    return ess

# ============================================================
#  Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool_path",        type=str, default=None)
    p.add_argument("--model_path",       type=str, default=None)
    p.add_argument("--device",           type=str, default="cuda:5")
    p.add_argument("--samples_per_beta", type=int, default=2000)
    p.add_argument("--batch_size",       type=int, default=1000)
    p.add_argument("--boundary",         type=str, default="OBC", choices=["OBC", "PBC"])
    p.add_argument("--z2",               action="store_true")
    p.add_argument("--n_mcmc_steps",     type=int, default=None,
                   help="MCMC steps for alpha estimation; <=0 means auto set")
    p.add_argument("--out_dir",          type=str, default=None)

    g = p.add_argument_group("Architecture (only needed for old checkpoints)")
    g.add_argument("--L",          type=int, default=64)
    g.add_argument("--patch_size", type=int, default=2)
    g.add_argument("--token_dim",  type=int, default=64)
    g.add_argument("--pos_dim",    type=int, default=256)
    g.add_argument("--n_layers",   type=int, default=2)
    g.add_argument("--n_heads",    type=int, default=4)

    args = p.parse_args()

    # ---- Load Data ----
    if args.pool_path is not None:
        pool    = SamplePoolBuilder.load_sample_pool(args.pool_path)
        out_dir = args.out_dir or os.path.dirname(args.pool_path) or "."
    elif args.model_path is not None:
        builder = SamplePoolBuilder(
            model_path=args.model_path, device=args.device,
            boundary=args.boundary, z2=args.z2,
            L=args.L, patch_size=args.patch_size,
            token_dim=args.token_dim, pos_dim=args.pos_dim,
            n_layers=args.n_layers, n_heads=args.n_heads,
        )
        pool = builder.build_sample_pool(
            samples_per_beta=args.samples_per_beta,
            batch_size=args.batch_size,
        )
        pool_path = os.path.join(os.path.dirname(args.model_path) or ".", "sample_pool.npz")
        builder.save_sample_pool(pool, pool_path)
        out_dir = args.out_dir or os.path.dirname(args.model_path) or "."
    else:
        p.error("Must specify either --pool_path or --model_path")

    os.makedirs(out_dir, exist_ok=True)

    # ---- Setup steps and Betas ----
    K                    = pool['log_probs_matrix'].shape[1]
    samples_per_beta_val = len(pool['energies']) // K
    if args.n_mcmc_steps is None or args.n_mcmc_steps <= 0:
        n_mcmc_steps = K * samples_per_beta_val
        print(f"INFO: n_mcmc_steps = K * samples_per_beta = {K} * {samples_per_beta_val} = {n_mcmc_steps}")
    else:
        n_mcmc_steps = args.n_mcmc_steps

    training_betas = pool['betas']
    total_samples  = len(pool['energies'])

    b_min, b_max = training_betas.min(), training_betas.max()
    target_betas = np.linspace(b_min, b_max, 200)

    print(f"Training betas : {len(training_betas)} points  [{b_min:.4f}, {b_max:.4f}]")
    print(f"Target betas   : {len(target_betas)} points")
    print(f"Total samples  : {total_samples}")
    print(f"MCMC steps     : {n_mcmc_steps}")

    # ---- Compute Metrics ----
    print("Computing ESS metrics...")

    ess_target = [compute_ess_for_beta(pool, b, n_mcmc_steps) for b in target_betas]
    ess_train  = [compute_ess_for_beta(pool, b, n_mcmc_steps) for b in training_betas]

    # Calculate fractions (normalized)
    ess_frac_target = np.array(ess_target) / total_samples
    ess_frac_train  = np.array(ess_train) / total_samples

    # ---- Save Numerical Results ----
    txt_path = os.path.join(out_dir, "normalized_ess_results.txt")
    with open(txt_path, "w") as f:
        f.write(f"{'beta':<12}  {'ESS':<15}  {'Norm_ESS':<15}\n")
        f.write("-" * 46 + "\n")
        for b, e, ess_f in zip(target_betas, ess_target, ess_frac_target):
            f.write(f"{b:<12.6f}  {e:<15.4f}  {ess_f:<15.4f}\n")
    print(f"Numerical results saved to {txt_path}")

    # ---- Save Training Beta Results ----
    train_txt_path = os.path.join(out_dir, "normalized_ess_results_training.txt")
    with open(train_txt_path, "w") as f:
        f.write(f"{'beta':<12}  {'ESS':<15}  {'Norm_ESS':<15}\n")
        f.write("-" * 46 + "\n")
        for b, e, ess_f in zip(training_betas, ess_train, ess_frac_train):
            f.write(f"{b:<12.6f}  {e:<15.4f}  {ess_f:<15.4f}\n")
    print(f"Training beta results saved to {train_txt_path}")

if __name__ == "__main__":
    main()