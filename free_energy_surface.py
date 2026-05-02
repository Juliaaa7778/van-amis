"""
Compare free energy from mixture importance sampling (MIS) vs exact Kac-Ward solution.
Usage (from existing sample_pool.npz):
    python free_energy_surface.py --pool_path out/.../sample_pool.npz --L 32

Usage (build pool on the fly, with Z2):
    python free_energy_surface.py \
        --model_path out/.../L32_OBC_ddp_seed7_final.pt \
        --device cuda:0 --samples_per_beta 10000 --z2
"""
import os
import argparse
import torch
import numpy as np
import scipy
import networkx as nx
from tqdm import tqdm
from scipy.special import logsumexp

from build_sample_pool import SamplePoolBuilder


# ============================================================
#  Exact Kac-Ward helpers (unchanged)
# ============================================================

def logcosh(x):
    return np.abs(x) + np.log(1 + np.exp(-2.0 * np.abs(x))) - np.log(2.0)


def compute_angle(i, j, k):
    k2j = j - k
    j2i = i - j
    return (np.arctan2(k2j[1], k2j[0]) - np.arctan2(j2i[1], j2i[0]) + np.pi) % (2.0 * np.pi) - np.pi


def calculate_exact_lnZ(L, beta):
    G = nx.grid_2d_graph(L, L)
    pos = {(i, j): np.array([i, j]) for i, j in G.nodes()}
    m = G.number_of_edges()
    directed_edge_list = ([(i, j) for (i, j) in G.edges()]
                          + [(j, i) for (i, j) in G.edges()])
    directed_edge_dic = {edge: idx for idx, edge in enumerate(directed_edge_list)}

    B = scipy.sparse.lil_matrix((2 * m, 2 * m), dtype=np.complex128)
    for i in G:
        for j in G.neighbors(i):
            for k in G.neighbors(j):
                if k != i:
                    B[directed_edge_dic[(i, j)], directed_edge_dic[(j, k)]] = np.exp(
                        1j / 2.0 * compute_angle(pos[i], pos[j], pos[k])
                    )

    D = scipy.sparse.lil_matrix((2 * m, 2 * m), dtype=np.complex128)
    for i, j in directed_edge_list:
        D[directed_edge_dic[(i, j)], directed_edge_dic[(i, j)]] = np.tanh(1.0 * beta)

    lnZ = G.number_of_nodes() * np.log(2.0)
    for i, j in G.edges():
        lnZ += logcosh(1.0 * beta)

    A = scipy.sparse.eye(2 * m, 2 * m, dtype=np.complex128) - B @ D
    A = scipy.sparse.csc_matrix(A)
    LU = scipy.sparse.linalg.splu(A)
    logdet = np.sum(np.log(LU.U.diagonal()))
    lnZ += 0.5 * np.real(logdet)
    return lnZ


# ============================================================
#  MCMC alpha estimation  (same logic as estimate_observables)
# ============================================================

def compute_mcmc_alpha(pool, target_beta, n_mcmc_steps=460000):
    """
    Run independent-proposal MH on the full sample pool to estimate
        alpha_k(beta) = N_acc_k / N_total
    where N_acc_k = # accepted states that came from training temperature k.

    Returns
    -------
    alpha   : (K,)  normalised weights, sums to 1
    N_total : int   total recorded steps (= n_mcmc_steps)
    N_acc_k : (K,)  raw counts per training temperature
    """
    energies         = pool['energies']           # (N_pool,)
    log_probs_matrix = pool['log_probs_matrix']   # (N_pool, K)
    K                = log_probs_matrix.shape[1]
    N_pool           = len(energies)
    samples_per_beta = N_pool // K

    # Uniform q_mix used as the base for MH acceptance
    log_q_mix = logsumexp(log_probs_matrix - np.log(K), axis=1)   # (N_pool,)
    log_w     = -target_beta * energies - log_q_mix                # (N_pool,)

    # Independent-proposal MH: proposal = uniform over pool
    current_idx  = np.random.randint(N_pool)
    log_rand     = np.log(np.random.rand(n_mcmc_steps))
    proposals    = np.random.randint(0, N_pool, size=n_mcmc_steps)
    recorded     = np.empty(n_mcmc_steps, dtype=np.int32)

    for i in range(n_mcmc_steps):
        prop      = proposals[i]
        log_ratio = log_w[prop] - log_w[current_idx]
        if log_ratio >= 0 or log_rand[i] < log_ratio:
            current_idx = prop
        recorded[i] = current_idx

    # Which training temperature does each recorded index belong to?
    k_labels = np.clip(recorded // samples_per_beta, 0, K - 1)
    N_acc_k  = np.bincount(k_labels, minlength=K)   # (K,)
    N_total  = n_mcmc_steps                          # every step is recorded
    alpha    = N_acc_k / N_total                     # (K,), sums to 1

    return alpha, N_total, N_acc_k


# ============================================================
#  Build beta-specific reweighted sub-pool
# ============================================================

def build_reweighted_pool(pool, N_acc_k):
    """
    Draw N_acc_k[k] samples from the original `samples_per_beta` samples
    of training temperature k, then concatenate → new pool of size N_total.
    """
    energies         = pool['energies']           # (N_pool,)
    configs          = pool['configs']            # (N_pool, L, L)
    log_probs_matrix = pool['log_probs_matrix']   # (N_pool, K)
    K                = log_probs_matrix.shape[1]
    N_pool           = len(energies)
    samples_per_beta = N_pool // K

    new_e, new_c, new_lp = [], [], []
    for k in range(K):
        n_draw = int(N_acc_k[k])
        if n_draw == 0:
            continue
        start   = k * samples_per_beta
        end     = start + samples_per_beta
        replace = n_draw > samples_per_beta
        chosen  = np.random.choice(samples_per_beta, size=n_draw, replace=replace)
        idx     = start + chosen

        new_e.append(energies[idx])
        new_c.append(configs[idx])
        new_lp.append(log_probs_matrix[idx])

    return {
        'energies':         np.concatenate(new_e,  axis=0),
        'configs':          np.concatenate(new_c,  axis=0),
        'log_probs_matrix': np.concatenate(new_lp, axis=0),
    }


# ============================================================
#  Free energy with adaptive q_mix
# ============================================================

def calculate_free_energy_adaptive(pool, target_betas, L, n_mcmc_steps, device):
    """
    For each target beta:
      1. Run MCMC on full pool → alpha_k(beta)
      2. Build reweighted sub-pool (size = N_total)
      3. Compute
             log Z_hat = logsumexp( -beta*E - log q_mix ) - log(N_total)
         with  log q_mix = logsumexp( log_alpha + log_q_k,  over k )
      4. f(beta) = -log Z_hat / (beta * L^2)

    Returns
    -------
    f_mis : (n_targets,) free energy per site
    """
    K = pool['log_probs_matrix'].shape[1]
    f_mis = []

    print("Calculating free energy surface (adaptive q_mix)...")
    for beta_t in tqdm(target_betas):

        # ---- Step 1 : MCMC alpha ----------------------------------------
        alpha, N_total, N_acc_k = compute_mcmc_alpha(
            pool, beta_t, n_mcmc_steps=n_mcmc_steps
        )

        # ---- Step 2 : reweighted sub-pool --------------------------------
        rw_pool = build_reweighted_pool(pool, N_acc_k)

        energies_rw  = rw_pool['energies']           # (M,)
        log_probs_rw = rw_pool['log_probs_matrix']   # (M, K)
        M            = len(energies_rw)

        # ---- Step 3 : log q_mix with adaptive alpha ----------------------
        log_alpha = np.log(np.clip(alpha, 1e-300, None))             # (K,)
        log_q_mix = logsumexp(
            log_probs_rw + log_alpha[np.newaxis, :], axis=1          # (M,)
        )

        # log w_j = -beta * E_j - log q_mix_j
        log_w = -beta_t * energies_rw - log_q_mix                   # (M,)

        # log Z_hat = log(1/M * sum exp(log_w))
        #           = logsumexp(log_w) - log(M)
        log_Z_hat = logsumexp(log_w) - np.log(M)

        # ---- Step 4 : free energy per site --------------------------------
        f_beta = -(1.0 / (beta_t * L ** 2)) * log_Z_hat
        f_mis.append(f_beta)

    return np.array(f_mis)


# ============================================================
#  (kept for comparison) original uniform-alpha version
# ============================================================

def calculate_free_energy_surface(pool, target_betas, L, device):
    """Original uniform-1/K q_mix estimator (kept for reference)."""
    energies     = torch.from_numpy(pool['energies']).to(device)
    log_p_matrix = torch.from_numpy(pool['log_probs_matrix']).to(device)
    K            = log_p_matrix.shape[1]
    N_total      = energies.shape[0]

    log_q_mix = torch.logsumexp(log_p_matrix - np.log(K), dim=1)

    free_energies = []
    for beta_t in target_betas:
        log_w     = -beta_t * energies - log_q_mix
        log_z_hat = torch.logsumexp(log_w, dim=0) - np.log(N_total)
        f_beta    = -(1.0 / (beta_t * L ** 2)) * log_z_hat.item()
        free_energies.append(f_beta)

    return np.array(free_energies)


# ============================================================
#  CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Free energy surface: adaptive-MIS vs Kac-Ward exact"
    )
    p.add_argument("--pool_path",        type=str,  default=None)
    p.add_argument("--model_path",       type=str,  default=None)
    p.add_argument("--device",           type=str,  default="cuda:1")
    p.add_argument("--samples_per_beta", type=int,  default=2000)
    p.add_argument("--batch_size",       type=int,  default=1000)
    p.add_argument("--boundary",         type=str,  default="OBC",
                   choices=["OBC", "PBC"])
    p.add_argument("--z2",               action="store_true")
    p.add_argument("--seed",             type=int,  default=11)
    p.add_argument("--n_targets",        type=int,  default=100)
    p.add_argument("--n_mcmc_steps",     type=int,  default=None,
                   help="MCMC steps per beta for alpha estimation; <=0 means auto set")

    p.add_argument("--compare_uniform",  action="store_true",
                   help="Also compute the original uniform-1/K MIS curve")
    p.add_argument("--out_dir",          type=str,  default=None)

    g = p.add_argument_group("Architecture (only needed for old checkpoints)")
    g.add_argument("--L",          type=int, default=64)
    g.add_argument("--patch_size", type=int, default=2)
    g.add_argument("--token_dim",  type=int, default=64)
    
    g.add_argument("--pos_dim",    type=int, default=256)
    g.add_argument("--n_layers",   type=int, default=2)
    g.add_argument("--n_heads",    type=int, default=4)
    return p.parse_args()


def main():
    args   = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # ---- Load or build sample pool --------------------------------------
    if args.pool_path is not None:
        pool    = SamplePoolBuilder.load_sample_pool(args.pool_path)
        out_dir = args.out_dir or os.path.dirname(args.pool_path) or "."
        L       = args.L
    elif args.model_path is not None:
        builder = SamplePoolBuilder(
            model_path=args.model_path,
            device=args.device,
            boundary=args.boundary,
            z2=args.z2,
            L=args.L, patch_size=args.patch_size,
            token_dim=args.token_dim, pos_dim=args.pos_dim,
            n_layers=args.n_layers, n_heads=args.n_heads,
        )
        pool = builder.build_sample_pool(
            samples_per_beta=args.samples_per_beta,
            batch_size=args.batch_size,
        )
        pool_path = os.path.join(
            os.path.dirname(args.model_path) or ".", "sample_pool.npz"
        )
        builder.save_sample_pool(pool, pool_path)
        L       = builder.cfg['L']
        out_dir = args.out_dir or os.path.dirname(args.model_path) or "."
    else:
        raise SystemExit("Must specify either --pool_path or --model_path")

    os.makedirs(out_dir, exist_ok=True)

    if args.n_mcmc_steps is None or args.n_mcmc_steps <= 0:
        K = len(pool['betas'])
        samples_per_beta_pool = len(pool['energies']) // K
        n_mcmc_steps = K * samples_per_beta_pool
        print(f"INFO: Setting n_mcmc_steps = K * samples_per_beta = {K} * {samples_per_beta_pool} = {n_mcmc_steps}")
    else:
        n_mcmc_steps = args.n_mcmc_steps

    training_betas = pool['betas']
    beta_min, beta_max = training_betas.min(), training_betas.max()
    target_betas   = np.linspace(beta_min, beta_max, args.n_targets)

    # ---- Adaptive MIS free energy ----------------------------------------
    f_adaptive = calculate_free_energy_adaptive(
        pool, target_betas, L,
        n_mcmc_steps=n_mcmc_steps,
        device=device,
    )

    # Training beta points with adaptive method
    f_training_adaptive = calculate_free_energy_adaptive(
        pool, training_betas, L,
        n_mcmc_steps=n_mcmc_steps,
        device=device,
    )

    # ---- Uniform MIS (optional comparison) --------------------------------
    if args.compare_uniform:
        f_uniform  = calculate_free_energy_surface(pool, target_betas, L, device)

    # ---- Exact Kac-Ward ---------------------------------------------------
    f_exact = []
    print("Calculating exact solution (Kac-Ward)...")
    for b in tqdm(target_betas):
        lnZ = calculate_exact_lnZ(L, b)
        f_exact.append(-lnZ / (L ** 2) / b)
    f_exact = np.array(f_exact)

    # ---- Save numerical results ------------------------------------------
    txt_path = os.path.join(out_dir, "free_energy_results.txt")
    with open(txt_path, "w") as f:
        f.write(f"{'beta':<12} {'f_adaptive':<18} {'f_exact':<18} {'rel_err':<15}\n")
        f.write("-" * 63 + "\n")
        for b, fa, fe in zip(target_betas, f_adaptive, f_exact):
            rel = abs(fa - fe) / abs(fe) if fe != 0 else float('nan')
            f.write(f"{b:<12.6f} {fa:<18.10f} {fe:<18.10f} {rel:<15.6e}\n")
    print(f"Results saved to {txt_path}")

if __name__ == "__main__":
    main()