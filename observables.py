import os
import argparse
import numpy as np
import torch
from scipy.special import logsumexp

from build_sample_pool import SamplePoolBuilder


def load_observables_txt(path):
    
    try:
        data = np.loadtxt(path, comments="#")
        if data.ndim == 1:
            return data[0:1], data[1:2], data[3:4]
        return data[:, 0], data[:, 1], data[:, 3]
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None, None, None

def compute_mcmc_alpha(pool, target_beta, n_mcmc_steps=92000):
    energies         = pool['energies']
    log_probs_matrix = pool['log_probs_matrix']
    K                = log_probs_matrix.shape[1]
    N_pool           = len(energies)
    samples_per_beta = N_pool // K

    log_q_mix = logsumexp(log_probs_matrix - np.log(K), axis=1)
    log_w     = -target_beta * energies - log_q_mix

    current_idx = np.random.randint(N_pool)
    accepted_indices = []

    log_rand = np.log(np.random.rand(n_mcmc_steps))
    proposals = np.random.randint(0, N_pool, size=n_mcmc_steps)

    for i in range(n_mcmc_steps):
        prop = proposals[i]
        log_ratio = log_w[prop] - log_w[current_idx]
        if log_ratio >= 0 or log_rand[i] < log_ratio:
            current_idx = prop
        accepted_indices.append(current_idx)

    accepted_indices = np.array(accepted_indices, dtype=np.int32)
    N_total = len(accepted_indices)

    k_of_sample = accepted_indices // samples_per_beta
    k_of_sample = np.clip(k_of_sample, 0, K - 1)

    N_acc_k = np.bincount(k_of_sample, minlength=K)
    alpha    = N_acc_k / N_total

    return alpha, N_total, N_acc_k

def build_reweighted_pool(pool, alpha, N_acc_k):
    energies         = pool['energies']
    configs          = pool['configs']
    log_probs_matrix = pool['log_probs_matrix']
    K                = log_probs_matrix.shape[1]
    N_pool           = len(energies)
    samples_per_beta = N_pool // K

    new_energies, new_configs, new_log_probs = [], [], []

    for k in range(K):
        n_draw = N_acc_k[k]
        if n_draw == 0: continue
        start = k * samples_per_beta
        replace = n_draw > samples_per_beta
        chosen = np.random.choice(samples_per_beta, size=n_draw, replace=replace)
        idx = start + chosen

        new_energies.append(energies[idx])
        new_configs.append(configs[idx])
        new_log_probs.append(log_probs_matrix[idx])

    return {
        'energies':         np.concatenate(new_energies, axis=0),
        'configs':          np.concatenate(new_configs, axis=0),
        'log_probs_matrix': np.concatenate(new_log_probs, axis=0),
        'alpha':            alpha,
    }

def estimate_observables(pool, target_beta, alpha=None):
    energies         = pool['energies']
    configs          = pool['configs']
    log_probs_matrix = pool['log_probs_matrix']
    K                = log_probs_matrix.shape[1]
    N                = configs.shape[1] ** 2

    if alpha is None:
        alpha = np.ones(K) / K

    log_alpha = np.log(np.clip(alpha, 1e-300, None))
    log_q_mix = logsumexp(log_probs_matrix + log_alpha[np.newaxis, :], axis=1)

    log_w     = -target_beta * energies - log_q_mix
    w_shifted = np.exp(log_w - np.max(log_w))
    w_norm    = w_shifted / np.sum(w_shifted)

    spins = 2 * configs - 1
    m_abs = np.abs(np.mean(spins, axis=(1, 2)))

    return {
        'beta':          target_beta,
        'energy':        np.sum(w_norm * energies) / N,
        'magnetization': np.sum(w_norm * m_abs),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool_path",        type=str,   default=None)
    p.add_argument("--model_path",       type=str,   default=None)
    p.add_argument("--device",           type=str,   default="cuda:1")
    p.add_argument("--seed",             type=int,   default=7)
    p.add_argument("--samples_per_beta", type=int,   default=10000)
    p.add_argument("--batch_size",       type=int,   default=1000)
    p.add_argument("--boundary",         type=str,   default="OBC", choices=["OBC", "PBC"])
    p.add_argument("--z2",               action="store_true")
    p.add_argument("--n_mcmc_steps",     type=int,   default=None)
    p.add_argument("--beta_min",         type=float, default=None)
    p.add_argument("--beta_max",         type=float, default=None)
    p.add_argument("--out_dir",          type=str,   default=None)
    
    p.add_argument("--L",                type=int,   default=64)
    p.add_argument("--patch_size",       type=int,   default=2)
    p.add_argument("--token_dim",        type=int,   default=128)
    p.add_argument("--pos_dim",          type=int,   default=128)
    p.add_argument("--n_layers",         type=int,   default=4)
    p.add_argument("--n_heads",          type=int,   default=4)

    p.add_argument("--eval_path",        type=str,   default=None,
                   help="Path to _observables.txt (columns: beta E_per_spin std_E M_per_spin ...)")

    args = p.parse_args()
    torch.manual_seed(args.seed)

    if args.pool_path is not None:
        pool    = SamplePoolBuilder.load_sample_pool(args.pool_path)
        out_dir = args.out_dir or os.path.dirname(args.pool_path) or "."
    elif args.model_path is not None:
        builder = SamplePoolBuilder(
            model_path=args.model_path, device=args.device, boundary=args.boundary, z2=args.z2,
            L=args.L, patch_size=args.patch_size, token_dim=args.token_dim, pos_dim=args.pos_dim,
            n_layers=args.n_layers, n_heads=args.n_heads,
        )
        pool = builder.build_sample_pool(samples_per_beta=args.samples_per_beta, batch_size=args.batch_size)
        out_dir = args.out_dir or os.path.dirname(args.model_path) or "."
    else:
        raise SystemExit("Must specify either --pool_path or --model_path")

    os.makedirs(out_dir, exist_ok=True)

    if args.n_mcmc_steps is None or args.n_mcmc_steps <= 0:
        K = len(pool['betas'])
        n_mcmc_steps = K * (len(pool['energies']) // K)
    else:
        n_mcmc_steps = args.n_mcmc_steps

    b_min = args.beta_min if args.beta_min is not None else pool['betas'].min()
    b_max = args.beta_max if args.beta_max is not None else pool['betas'].max()
    
    target_betas = np.linspace(b_min, b_max, 200)

    results = []
    for tb in target_betas:
        alpha, N_total, N_acc_k = compute_mcmc_alpha(pool, tb, n_mcmc_steps=n_mcmc_steps)
        rw_pool = build_reweighted_pool(pool, alpha, N_acc_k)
        results.append(estimate_observables(rw_pool, tb, alpha=alpha))

    txt_path = os.path.join(out_dir, "estimate_observables_fixed.txt")
    with open(txt_path, "w") as f:
        f.write(f"{'beta':<12} {'E/n':<15} {'|m|':<15}\n")
        f.write("-" * 45 + "\n")
        for r in results:
            f.write(f"{r['beta']:<12.6f} {r['energy']:<15.8f} {r['magnetization']:<15.8f}\n")
    print(f"Results saved to {txt_path}")

if __name__ == "__main__":
    main()