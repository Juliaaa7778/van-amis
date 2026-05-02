import os
import argparse
import numpy as np
import torch
from scipy.special import logsumexp

from build_sample_pool import SamplePoolBuilder

def load_betas_file(path):
    betas = []
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    betas.append(float(line.split()[0]))
                except ValueError:
                    continue
    except Exception as e:
        print(f"Error loading betas file {path}: {e}")
        return None
    if not betas:
        return None
    return np.array(sorted(betas))


def compute_mcmc_alpha(pool, target_beta, n_mcmc_steps):
    energies         = pool['energies']
    log_probs_matrix = pool['log_probs_matrix']
    K                = log_probs_matrix.shape[1]
    N_pool           = len(energies)
    samples_per_beta = N_pool // K

    log_q_mix = logsumexp(log_probs_matrix - np.log(K), axis=1)
    log_w     = -target_beta * energies - log_q_mix

    current_idx = np.random.randint(N_pool)
    accepted_indices = []

    log_rand  = np.log(np.random.rand(n_mcmc_steps))
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
    alpha   = N_acc_k / N_total
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
        if n_draw == 0:
            continue
        start   = k * samples_per_beta
        replace = n_draw > samples_per_beta
        chosen  = np.random.choice(samples_per_beta, size=n_draw, replace=replace)
        idx     = start + chosen
        new_energies.append(energies[idx])
        new_configs.append(configs[idx])
        new_log_probs.append(log_probs_matrix[idx])

    return {
        'energies':         np.concatenate(new_energies,  axis=0),
        'configs':          np.concatenate(new_configs,   axis=0),
        'log_probs_matrix': np.concatenate(new_log_probs, axis=0),
        'alpha':            alpha,
    }


def estimate_observables(pool, target_beta, alpha=None):
    energies         = pool['energies']
    configs          = pool['configs']
    log_probs_matrix = pool['log_probs_matrix']
    K = log_probs_matrix.shape[1]
    L = configs.shape[1]
    N = L * L

    if alpha is None:
        alpha = np.ones(K) / K

    log_alpha = np.log(np.clip(alpha, 1e-300, None))
    log_q_mix = logsumexp(log_probs_matrix + log_alpha[np.newaxis, :], axis=1)

    log_w     = -target_beta * energies - log_q_mix
    w_shifted = np.exp(log_w - np.max(log_w))
    w_norm    = w_shifted / np.sum(w_shifted)

    spins  = 2 * configs - 1
    m_abs  = np.abs(np.mean(spins, axis=(1, 2)))

    e_mean = np.sum(w_norm * energies)
    e_var  = np.sum(w_norm * (energies ** 2)) - e_mean ** 2
    m_mean = np.sum(w_norm * m_abs)

    return {
        'beta':          target_beta,
        'energy':        e_mean / N,
        'magnetization': m_mean,
        'specific_heat': (target_beta ** 2) * e_var / N,
    }


def run_amis_for_betas(pool, target_betas, n_mcmc_steps, label=""):
    results = []
    total = len(target_betas)
    for i, tb in enumerate(target_betas):
        if (i + 1) % max(1, total // 10) == 0 or i == 0:
            print(f"  {label} [{i+1}/{total}] beta={tb:.4f}")
        alpha, _, N_acc_k = compute_mcmc_alpha(pool, tb, n_mcmc_steps=n_mcmc_steps)
        rw_pool = build_reweighted_pool(pool, alpha, N_acc_k)
        obs = estimate_observables(rw_pool, tb, alpha=alpha)
        results.append(obs)
    betas_arr = np.array([r['beta'] for r in results])
    cv_arr    = np.array([r['specific_heat'] for r in results])
    return betas_arr, cv_arr


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Specific heat computation (AMIS) with training-point output")
    p.add_argument("--pool_path",        type=str,   default=None)
    p.add_argument("--model_path",       type=str,   default=None)
    p.add_argument("--device",           type=str,   default="cuda:5")
    p.add_argument("--seed",             type=int,   default=7)
    p.add_argument("--samples_per_beta", type=int,   default=10000)
    p.add_argument("--batch_size",       type=int,   default=1000)
    p.add_argument("--boundary",         type=str,   default="OBC", choices=["OBC", "PBC"])
    p.add_argument("--z2",               action="store_true")
    p.add_argument("--out_dir",          type=str,   default=None)

    p.add_argument("--L",                type=int,   default=64)
    p.add_argument("--patch_size",       type=int,   default=2)
    p.add_argument("--token_dim",        type=int,   default=64)
    p.add_argument("--pos_dim",          type=int,   default=256)
    p.add_argument("--n_layers",         type=int,   default=2)
    p.add_argument("--n_heads",          type=int,   default=4)

    p.add_argument("--betas_file",       type=str,   default=None,
                   help="Path to selected_betas_*.txt (one beta per line). "
                        "If provided, AMIS is also evaluated at these betas and saved "
                        "as training_points_cv.txt.")

    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

 
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
            samples_per_beta=args.samples_per_beta, batch_size=args.batch_size,
        )
        pool_path = os.path.join(os.path.dirname(args.model_path) or ".", "sample_pool.npz")
        builder.save_sample_pool(pool, pool_path)
        out_dir = args.out_dir or os.path.dirname(args.model_path) or "."
    else:
        raise SystemExit("Must specify --pool_path or --model_path")

    os.makedirs(out_dir, exist_ok=True)

    K = len(pool['betas'])
    samples_per_beta_pool = len(pool['energies']) // K
    n_mcmc_steps = K * samples_per_beta_pool
    print(f"INFO: K={K}, samples_per_beta={samples_per_beta_pool}, n_mcmc_steps={n_mcmc_steps}")

    beta_low   = np.arange(0.1,  0.44 - 1e-6, 0.01)
    beta_dense = np.arange(0.44, 0.46 + 1e-6, 0.0001)
    beta_high  = np.arange(0.47, 1.0  + 1e-6, 0.01)
    target_betas = np.concatenate([beta_low, beta_dense, beta_high])
    print(f"INFO: Dense grid: {len(target_betas)} points")

    print("Running AMIS on dense grid...")
    plot_betas, plot_cv = run_amis_for_betas(pool, target_betas, n_mcmc_steps, label="dense")

    max_idx    = np.argmax(plot_cv)
    max_beta   = plot_betas[max_idx]
    max_cv_val = plot_cv[max_idx]
    print(f"INFO: Peak Cv at beta={max_beta:.4f} (Cv={max_cv_val:.4f})")

    txt_path = os.path.join(out_dir, "specific_heat_data.txt")
    with open(txt_path, "w") as f:
        f.write(f"{'beta':<12} {'AMIS_Cv':<15}\n")
        f.write("-" * 28 + "\n")
        for b, c in zip(plot_betas, plot_cv):
            f.write(f"{b:<12.6f} {c:<15.8f}\n")
    print(f"Dense grid saved to {txt_path}")


    if args.betas_file is not None:
        train_betas_arr = load_betas_file(args.betas_file)
        if train_betas_arr is not None:
            print(f"INFO: Training betas loaded: {len(train_betas_arr)} points")
            print(f"  beta range = [{train_betas_arr.min():.4f}, {train_betas_arr.max():.4f}]")
            print("Running AMIS on training betas...")
            train_betas_arr, train_cv_arr = run_amis_for_betas(
                pool, train_betas_arr, n_mcmc_steps, label="train"
            )
            train_txt = os.path.join(out_dir, "training_points_cv.txt")
            with open(train_txt, "w") as f:
                f.write(f"{'beta':<12} {'AMIS_Cv':<15}\n")
                f.write("-" * 28 + "\n")
                for b, c in zip(train_betas_arr, train_cv_arr):
                    f.write(f"{b:<12.6f} {c:<15.8f}\n")
            print(f"Training-point Cv saved to {train_txt}")
        else:
            print("WARNING: Could not load betas_file, skipping training-point output.")


if __name__ == "__main__":
    main()