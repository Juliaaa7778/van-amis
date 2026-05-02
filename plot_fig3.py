import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from matplotlib.ticker import LogLocator, AutoMinorLocator, MultipleLocator
from scipy.special import logsumexp
from build_sample_pool import SamplePoolBuilder

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":   "stix",
    "font.size":          9,
    "axes.labelsize":     10,
    "axes.titlesize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          False,
    "ytick.right":        False,
    "xtick.major.size":   4,
    "xtick.minor.size":   2.5,
    "ytick.major.size":   4,
    "ytick.minor.size":   2.5,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "axes.linewidth":     0.8,
    "axes.edgecolor":     "black",
    "lines.linewidth":    1.2,
    "lines.markersize":   4.5,
})

# ============================================================
#  Core Computation for Heatmap
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
    return alpha, N_acc_k


# ============================================================
#  Main Logic
# ============================================================
def main():
    p = argparse.ArgumentParser(description="Combined Plot for NeurIPS")
    
    # --- ESS Plot Arguments ---
    default_target_file = "out/L64_obc_1/normalized_ess_results.txt"
    default_train_file  = "out/L64_obc_1/normalized_ess_results_training.txt"
    p.add_argument("--target_file", type=str, default=default_target_file)
    p.add_argument("--train_file",  type=str, default=default_train_file)
    
    # --- Heatmap Arguments ---
    p.add_argument("--pool_path",        type=str, default=None)
    p.add_argument("--model_path",       type=str, default=None)
    p.add_argument("--device",           type=str, default="cuda:5")
    p.add_argument("--samples_per_beta", type=int, default=2000)
    p.add_argument("--batch_size",       type=int, default=1000)
    p.add_argument("--boundary",         type=str, default="OBC", choices=["OBC", "PBC"])
    p.add_argument("--z2",               action="store_true")
    p.add_argument("--n_mcmc_steps",     type=int, default=None)
    p.add_argument("--n_targets",        type=int, default=100)
    p.add_argument("--seed",             type=int, default=7)
    p.add_argument("--cmap",             type=str, default="magma")
    p.add_argument("--vmin_floor",       type=float, default=1e-5)
    
    # --- Common & Output ---
    p.add_argument("--out_dir",          type=str, default=None)
    
    # --- Architecture Arguments ---
    g = p.add_argument_group("Architecture")
    g.add_argument("--L",          type=int, default=64)
    g.add_argument("--patch_size", type=int, default=2)
    g.add_argument("--token_dim",  type=int, default=64)
    g.add_argument("--pos_dim",    type=int, default=256)
    g.add_argument("--n_layers",   type=int, default=2)
    g.add_argument("--n_heads",    type=int, default=4)

    args = p.parse_args()
    np.random.seed(args.seed)

    # ==========================================
    # 1. Load Data for ESS Plot (ax1)
    # ==========================================
    if not os.path.exists(args.target_file) or not os.path.exists(args.train_file):
        raise FileNotFoundError("ESS data files not found. Check --target_file and --train_file.")

    print(f"Loading ESS data from:\n  - {args.target_file}\n  - {args.train_file}")
    target_data = np.loadtxt(args.target_file, skiprows=2)
    ess_target_betas    = target_data[1:, 0]
    ess_frac_target     = target_data[1:, 2]  

    train_data = np.loadtxt(args.train_file, skiprows=2)
    ess_training_betas  = train_data[1:, 0]
    ess_frac_train      = train_data[1:, 2]

    # ==========================================
    # 2. Compute Data for Heatmap (ax2)
    # ==========================================
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
        p.error("Must specify either --pool_path or --model_path for the heatmap data.")

    os.makedirs(out_dir, exist_ok=True)

    K = pool['log_probs_matrix'].shape[1]
    samples_per_beta_val = len(pool['energies']) // K
    n_mcmc_steps = args.n_mcmc_steps if (args.n_mcmc_steps and args.n_mcmc_steps > 0) else K * samples_per_beta_val

    hm_training_betas = np.asarray(pool['betas'], dtype=np.float64)
    hm_b_min, hm_b_max = hm_training_betas.min(), hm_training_betas.max()
    hm_target_betas = np.linspace(hm_b_min, hm_b_max, args.n_targets)

    print("Computing α_k(β*) for each target β*...")
    alpha_matrix = np.zeros((K, len(hm_target_betas)), dtype=np.float64)
    for t, b_star in enumerate(hm_target_betas):
        alpha, _ = compute_mcmc_alpha(pool, b_star, n_mcmc_steps)
        alpha_matrix[:, t] = alpha

    # ==========================================
    # 3. Plotting the Combined 1x2 Figure
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.4))

    # --------- Panel (a): ESS Plot (ax1) ---------
    line_color = '#1f77b4'
    ax1.plot(ess_target_betas, ess_frac_target, color=line_color, linewidth=1.5, alpha=0.9, label='MH-AMIS', zorder=3)
    ax1.scatter(ess_training_betas, ess_frac_train, edgecolor='none', facecolor='#d62728', s=9, linewidth=1.0, zorder=4, label='VAN')
    
    ax1.set_xlabel(r'$\beta$', labelpad=2)
    ax1.set_ylabel(r'$\mathrm{ESS}/N$', labelpad=2)

    ax1.set_title("(a)", loc="left", fontsize=10, fontweight="bold")
    
    ess_b_min_val = min(ess_target_betas.min(), ess_training_betas.min())
    ess_b_max_val = max(ess_target_betas.max(), ess_training_betas.max())
    y_min = max(0.0, min(ess_frac_target.min(), ess_frac_train.min()) - 0.05)
    y_max = min(1.05, max(ess_frac_target.max(), ess_frac_train.max()) + 0.1)
    
    ax1.set_xlim(ess_b_min_val - 0.01, ess_b_max_val + 0.01)
    ax1.set_ylim(y_min, y_max)
    
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune='lower'))
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax1.legend(frameon=False, loc='upper left', fontsize=8, borderpad=0, labelspacing=0.2, handlelength=1.0, handletextpad=0.4)

    # --------- Panel (b): Heatmap (ax2) ---------
    positive = alpha_matrix[alpha_matrix > 0]
    vmin = max(args.vmin_floor, (positive.min() * 0.5) if positive.size else args.vmin_floor)
    vmax = min(1.0, alpha_matrix.max() * 1.05)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    plot_matrix = np.clip(alpha_matrix, a_min=vmin, a_max=None)

    def edges_from_centers(c):
        c   = np.asarray(c, dtype=np.float64)
        mid = 0.5 * (c[:-1] + c[1:])
        lo  = c[0]  - (mid[0]  - c[0])
        hi  = c[-1] + (c[-1] - mid[-1])
        return np.concatenate([[lo], mid, [hi]])

    x_edges = edges_from_centers(hm_target_betas)
    y_edges = edges_from_centers(hm_training_betas)

    pcm = ax2.pcolormesh(
        x_edges, y_edges, plot_matrix,
        cmap=args.cmap, norm=norm,
        shading="flat", rasterized=True,
    )

    ax2.set_xlabel(r"$\beta^{\star}$", labelpad=2)
    ax2.set_ylabel(r"$\beta_k$", labelpad=2)
    ax2.set_title("(b)", loc="left", fontsize=10, fontweight="bold")
    
    global_b_min = min(hm_target_betas.min(), hm_training_betas.min())
    global_b_max = max(hm_target_betas.max(), hm_training_betas.max())
    
    ax2.set_xlim(global_b_min, global_b_max)
    ax2.set_ylim(global_b_min, global_b_max)
    
    ax2.xaxis.set_major_locator(MultipleLocator(0.2))
    ax2.yaxis.set_major_locator(MultipleLocator(0.2))
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Heatmap colorbar 
    cbar = fig.colorbar(pcm, ax=ax2, pad=0.03, fraction=0.055)
    cbar.set_label(r"$\alpha_k(\beta^{\star})$", labelpad=4)
    cbar.ax.tick_params(which="both", direction="in", labelsize=8)
    cbar.locator = LogLocator(base=10.0, numticks=6)
    cbar.update_ticks()

    # Layout adjustment
    plt.tight_layout(pad=0.4, w_pad=1.4)

    # ==========================================
    # 4. Save Outputs
    # ==========================================
    pdf_path = os.path.join(out_dir, "fig3.pdf")
    png_path = os.path.join(out_dir, "fig3.png")
    fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Combined figure saved to:\n  - {pdf_path}\n  - {png_path}")

    npz_path = os.path.join(out_dir, "alpha_matrix.npz")
    np.savez_compressed(npz_path, alpha_matrix=alpha_matrix, target_betas=hm_target_betas, training_betas=hm_training_betas, n_mcmc_steps=n_mcmc_steps)
    print(f"Data saved to {npz_path}")


if __name__ == "__main__":
    main()