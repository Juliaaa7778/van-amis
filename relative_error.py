import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":   "stix",
    "font.size":          9,
    "axes.labelsize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          False,
    "ytick.right":        False,
    "xtick.major.size":   3.2,
    "xtick.minor.size":   2.0,
    "ytick.major.size":   3.2,
    "ytick.minor.size":   2.0,
    "xtick.major.width":  0.7,
    "ytick.major.width":  0.7,
    "axes.linewidth":     0.7,
    "axes.edgecolor":     "black",
    "lines.linewidth":    1.0,
    "lines.markersize":   3.2,
})

def bootstrap_ci(data, n_bootstrap=5000, ci=68.27, seed=42):
    rng = np.random.default_rng(seed)
    n_runs, n_temps = data.shape
    boot_means = np.empty((n_bootstrap, n_temps))
    for i in range(n_bootstrap):
        idx = rng.integers(0, n_runs, size=n_runs)
        boot_means[i] = data[idx].mean(axis=0)
    alpha = (100 - ci) / 2
    lo   = np.percentile(boot_means, alpha,       axis=0)
    hi   = np.percentile(boot_means, 100 - alpha, axis=0)
    mean = data.mean(axis=0)
    return mean, mean - lo, hi - mean

def calc_relative_error(x, y):
    return np.abs(x - y) / np.abs(x)

exact_file = "kacward_sampling_master/output/obc_L64_free_energy.txt"
exact_data = np.loadtxt(exact_file)
beta_list  = exact_data[:, 1]
f_exact    = exact_data[:, 3]

file_paths = [
    "out/L64_obc_1/L64_OBC_multitemp_seed11_vfe_multitemp.txt",
    "out/L64_obc_2/L64_OBC_multitemp_seed7_vfe_multitemp.txt",
    "out/L64_obc_3/L64_OBC_multitemp_seed25_vfe_multitemp.txt",
    "out/L64_obc_4/L64_OBC_multitemp_seed24_vfe_multitemp.txt",
]

all_rel_errors = []
for fp in file_paths:
    df  = pd.read_csv(fp, sep=r"\s+", header=None, comment='#')
    vfe = df.iloc[:, 1].values
    all_rel_errors.append(calc_relative_error(np.array(f_exact), np.array(vfe)))

all_rel_errors = np.array(all_rel_errors)

# ==========================================
# Bootstrap
# ==========================================
mean_err, lo_err, hi_err = bootstrap_ci(all_rel_errors)
lower_bound = np.clip(mean_err - lo_err, 1e-10, None)
upper_bound = mean_err + hi_err

fig, ax = plt.subplots(figsize=(3.2, 2.2))

for spine in ax.spines.values():
    spine.set_linewidth(0.7)
    spine.set_color("black")

c_main  = "#2e7d32"   
c_shade = "#66bb6a"


ax.fill_between(
    beta_list, lower_bound, upper_bound,
    color=c_shade, alpha=0.25, linewidth=0, zorder=2,
)
ax.plot(
    beta_list, mean_err,
    "-o", color=c_main,
    markerfacecolor=c_main, markeredgecolor=c_main, markeredgewidth=0.5,
    linewidth=1.0, markersize=2.2, zorder=3,
    label="VAN",
)


ax.set_xlabel(r"$\beta$", labelpad=2)
ax.set_ylabel(r"$\vert f - f_{\mathrm{exact}} \vert / \vert f_{\mathrm{exact}} \vert$", labelpad=3)
ax.set_yscale("log")
ax.set_xlim(beta_list.min() - 0.01, beta_list.max() + 0.01)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))

ax.grid(False)
ax.legend(frameon=False, loc="upper right", handlelength=1.5, borderaxespad=0.4)

plt.tight_layout(pad=0.2)

fig.savefig("relative_error.pdf", format="pdf", bbox_inches="tight")
fig.savefig("relative_error.png", format="png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved: relative_error.pdf  relative_error.png")