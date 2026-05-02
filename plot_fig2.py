import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":  "stix",
    "font.size":         9,
    "axes.labelsize":    10,
    "axes.titlesize":    10,
    "legend.fontsize":   8,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "lines.linewidth":   1.2,
    "lines.markersize":  3,
    "figure.dpi":        300,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "axes.linewidth":    0.8,
})

def darken_color(color, factor=0.55):
    r, g, b = to_rgb(color)
    return (r * factor, g * factor, b * factor)

# ------------------------------------------
# Panel (a)     ⟨E⟩/N  +  |m|
# ------------------------------------------
def load_eval_data(path):
    data = np.loadtxt(path, comments="#")
    return data[:, 0], data[:, 1], data[:, 3]      # beta, E/N, |m|

def load_amis_obs(path):
    data = np.loadtxt(path, skiprows=2)
    return data[:, 0], data[:, 1], data[:, 2]      # beta, E/N, |m|

work_dir_obs = "out/L64_obc_1"
eval_b, eval_E, eval_m = load_eval_data(
    os.path.join(work_dir_obs, "L64_OBC_multitemp_seed11_observables.txt"))
amis_b, amis_E, amis_m = load_amis_obs(
    os.path.join(work_dir_obs, "estimate_observables_fixed.txt"))

# ------------------------------------------
# Panel (b)     Cv/N  for  L=16, 32, 64
# ------------------------------------------
def load_kacward(path):
    betas, cvs = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            betas.append(float(parts[0]))
            cvs.append(float(parts[4]))
    return np.array(betas), np.array(cvs)

def load_two_col(path):
    data = np.loadtxt(path, skiprows=2)
    return data[:, 0], data[:, 1]

cv_configs = [
    {"L": 16, "color": "#1f77b4",
     "amis":  "out/L16_obc/specific_heat_data.txt",
     "kw":    "kacward_sampling_master/output/obc_cv_L16_0.0001.txt",
     "train": "out/L16_obc/training_points_cv.txt"},
    {"L": 32, "color": "#2ca02c",
     "amis":  "out/L32_obc/specific_heat_data.txt",
     "kw":    "kacward_sampling_master/output/obc_cv_L32_0.0001.txt",
     "train": "out/L32_obc/training_points_cv.txt"},
    {"L": 64, "color": "#d62728",
     "amis":  "out/L64_obc_1/specific_heat_data.txt",
     "kw":    "kacward_sampling_master/output/obc_cv_L64_0.0001.txt",
     "train": "out/L64_obc_1/training_points_cv.txt"},
]

# ==========================================
# Figure
# ==========================================
fig, (axA, axB) = plt.subplots(1, 2, figsize=(5.5, 2.4))

# ---------- Panel (a) : double y-axis ----------
c_eng = "#e41a1c"
c_mag = "#377eb8"

axA.set_xlabel(r"$\beta$", labelpad=2)
axA.set_ylabel(r"$\langle E\rangle/N$", color=c_eng, labelpad=3)
axA.tick_params(axis="y", labelcolor=c_eng)
axA.spines["left"].set_color(c_eng)
axA.plot(eval_b, eval_E, color=c_eng, alpha=0.6, ls="none",
         marker="o", ms=2.8, mfc="none")
axA.plot(amis_b, amis_E, color=c_eng, ls="-", lw=1.3)

axA2 = axA.twinx()
axA2.set_ylabel(r"$|m|$", color=c_mag, labelpad=3)
axA2.tick_params(axis="y", labelcolor=c_mag, direction="in")
axA2.spines["right"].set_color(c_mag)
axA2.spines["right"].set_visible(True)
axA2.plot(eval_b, eval_m, color=c_mag, alpha=0.6, ls="none",
          marker="o", ms=2.8, mfc="none")
axA2.plot(amis_b, amis_m, color=c_mag, ls="-", lw=1.3)

handles_a = [
    mlines.Line2D([], [], color="gray", ls="none", marker="o", ms=2.8,
                  mfc="none", label="VAN"),
    mlines.Line2D([], [], color="gray", ls="-",  lw=1.3, label="MH-AMIS"),
]
axA.legend(handles=handles_a, loc="center right", frameon=False,
           handlelength=1.5, borderaxespad=0.4)
axA.text(0.12, 0.96, '(a)', transform=axA.transAxes,
         fontsize=10, va='top', ha='left', fontweight='bold')

# ---------- Panel (b) : Cv/N for multi-L ----------
for cfg in cv_configs:
    # AMIS 
    b, c = load_two_col(cfg["amis"])
    axB.plot(b, c, color=cfg["color"], ls='-', lw=1.2, alpha=0.55, zorder=3)
    # Kac-Ward 
    b, c = load_kacward(cfg["kw"])
    axB.plot(b, c, color=cfg["color"], ls='--', lw=1.2, alpha=0.9, zorder=4)
    # Training Points
    b, c = load_two_col(cfg["train"])
    axB.scatter(b, c, s=5, color=darken_color(cfg["color"]),
                zorder=5, linewidths=0)

axB.set_xlabel(r"$\beta$", labelpad=2)
axB.set_ylabel(r"$C_v/N$", labelpad=3)
axB.set_xlim(0.08, 1.02)
axB.set_ylim(bottom=0)
axB.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
axB.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

legend_b = [
    Line2D([], [], color='#1f77b4', ls='-', lw=1.2, label='$L=16$'),
    Line2D([], [], color='#2ca02c', ls='-', lw=1.2, label='$L=32$'),
    Line2D([], [], color='#d62728', ls='-', lw=1.2, label='$L=64$'),
    Line2D([], [], color='gray',    ls='-',  lw=1.2, alpha=0.55, label='MH-AMIS'),
    Line2D([], [], color='gray',    ls='--', lw=1.2, alpha=0.9,  label='Exact'),
    Line2D([], [], color='gray',    marker='o', ls='None', ms=2.8, label='VAN'),
]
axB.legend(handles=legend_b, frameon=False, loc='upper right',
           ncol=1, handlelength=1.3, borderpad=0.3, labelspacing=0.22,
           handletextpad=0.4)
axB.text(0.04, 0.96, '(b)', transform=axB.transAxes,
         fontsize=10, va='top', ha='left', fontweight='bold')

plt.tight_layout(pad=0.4, w_pad=1.4)
plt.savefig('fig2.pdf', bbox_inches='tight')
plt.savefig('fig2.png', dpi=300, bbox_inches='tight')
plt.close()