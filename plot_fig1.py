import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

# ------------------------------------------
# 1. L=16 PBC : AMIS + Exact + VaTD
# ------------------------------------------
amis_L16_path = 'out/L16_pbc/pbc_free_energy_results.txt'
d16 = np.loadtxt(amis_L16_path, skiprows=2)
beta_L16        = d16[:, 0]
f_amis_L16      = d16[:, 1]
f_exact_L16     = d16[:, 2]
logZ_amis_L16   = -beta_L16 * f_amis_L16       # log Z / N
logZ_exact_L16  = -beta_L16 * f_exact_L16

vatd_path = 'out/isinglnZ.npy'
with open(vatd_path, 'rb') as fv:
    vatd_beta = np.load(fv)
    _         = np.load(fv)          
    vatd_lnZ  = np.load(fv)

# relative error
exact_at_vatd  = np.interp(vatd_beta, beta_L16, logZ_exact_L16)
err_vatd_L16   = np.abs((exact_at_vatd - vatd_lnZ) / exact_at_vatd)
err_amis_L16   = np.abs((logZ_exact_L16 - logZ_amis_L16) / logZ_exact_L16)

# ------------------------------------------
# 2. L=64 OBC : AMIS + Exact 
# ------------------------------------------
amis_L64_path = 'out/L64_obc_1/free_energy_results.txt'
d64 = np.loadtxt(amis_L64_path, skiprows=2)
beta_L64     = d64[:, 0]
f_amis_L64   = d64[:, 1]
f_exact_L64  = d64[:, 2]
err_amis_L64 = np.abs((f_exact_L64 - f_amis_L64) / f_exact_L64)   # 与 txt 第 4 列一致

# ==========================================
# Figure:  1 x 2 , NeurIPS full-width
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.4))

# -------- Panel (a): L=16 log Z / N ---------
ax1.plot(beta_L16,  logZ_exact_L16, color='dimgrey',        ls='-',  lw=1.6, alpha=0.9,
         label='Exact', zorder=1)
ax1.plot(vatd_beta, vatd_lnZ,       color='mediumseagreen', ls='-.', lw=1.3,
         label='VaTD',  zorder=2)
ax1.plot(beta_L16,  logZ_amis_L16,  color='hotpink',        ls='--', lw=1.3,
         label='MH-AMIS',  zorder=3)

ax1.set_xlabel(r'$\beta$', labelpad=2)
ax1.set_ylabel(r'$\log\mathcal{Z}\,/\,N$', labelpad=2)
ax1.legend(loc='lower right', frameon=False)
ax1.text(0.04, 0.96, '(a)', transform=ax1.transAxes,
         fontsize=10, va='top', ha='left', fontweight='bold')

# Inset (a) : Relative Error  
axins1 = ax1.inset_axes([0.22, 0.58, 0.36, 0.32])
axins1.semilogy(vatd_beta, err_vatd_L16, color='mediumseagreen', lw=1.0)
axins1.semilogy(beta_L16,  err_amis_L16, color='hotpink',        lw=1.0)
axins1.set_ylabel('Rel. Error', fontsize=7, labelpad=1)
axins1.tick_params(axis='both', which='major', labelsize=6, pad=1)
axins1.tick_params(axis='both', which='minor', labelsize=6)

# -------- Panel (b): L=64 f(beta) ----------
ax2.plot(beta_L64, f_exact_L64, color='black',    ls='--', lw=1.3,
         label='Exact', zorder=2)
ax2.plot(beta_L64, f_amis_L64,  color='tab:blue', ls='-',  lw=1.3, alpha=0.85,
         label='MH-AMIS',  zorder=3)

ax2.set_xlabel(r'$\beta$', labelpad=2)
ax2.set_ylabel(r'$f(\beta)$', labelpad=2)
ax2.legend(loc='upper right', frameon=False, bbox_to_anchor=(0.98, 0.96))
ax2.text(0.04, 0.96, '(b)', transform=ax2.transAxes,
         fontsize=10, va='top', ha='left', fontweight='bold')

# Inset (b) : Relative Error
axins2 = ax2.inset_axes([0.46, 0.18, 0.50, 0.40])
axins2.semilogy(beta_L64, err_amis_L64, color='red', lw=1.0)
axins2.set_ylabel('Rel. Error', fontsize=7, labelpad=1)
axins2.tick_params(axis='both', which='major', labelsize=6, pad=1)
axins2.set_yticks([1e-8, 1e-7, 1e-6])

plt.tight_layout(pad=0.4, w_pad=1.2)
plt.savefig('fig1.pdf', bbox_inches='tight')
plt.savefig('fig1.png', dpi=300, bbox_inches='tight')
plt.close()