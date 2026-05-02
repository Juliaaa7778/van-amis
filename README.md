# Variational Neural Annealing for the 2D Ising Model with MH-AMIS

A FlashAttention-based, patch-tokenized, **multi-temperature autoregressive
transformer** that learns the Boltzmann distribution of the 2D Ising model,
combined with **Metropolis–Hastings Adaptive Mixture Importance Sampling
(MH-AMIS)** for computing thermodynamic observables (free energy, internal
energy, magnetization, specific heat) at arbitrary inverse temperatures.

---

## 1. Method Overview

Three stages, each implemented as one or more standalone scripts:

1. **Training** (`train_observables.py`)
   A single autoregressive transformer is trained jointly on `K` inverse
   temperatures `β = (β_1, …, β_K)` using a REINFORCE-style variational
   free-energy loss. Key components:
   - Patch tokenization (`patch_size=2` ⇒ vocab size `2^4 = 16`)
   - β-conditioned input through a learned β-embedding token
   - Z₂ symmetry enforced by a `LogProbWrapper`
     (`log q(s) = log[(q_raw(s) + q_raw(−s))/2]`)
   - Three-phase critical annealing schedule (warmup → wandering near β_c
     → low-T hardening)
   - Muon optimizer (default), with AdamW / Adam fallbacks
   - FlashAttention 2 in the forward pass and KV-cached AR sampling

2. **Sample-pool construction** (`build_sample_pool.py`)
   For every training temperature `β_k`, draw `M` samples and store
   `(energy, configuration, log q(·|β_j))` for **all** `j = 1…K`. The
   cross-temperature `log q` matrix is what enables MH-AMIS at any target
   `β*`.

3. **MH-AMIS reweighting** (`free_energy_surface.py`, `observables.py`,
   `specific-heat.py`, `ess.py`)
   For each target `β*`, run independent-proposal Metropolis–Hastings on
   the pool with weights `w ∝ exp(−β* E) / q_mix` to estimate adaptive
   mixture coefficients `α_k(β*)`. The reweighted pool then yields
   self-normalized importance estimates of `log Z`, `⟨E⟩`, `⟨|m|⟩`, `C_v`,
   and the effective sample size.

---

## 2. Repository Layout

```
.
├── README.md
├── LICENSE                    # Apache-2.0
├── requirements.txt           # pip dependencies (base packages)
│
├── attention.py               # FlashAttention multi-head attention + transformer block
├── flash_ar.py                # Base AR model (with KV-cached sampling)
├── model_ising.py             # Patch-based, β-conditioned Ising model head
├── hamiltonian_ising.py       # 2D nearest-neighbor Ising energy (OBC / PBC)
├── muon.py                    # Muon optimizer (Newton–Schulz orthogonalization)
├── utils.py                   # Logger, checkpoint manager, seeding helpers
│
├── train_observables.py       # ➊ Training entry point
├── build_sample_pool.py       # ➋ Build cross-temperature sample pool
│
├── free_energy_surface.py     # ➌ AMIS free energy vs. Kac–Ward exact
├── observables.py             # ➌ ⟨E⟩/N, ⟨|m|⟩
├── specific-heat.py           # ➌ Specific heat C_v on a dense β grid
├── ess.py                     # ➌ Effective sample size diagnostics
│
├── plot_fig1.py               # log Z plots (L=16 PBC + L=64 OBC)
├── plot_fig2.py               # ⟨E⟩, |m| and C_v across L
├── plot_fig3.py               # ESS curve + α_k(β*) heatmap
├── relative_error.py          # Bootstrap-CI relative error of variational free energy
│
├── kacward_sampling_master/   # Bundled exact Kac–Ward baselines (code + data)
│   ├── kacward_sparse.py        #   sparse-LU log Z at an arbitrary β list (used to make obc_L64_free_energy.txt)
│   ├── ising_specific_heat.py   #   OBC C_v scan via Kac–Ward + central finite differences (GPU, fp64)
│   ├── pbc_specific_heat.py     #   PBC C_v scan via the Onsager solution evaluated with mpmath
│   ├── beta.py / pbc_beta.py    #   pick training β from a fixed-entropy ΔS criterion on the C_v curve
│   └── output/                  #   obc_cv_L{16,32,64}_0.0001.txt, pbc_cv_L16_0.0001.txt,
│                                #   obc_L64_free_energy.txt,
│                                #   obc_selected_betas_L{16,32,64}.txt, pbc_selected_betas_L16.txt
└── out/                       # Bundled training outputs / baselines
    ├── isinglnZ.npy           #   VaTD reference for L=16 PBC
    ├── L16_obc/  L16_pbc/  L32_obc/
    ├── L64_obc_1/  L64_obc_2/  L64_obc_3/  L64_obc_4/
    │   ├── *_final.pt                       # trained checkpoint
    │   ├── *_observables.txt, *_summary.txt # training-time evaluation
    │   ├── *_vfe_multitemp.txt              # variational free energy per β
    │   ├── sample_pool.npz                  # cross-temperature sample pool
    │   ├── free_energy_results.txt          # AMIS free energy + Kac-Ward
    │   ├── estimate_observables_fixed.txt   # AMIS E/N, |m|
    │   ├── specific_heat_data.txt           # AMIS C_v on dense grid
    │   ├── training_points_cv.txt           # AMIS C_v at training β only
    │   ├── normalized_ess_results*.txt      # ESS diagnostics
    │   └── alpha_matrix.npz                 # α_k(β*) for the heatmap
    └── …
```

All baseline data (`isinglnZ.npy`, the `kacward_sampling_master/` folder, and
the `out/L*/` reference runs) are tracked in this repository, so the plotting
scripts (`plot_fig1.py`, `plot_fig2.py`, `plot_fig3.py`, `relative_error.py`)
work out of the box.

---

## 3. Hardware Requirements

- **NVIDIA GPU**, Ampere or newer recommended (Turing also works for the
  fallback path). FlashAttention 2 is required for full performance.
- ~16 GB of GPU memory is sufficient up to `L = 64` with the default
  architecture (`patch_size=2, n_layers=2, n_heads=4`).
- A reasonably recent NVIDIA driver (≥ 525). The environment below has been
  tested with driver-reported CUDA 12.8 paired with the cu126 PyTorch wheel
  (CUDA drivers are backward-compatible across minor versions).

---

## 4. Installation

The following recipe is the exact procedure used to produce the bundled
results. Other combinations (e.g. CUDA 11.8) usually work too, but the
versions below are the ones that have been verified end-to-end.

### 4.1 Create a Python 3.11 environment

```bash
conda create -n python311 python=3.11 -y
conda activate python311
```

### 4.2 Install PyTorch 2.6.0 (cu126 wheels)

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.6.0+cu126 True
```

### 4.3 Install the rest of the Python dependencies

```bash
pip install -r requirements.txt
```

### 4.4 Install FlashAttention from a prebuilt wheel

`pip install flash-attn --no-build-isolation` will work but is slow and
fragile (it has to compile against your local toolchain). The robust
recipe is to download a matching prebuilt wheel from the
[FlashAttention releases page](https://github.com/Dao-AILab/flash-attention/releases)
and install it directly.

For Python 3.11 + PyTorch 2.6 + cu12 you need a wheel of the form

```
flash_attn-2.7.4.post1+cu12torch2.6cxx11abi{TRUE|FALSE}-cp311-cp311-linux_x86_64.whl
```

The `cxx11abi` suffix **must** match how your PyTorch was compiled. Check it:

```bash
python - <<'EOF'
import torch
print("Torch CXX11 ABI:", torch._C._GLIBCXX_USE_CXX11_ABI)
EOF
```

- Output `True`  → download the `cxx11abiTRUE` wheel
- Output `False` → download the `cxx11abiFALSE` wheel

Make sure the build prerequisites are present, then install:

```bash
pip install packaging ninja
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
```

(Replace `TRUE` with `FALSE` if that is what your PyTorch reported. If you
pick the wrong one, `import flash_attn_2_cuda` will fail with an
`undefined symbol` error at runtime — uninstall and reinstall the other
ABI variant.)

Verify:

```bash
python -c "from flash_attn import flash_attn_with_kvcache; print('flash-attn OK')"
```

### 4.5 (Optional) Install a CUDA Toolkit for `nvcc`

The PyTorch wheels above ship their own CUDA libraries, so a system-wide
CUDA Toolkit is **not required** to run this code. It is only needed if
you want to compile CUDA extensions yourself (e.g. building flash-attn
from source). If you do, follow the
[official archive](https://developer.nvidia.com/cuda-toolkit-archive),
choose the toolkit-only install option (skip the bundled driver), and
add it to your shell:

```bash
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH'             >> ~/.zshrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.zshrc
source ~/.zshrc
```

---

## 5. End-to-End Usage

### Step 1 — Train the model

```bash
python train_observables.py \
    --L 64 --boundary OBC \
    --patch_size 2 --token_dim 64 --pos_dim 256 \
    --n_layers 2 --n_heads 4 \
    --total_batch_size 2000 --num_steps 20000 \
    --annealing_steps 15000 \
    --optimizer muon --lr 1e-3 --scheduler \
    --seed 11 --cuda 0 \
    --betas_file out/L64_obc_1/selected_betas_L64.txt \
    --out_dir out/L64_obc_1
```

The list of training inverse temperatures `β_1, …, β_K` is supplied via
`--betas_file` (one β per line, `#`-comments allowed). This is the
preferred way once `K` becomes large; passing them inline with `--betas
0.1 0.2 0.3 …` also works and is convenient for short lists. If both
are omitted, `train_observables.py` falls back to its `DEFAULT_BETAS`
(0.1, 0.2, …, 1.0).

Output of this step (under `--out_dir`):
- `*_final.pt` — model checkpoint (contains `model_config`, `boundary`, `betas`)
- `*_summary.txt`, `*_observables.txt`, `*_vfe_multitemp.txt`, `*.log`

### Step 2 — Build the cross-temperature sample pool

```bash
python build_sample_pool.py \
    --model_path out/L64_obc_1/L64_OBC_multitemp_seed11_final.pt \
    --device cuda:0 \
    --samples_per_beta 10000 --batch_size 1000 \
    --boundary OBC --z2
```

⚠ The `--z2` flag **must** match what was used at training time.

Produces `out/L64_obc_1/sample_pool.npz`.

### Step 3 — Compute observables

```bash
# Free energy curve + comparison to Kac–Ward
python free_energy_surface.py \
    --pool_path out/L64_obc_1/sample_pool.npz --L 64

# Energy and magnetization
python observables.py \
    --pool_path out/L64_obc_1/sample_pool.npz

# Specific heat (dense grid through β_c, plus AMIS values at training β)
python specific-heat.py \
    --pool_path out/L64_obc_1/sample_pool.npz \
    --betas_file out/L64_obc_1/selected_betas_L64.txt

# Effective sample size
python ess.py \
    --pool_path out/L64_obc_1/sample_pool.npz
```

### Step 4 — Reproduce the figures

```bash
python plot_fig1.py            # log Z and relative error
python plot_fig2.py            # E, |m|, C_v across L
python plot_fig3.py --pool_path out/L64_obc_1/sample_pool.npz
python relative_error.py       # variational free-energy error with bootstrap CI
```

These scripts read the bundled baselines under `out/` and
`kacward_sampling_master/output/`, so they should run without further
configuration on a fresh clone.

---

## 6. Output File Conventions

| File | Produced by | Columns |
|---|---|---|
| `*_final.pt` | `train_observables.py` | `model_state_dict`, `betas`, `model_config`, `boundary`, … |
| `*_observables.txt` | `train_observables.py` | `β  E/N  std(E/N)  |m|  …` |
| `*_vfe_multitemp.txt` | `train_observables.py` | `β  F/N  std(F/N)` |
| `sample_pool.npz` | `build_sample_pool.py` | `energies`, `configs`, `log_probs_matrix`, `betas` |
| `free_energy_results.txt` | `free_energy_surface.py` | `β  f_AMIS  f_exact  rel_err` |
| `estimate_observables_fixed.txt` | `observables.py` | `β  E/N  |m|` |
| `specific_heat_data.txt` | `specific-heat.py` | `β  C_v` |
| `training_points_cv.txt` | `specific-heat.py --betas_file …` | `β  C_v` (training β only) |
| `normalized_ess_results*.txt` | `ess.py` | `β  ESS  ESS/N` |
| `alpha_matrix.npz` | `plot_fig3.py` | `α_k(β*)` matrix |

---

## 7. Notes & Caveats


- `--z2` at sample-pool time computes
  `log q_z2(s) = logsumexp(log q(s), log q(−s)) − log 2`, which is
  required for the importance weights to be normalized when the model
  itself was trained with the Z₂ wrapper.
- Random seeds are not perfectly deterministic on GPU because of
  FlashAttention's atomic reductions; expect small run-to-run variation
  even with a fixed `--seed`.

---

## 8. License

This project is released under the **Apache License 2.0** (see
[`LICENSE`](LICENSE)).

`muon.py` is adapted from the
[Moonlight](https://github.com/MoonshotAI/Moonlight) and
[KellerJordan/Muon](https://github.com/KellerJordan/Muon)
implementations; see comments at the top of that file for upstream
attribution.
