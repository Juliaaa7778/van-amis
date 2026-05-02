import math
import time
import torch
import networkx as nx
import numpy as np
import fire

# ── Precision locks ────────────────────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_default_dtype(torch.float64)
# ───────────────────────────────────────────────────────────────────────────────

def compute_angle(i, j, k):
    """Signed turning angle at j when traversing k->j->i, in (-pi, pi]."""
    k2j = j - k
    j2i = i - j
    return (
        math.atan2(k2j[1], k2j[0]) - math.atan2(j2i[1], j2i[0]) + math.pi
    ) % (2.0 * math.pi) - math.pi

def precompute_B_matrix(G):
    """
    Build the Kac-Ward phase matrix B (topology only, no beta dependence).

    directed_edge_list layout:
        [0,  m)  -> canonical undirected edges (i,j) from G.edges()
        [m, 2m)  -> reversed edges (j,i)

    => weights_tensor[:m] are the m undirected edge weights for logcosh.

    Returns:
        B_tensor       -- complex128, shape (2m, 2m), on CPU
        weights_tensor -- float64,   shape (2m,),    on CPU
        m              -- int, number of undirected edges
    """
    pos = {n: np.array(n) for n in G.nodes()}
    m = G.number_of_edges()

    directed_edge_list = (
        [(i, j) for (i, j) in G.edges()]
        + [(j, i) for (i, j) in G.edges()]
    )
    directed_edge_dic = {edge: idx for idx, edge in enumerate(directed_edge_list)}

    B_np = np.zeros((2 * m, 2 * m), dtype=np.complex128)
    for i in G:
        for j in G.neighbors(i):
            for k in G.neighbors(j):
                if k != i:
                    angle = compute_angle(pos[i], pos[j], pos[k])
                    B_np[directed_edge_dic[(i, j)],
                         directed_edge_dic[(j, k)]] = np.exp(0.5j * angle)

    weights_np = np.array(
        [G[i][j]["weight"] for i, j in directed_edge_list], dtype=np.float64
    )
    return torch.from_numpy(B_np), torch.from_numpy(weights_np), m

def compute_logZ_scalar(B_tensor, weights_tensor, m, beta_val, num_nodes):
    """Pure forward pass, no gradient graph. Used for finite-difference derivatives."""
    with torch.no_grad():
        beta = torch.tensor(beta_val, dtype=torch.float64, device=B_tensor.device)
        tanh_w = torch.tanh(weights_tensor * beta)
        D_diag = tanh_w.to(torch.complex128)
        I = torch.eye(2 * m, dtype=torch.complex128, device=B_tensor.device)
        A = I - B_tensor * D_diag
        sign, logabsdet = torch.linalg.slogdet(A)
        logdet_real = (torch.log(sign) + logabsdet).real
        bw = weights_tensor[:m] * beta
        abs_bw = torch.abs(bw)
        logcosh_sum = torch.sum(
            abs_bw + torch.log1p(torch.exp(-2.0 * abs_bw)) - math.log(2.0)
        )
        logZ = num_nodes * math.log(2.0) + logcosh_sum + 0.5 * logdet_real
        return logZ.item()

def emit(line, fh=None):
    """Print to stdout and optionally mirror to an open file handle."""
    print(line)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()

def scan(L=64, beta_start=0.1, beta_end=1.0, beta_step=0.0001, output=None, device=None, h=1e-4):
    """
    Scan beta and compute thermodynamic observables for an L x L Ising model.

    Derivatives via central finite difference (3 forward passes per beta point).
    This avoids double-backward noise that appears when Cv is small at high beta.

    Step size h=1e-4 balances truncation error O(h^2) and float64 roundoff O(eps/h^2).
    Increase to h=2e-4 if L grows well beyond 128.

    Args:
        L          : linear lattice size
        beta_start : starting inverse temperature
        beta_end   : ending inverse temperature
        beta_step  : step size
        output     : output file path; defaults to <script_dir>/output/kacward_L{L}.txt
        device     : torch device string, e.g. 'cuda', 'cuda:0', 'cuda:1', 'cpu'
        h          : finite-difference step size (default 1e-4)
    """
    import os

    # -- Resolve device -------------------------------------------------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device '{device}' but CUDA is not available.")
    # -------------------------------------------------------------------------

    # -- Resolve output path --------------------------------------------------
    if output is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(script_dir, "output")
        output = os.path.join(out_dir, f"obc_cv_L{L}_0.0001.txt")
    else:
        out_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(out_dir, exist_ok=True)
    fh = open(output, "w")
    print(f"# Output -> {output}")
    # -------------------------------------------------------------------------

    emit(f"# Device: {device} | TF32 disabled: {not torch.backends.cuda.matmul.allow_tf32}", fh)
    emit(f"# L={L},  N={L*L} spins,  beta_c = ln(1+sqrt(2))/2 ~ 0.44069", fh)
    emit(f"# Derivatives: central finite difference, h={h}", fh)

    graph = nx.grid_2d_graph(L, L)
    for u, v in graph.edges():
        graph[u][v]["weight"] = 1.0
    num_nodes = graph.number_of_nodes()

    t0 = time.time()
    B_tensor, weights_tensor, m = precompute_B_matrix(graph)
    B_tensor = B_tensor.to(device)
    weights_tensor = weights_tensor.to(device)
    matrix_size = 2 * m
    vram_gb_each = (matrix_size ** 2 * 16) / 1024 ** 3
    emit(
        f"# Precomputation: {time.time() - t0:.2f}s | "
        f"Matrix: {matrix_size}x{matrix_size} | "
        f"~{vram_gb_each:.2f} GB/tensor",
        fh,
    )
    emit(
        f"# {'beta':>8}  {'logZ/N':>16}  {'FreeEnergy/N':>16}  "
        f"{'Energy/N':>16}  {'Cv':>16}  {'dt(s)':>8}",
        fh,
    )

    betas = np.arange(beta_start, beta_end + beta_step / 2.0, beta_step)

    for b_val in betas:
        t1 = time.time()

        # Three no_grad forward passes -- no gradient graph, no matrix-inverse noise
        logZ_c = compute_logZ_scalar(B_tensor, weights_tensor, m, b_val,     num_nodes)
        logZ_p = compute_logZ_scalar(B_tensor, weights_tensor, m, b_val + h, num_nodes)
        logZ_m = compute_logZ_scalar(B_tensor, weights_tensor, m, b_val - h, num_nodes)

        # Central difference: O(h^2) truncation error
        d1_val = (logZ_p - logZ_m) / (2.0 * h)
        d2_val = (logZ_p - 2.0 * logZ_c + logZ_m) / (h ** 2)

        free_energy    = -logZ_c / num_nodes / b_val
        energy_density = -d1_val / num_nodes
        specific_heat  = (b_val ** 2 / num_nodes) * d2_val

        dt = time.time() - t1
        emit(
            f"{b_val:8.4f}  {logZ_c / num_nodes:16.10f}  {free_energy:16.10f}  "
            f"{energy_density:16.10f}  {specific_heat:16.10f}  {dt:8.3f}",
            fh,
        )

    fh.close()
    print(f"# Done. Results saved to {output}")


if __name__ == "__main__":
    fire.Fire(scan)
    