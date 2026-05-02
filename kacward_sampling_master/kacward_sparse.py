import math
import fire
import networkx as nx
import numpy as np
import scipy


def logcosh(x):
    return np.abs(x) + np.log(1 + np.exp(-2.0 * np.abs(x))) - np.log(2.0)


def compute_angle(i, j, k):
    k2j = j - k
    j2i = i - j
    return (math.atan2(k2j[1], k2j[0]) - math.atan2(j2i[1], j2i[0]) + math.pi) % (2.0 * math.pi) - math.pi


def logZ(G, beta):
    pos = {(i, j): np.array([i, j]) for i, j in G.nodes()}
    m = G.number_of_edges()
    directed_edge_list = [(i, j) for (i, j) in G.edges()] + [(j, i) for (i, j) in G.edges()]
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
        D[directed_edge_dic[(i, j)], directed_edge_dic[(i, j)]] = np.tanh(G[i][j]["weight"] * beta)
    logZ_val = G.number_of_nodes() * np.log(2.0)
    for i, j in G.edges():
        logZ_val += logcosh(G[i][j]["weight"] * beta)
    A = scipy.sparse.eye(2 * m, 2 * m, dtype=np.complex128) - B @ D
    A = scipy.sparse.csc_matrix(A)
    LU = scipy.sparse.linalg.splu(A)
    logdet = np.sum(np.log(LU.U.diagonal()))
    logdet = np.real(logdet)
    logZ_val += 0.5 * logdet

    return logZ_val


def run(L=16, beta_file=None, output_file=None, seed=1):
    rng = np.random.default_rng(seed)
    graph = nx.grid_2d_graph(L, L)
    for u, v in graph.edges():
        graph[u][v]["weight"] = 1
    
    if beta_file:
        with open(beta_file, 'r') as f:
            betas = [float(line.strip()) for line in f if line.strip() and not line.startswith('#')]
        
        results = []
        for beta in betas:
            lnZ = logZ(graph, beta)
            free_energy = -lnZ / (L**2) / beta
            results.append(f"{L} {beta} {lnZ} {free_energy}\n")
            print(f"{L} {beta} {lnZ} {free_energy}")
        
        if output_file:
            with open(output_file, 'w') as f:
                f.writelines(results)
    else:
        lnZ = logZ(graph, beta)
        free_energy = -lnZ / (L**2) / beta
        print(f"{L} {beta} {lnZ} {free_energy}")


if __name__ == "__main__":
    fire.Fire(run)