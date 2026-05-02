import os
import fire
import mpmath
import numpy as np


def log_z_mp(n, j, beta, dps=50):
    """
    Compute log Z for n x n PBC Ising model using mpmath.
    All intermediate calculations done at `dps` decimal digits of precision.
    """
    mpmath.mp.dps = dps

    beta = mpmath.mpf(beta)
    j = mpmath.mpf(j)
    n_mp = mpmath.mpf(n)
    pi = mpmath.pi

    h_val = beta * j
    h_star_val = mpmath.atanh(mpmath.exp(-2 * h_val))
    beta_c = mpmath.log(1 + mpmath.sqrt(2)) / 2
    s = -1 if beta > beta_c else 1

    c2hs = mpmath.cosh(2 * h_star_val)
    s2hs = mpmath.sinh(2 * h_star_val)
    c2h = mpmath.cosh(2 * h_val)
    s2h = mpmath.sinh(2 * h_val)

    def gamma_val(r):
        arg = c2hs * c2h - s2hs * s2h * mpmath.cos(r * pi / n_mp)
        if arg < 1:
            arg = mpmath.mpf(1)
        return mpmath.acosh(arg)

    # Accumulate log(2*cosh(...)) and log(2*sinh(...)) terms
    # Use mpmath.log and handle sinh(0)=0 case explicitly
    log2 = mpmath.log(2)

    def log2cosh(x):
        return mpmath.log(2 * mpmath.cosh(x))

    def log2sinh(x):
        if x == 0:
            return mpmath.ninf  # log(0) = -inf, contributes 0 in exp-sum
        return mpmath.log(2 * mpmath.sinh(x))

    z_even_cosh = mpmath.mpf(0)
    z_even_sinh = mpmath.mpf(0)
    z_odd_cosh = mpmath.mpf(0)
    z_odd_sinh = mpmath.mpf(0)

    even_sinh_neginf = False
    for r in range(n):
        g_even = gamma_val(2 * r)
        g_odd = gamma_val(2 * r + 1)
        half_n = n_mp / 2

        z_even_cosh += log2cosh(half_n * g_even)
        z_odd_cosh += log2cosh(half_n * g_odd)
        z_odd_sinh += log2sinh(half_n * g_odd)

        val = half_n * g_even
        if val == 0:
            even_sinh_neginf = True
        else:
            z_even_sinh += log2sinh(val)

    # Combine four sectors using logsumexp logic in mpmath
    # Z = (1/2) * [2*sinh(2h)]^(n^2/2) * (e^z_odd_cosh + e^z_odd_sinh + e^z_even_cosh + s * e^z_even_sinh)
    terms = [z_odd_cosh, z_odd_sinh, z_even_cosh]
    signs = [1, 1, 1]

    if not even_sinh_neginf:
        terms.append(z_even_sinh)
        signs.append(s)
    # else: s * exp(-inf) = 0, skip

    # logsumexp with signs
    max_t = max(terms)
    exp_sum = sum(sgn * mpmath.exp(t - max_t) for t, sgn in zip(terms, signs))
    log_sum_part = max_t + mpmath.log(exp_sum)

    result = -log2 + mpmath.mpf(n**2) / 2 * mpmath.log(2 * mpmath.sinh(2 * h_val)) + log_sum_part
    return result


def scan(n_size=16, j=1.0, beta_start=0.1, beta_end=1.0, beta_step=0.0001,
         h_fd=1e-5, dps=50, output=None):
    """
    Compute specific heat per spin via central finite differences on log Z.

    Args:
        n_size     : linear lattice size (n x n)
        j          : coupling constant
        beta_start : starting inverse temperature
        beta_end   : ending inverse temperature
        beta_step  : scan step
        h_fd       : finite-difference step (can be small thanks to mpmath precision)
        dps        : decimal digits of precision for mpmath (default 50)
        output     : output file path
    """
    N = n_size ** 2

    if output is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))
        output = os.path.join(out_dir, f"pbc_cv_L{n_size}_{beta_step}.txt")
    out_dir = os.path.dirname(os.path.abspath(output))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    betas = np.arange(beta_start, beta_end + beta_step / 2.0, beta_step)

    print(f"# PBC Ising exact specific heat (mpmath, dps={dps})")
    print(f"# L={n_size}, N={N}, h_fd={h_fd}")
    header = f"# {'beta':>10s}  {'logZ/N':>18s}  {'energy/N':>18s}  {'cv':>18s}"
    print(header)

    rows = []
    for idx, b in enumerate(betas):
        b_val = float(b)

        lz_c = log_z_mp(n_size, j, b_val, dps)
        lz_p = log_z_mp(n_size, j, b_val + h_fd, dps)
        lz_m = log_z_mp(n_size, j, b_val - h_fd, dps)

        d1 = (lz_p - lz_m) / (2 * mpmath.mpf(h_fd))
        d2 = (lz_p - 2 * lz_c + lz_m) / (mpmath.mpf(h_fd) ** 2)

        energy = float(-d1 / N)
        cv = float(mpmath.mpf(b_val) ** 2 / N * d2)
        lzn = float(lz_c / N)

        rows.append((b_val, lzn, energy, cv))

        if idx % 1000 == 0 or idx == 0:
            print(f"{b_val:10.4f}  {lzn:18.12f}  {energy:18.12f}  {cv:18.12f}")

    with open(output, "w") as f:
        f.write(f"# PBC Ising L={n_size}, dps={dps}, h_fd={h_fd}\n")
        f.write(f"# {'beta':>10s}  {'logZ/N':>18s}  {'energy/N':>18s}  {'cv':>18s}\n")
        for b, lzn, en, cv in rows:
            f.write(f"{b:10.4f}  {lzn:18.12f}  {en:18.12f}  {cv:18.12f}\n")

    print(f"\n# Done. Results saved to {output}")


if __name__ == "__main__":
    fire.Fire(scan)