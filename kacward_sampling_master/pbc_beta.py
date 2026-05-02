import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent / "output" / "pbc_cv_L16_0.0001.txt"

df = pd.read_csv(DATA_PATH, comment="#", sep=r"\s+", header=None,
                 usecols=[0, 3], engine="python")

beta = df.iloc[:, 0].to_numpy(dtype=float).copy()
cv   = df.iloc[:, 1].to_numpy(dtype=float).copy()

L      = 16
d_beta = np.diff(beta)
ds = L * np.sqrt(np.maximum(cv[:-1], 0.0)) / beta[:-1] * d_beta
DELTA_S = 1.0

selected_beta = [beta[0]]
cumsum = 0.0

for i, dsi in enumerate(ds):
    cumsum += dsi
    if cumsum >= DELTA_S:
        selected_beta.append(beta[i + 1])
        cumsum = 0.0

if selected_beta[-1] < beta[-1] - 1e-8:
    selected_beta.append(beta[-1])

selected_beta = np.array(selected_beta)
out_txt = Path(__file__).parent / f"pbc_selected_betas_L{L}.txt"
np.savetxt(out_txt, selected_beta, fmt="%.4f",
           header=f"Selected beta points (L={L}, delta_S={DELTA_S})")
