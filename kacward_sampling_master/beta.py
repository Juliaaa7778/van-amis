import numpy as np
import pandas as pd

L       = 16       
DELTA_S = 1.0

DATA_PATH = f"output/obc_cv_L{L}_0.0001.txt"

df = pd.read_csv(
    DATA_PATH,
    comment="#",
    sep=r"\s+",
    header=None,
    usecols=[0, 4],
    names=["beta", "cv"],
)

beta = df["beta"].to_numpy(dtype=float)
cv   = df["cv"].to_numpy(dtype=float)


d_beta = np.diff(beta)
ds = L * np.sqrt(np.maximum(cv[:-1], 0.0)) / beta[:-1] * d_beta


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


out_txt = f"selected_betas_{L}.txt"
np.savetxt(
    out_txt,
    selected_beta,
    fmt="%.4f",
    header=f"Selected beta points (L={L}, delta_S={DELTA_S})",
)