import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from natsort import natsorted


import scienceplots # noqa: F401
plt.style.use(['science', 'nature'])

csv_files = natsorted(glob.glob('./data/*.csv'))
if not csv_files:
    raise RuntimeError("No CSV files found")

cmap1 = colormaps['tab20']
cmap2 = colormaps['tab20b']
colors = np.vstack([
    cmap1(np.linspace(0, 1, 20)),
    cmap2(np.linspace(0, 1, 20))
])[:len(csv_files)]

ours_min_wtr = []

for col, path in zip(colors, csv_files):
    df = pd.read_csv(path)
    columns_lower = {c.lower(): c for c in df.columns}
    tm_col = columns_lower['negative_tm_score']
    wt_col = columns_lower['recovery']
    plddt_col = columns_lower.get('negative_plddt', None)
    seq_col = columns_lower.get('query_sequence', None)

    if seq_col:
        df = df.drop_duplicates(subset=seq_col, keep='first')

    df = df[df[tm_col] <= -0.9] if tm_col else df
    df = df[df[plddt_col] <= -90] if plddt_col else df

    min_val = df[wt_col].min() if wt_col in df.columns else None
    if min_val is not None:
        ours_min_wtr.append(min_val)

pmpnn_files = sorted(glob.glob('./pMPNNdata/*.csv'))
pmpnn_wtr = []
for path in pmpnn_files:
    df = pd.read_csv(path)
    if 'recovery' in df.columns and not df.empty:
        pmpnn_wtr.extend(df['recovery'].tolist())

plt.figure(figsize=(7, 5))

x1 = np.random.normal(1, 0.07, size=len(ours_min_wtr))
x2 = np.random.normal(2, 0.07, size=len(pmpnn_wtr))
plt.scatter(x1, ours_min_wtr, s=40, alpha=0.8, color='tab:blue', label="Ours")
plt.scatter(x2, pmpnn_wtr, s=24, alpha=0.5, color='tab:orange', label="proteinMPNN only")

plt.xticks([1, 2], ["Ours", "proteinMPNN only"])
plt.ylabel("f_recovery")
plt.title("f_recovery (tmp = 0.3)")
plt.tight_layout()
plt.savefig("./plot/wtr_Ours_pMPNNonly.png", dpi=300)
# plt.show()
plt.close()
