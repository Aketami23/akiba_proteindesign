import glob
import os
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

plt.figure(figsize=(10, 6))

for col, path in zip(colors, csv_files):
    df = pd.read_csv(path)
    columns_lower = {c.lower(): c for c in df.columns}
    tm_col = columns_lower['negative_tm_score']
    tmscores = df[tm_col].values

    min_so_far = np.inf
    min_values = []
    step_ids = []

    for i in range(0, len(tmscores), 100):
        window = tmscores[:i+1]
        if len(window) == 0:
            continue
        min_so_far = np.min(window)
        step_ids.append(i)
        min_values.append(min_so_far)

    if step_ids:
        plt.plot(step_ids, min_values, marker='o', markersize=4,
                 linewidth=1.5, color=col, alpha=0.9,
                 label=os.path.basename(path))

plt.xlabel('', fontsize=20)
plt.ylabel(r'$\mathrm{f}_{\text{structure}}$')
plt.tick_params(labelsize=15)
plt.ylim(0, -1)
plt.legend(fontsize='x-small', markerscale=0.8,
           bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.0, title_fontsize=12)

plt.tight_layout()
plt.savefig("./plot/f_structure_every5gen.png", format="png", dpi=300)
# plt.show()
plt.close()
