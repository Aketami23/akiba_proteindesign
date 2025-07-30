import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps, colors
import numpy as np
from natsort import natsorted
import glob
from pathlib import Path

import scienceplots # noqa: F401
plt.style.use(['science', 'nature'])

csv_files = natsorted(glob.glob('./data/*.csv'))
if not csv_files:
    raise RuntimeError("No CSV files found")

for csv_path in csv_files:

    name = Path(csv_path).stem
    print(f"Processing {name}...")

    df = pd.read_csv(csv_path)

    columns_lower = {c.lower(): c for c in df.columns}
    tm_col = columns_lower['negative_tm_score']
    wt_col = columns_lower['recovery']
    qseq_col = columns_lower.get('query_sequence')

    max_generation = 500
    points_per_generation = 20

    tm_list, wt_list, group_ids = [], [], []
    seen_qseq = set()

    for idx, row in df.iterrows():
        gen_id = idx // points_per_generation + 1 
        if gen_id > max_generation:
            break

        if qseq_col is not None:
            qseq = row[qseq_col]
            if qseq in seen_qseq:
                continue
            seen_qseq.add(qseq)

        tm_list.append(row[tm_col])
        wt_list.append(row[wt_col])
        group_ids.append(gen_id)

    cmap = colormaps['plasma']
    fixed_colors = cmap(np.linspace(0, 1, max_generation+1))
    discrete_cmap = colors.ListedColormap(fixed_colors[1:])
    norm = colors.BoundaryNorm(np.arange(0.5, max_generation + 1.5, 1), max_generation)

    plt.rcParams["font.size"] = 25

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(tm_list, wt_list, c=group_ids,
                    cmap=discrete_cmap, norm=norm,
                    s=20, edgecolors='none')

    cbar = plt.colorbar(sc, label='Generation Number',)
    tick_positions = np.concatenate([[1], np.arange(50, max_generation + 1, 50)])
    cbar.set_ticks(tick_positions)

    plt.xlim(-1.0, 0.0)
    plt.ylim(0.0, 0.33)

    plt.xlabel(r'$\mathrm{f}_{\text{structure}}$')
    plt.ylabel(r'$\mathrm{f}_{\text{recovery}}$')
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'plot/fig3/{name}_tm_wtr_generation_scatter.png', dpi=300, format='png')
    plt.close()

