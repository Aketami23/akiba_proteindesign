import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots  # noqa: F401
plt.style.use(['science', 'nature'])


def first_front_only(values1, values2):
    assert len(values1) == len(values2) and len(values1) > 0
    n = len(values1)
    dominated = np.zeros(n, int)
    front = []

    for p in range(n):
        print(f"Processing point {p+1}/{n}")
        for q in range(n):
            if (values1[q] < values1[p] and values2[q] <= values2[p]) or \
               (values1[q] <= values1[p] and values2[q] < values2[p]):
                dominated[p] += 1
        if dominated[p] == 0:
            front.append(p)

    assert len(front) > 0
    return front


csv_files = [
    "output_mpnn_03.csv",
    "output_mpnn_07.csv",
    "output_mpnn_10.csv",
    "output_mpnn_20.csv",
    "output_mpnn_30.csv",
    "data/seed01.csv"
]

# seabornパレットを使用
palette = sns.color_palette([
    '#E69F00', '#56B4E9', '#009E73',
    '#0072B2', '#D55E00',
    '#CC79A7', '#000000'
])[:len(csv_files)]

plt.rcParams["font.size"] = 25
plt.figure(figsize=(9, 6))

hypervolume = []

for col, path in zip(palette, csv_files):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    tm, wt, plddt, seq = cols['negative_tm_score'], cols['recovery'], cols['negative_plddt'],cols['query_sequence']
    df = df.drop_duplicates(subset=seq, keep='first')
    df = df[df[plddt] <= -0]
    df = df[df[tm] <= -0]

    vals1, vals2 = df[tm].to_numpy(), df[wt].to_numpy()
    idx = first_front_only(vals1, vals2)
    pareto = np.atleast_2d(df.iloc[idx][[tm, wt]].to_numpy())
    pareto = pareto[np.argsort(pareto[:, 0])]

    base = os.path.basename(path)
    if base.startswith("output_mpnn_03"):
        label = "ProtienMPNN (temp0.3)"
    elif base.startswith("output_mpnn_07"):
        label = "ProtienMPNN (temp0.7)"
    elif base.startswith("output_mpnn_10"):
        label = "ProtienMPNN (temp1.0)"
    elif base.startswith("output_mpnn_20"):
        label = "ProtienMPNN (temp2.0)"
    elif base.startswith("output_mpnn_30"):
        label = "ProtienMPNN (temp3.0)"
    elif "seed01" in base:
        label = "Our_method(03)"
    else:
        label = "Unknown"
    plt.plot(
        pareto[:, 0], pareto[:, 1], 'o-', ms=4, lw=1.5,
        alpha=0.8, color=col, label=label
    )
    area = np.trapz(pareto[:, 1], pareto[:, 0])
    hypervolume.append((label, area))

plt.xlabel(r'$\mathrm{f}_{\text{structure}}$')
plt.ylabel(r'$\mathrm{f}_{\text{recovery}}$')
plt.tick_params(labelsize=15)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig('fig1_b.png', dpi=400)
plt.show()

for label, area in hypervolume:
    print(f"Hypervolume for {label}: {area:.3f}")