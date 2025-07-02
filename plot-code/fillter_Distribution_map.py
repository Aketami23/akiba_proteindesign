
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps

try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except ImportError:
    pass

csv_files = sorted(glob.glob('./data/*.csv'))
if not csv_files:
    raise RuntimeError("No CSV files found")

cmap1 = colormaps['tab20']
cmap2 = colormaps['tab20b']
colors = np.vstack([
    cmap1(np.linspace(0, 1, 20)),
    cmap2(np.linspace(0, 1, 20))
])[:len(csv_files)]

def fast_non_dominated_sort(values1, values2):
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            if (values1[p] < values1[q] and values2[p] < values2[q]) or \
                (values1[p] <= values1[q] and values2[p] < values2[q]) or \
                (values1[p] < values1[q] and values2[p] <= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] < values1[p] and values2[q] < values2[p]) or \
                (values1[q] <= values1[p] and values2[q] < values2[p]) or \
                (values1[q] < values1[p] and values2[q] <= values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front

plt.figure(figsize=(9, 6))

for col, path in zip(colors, csv_files):
    df = pd.read_csv(path)
    columns_lower = {c.lower(): c for c in df.columns}
    tm_col = columns_lower['tm_score']
    wt_col = columns_lower['wild_type_recovery']
    plddt_col = columns_lower.get('plddt', None)
    seq_col = columns_lower.get('query_sequence', None)

    if seq_col:
        df = df.drop_duplicates(subset=seq_col, keep='first')

    df = df[df[tm_col] <= -0.9] if tm_col else df
    df = df[df[plddt_col] >= 90] if plddt_col else df
    values1 = df[tm_col].values
    values2 = df[wt_col].values
    fronts = fast_non_dominated_sort(values1, values2)
    if not fronts or not fronts[0]:
        print(f"{os.path.basename(path)}: No Pareto front found, skipping.")
        continue
    selected_idx = fronts[0]
    pareto = df.iloc[selected_idx][[tm_col, wt_col]].values
    pareto_sorted = pareto[np.argsort(pareto[:, 0])]
    plt.plot(pareto_sorted[:, 0], pareto_sorted[:, 1],
             marker='o', markersize=4, linewidth=1.5,
             alpha=0.9, color=col, label=os.path.basename(path))

plt.xlabel('TM-score (minimize)')
plt.ylabel('Wild-type Recovery (minimize)')
plt.title('Pareto Fronts')
plt.legend(title='CSV Files', fontsize='x-small', markerscale=0.8,
           bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.0)

plt.tight_layout()
plt.savefig("plot/Distribution_map.png", format="png", dpi=300)
plt.show()
