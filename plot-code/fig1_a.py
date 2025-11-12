import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots  # noqa: F401

plt.style.use(['science', 'nature'])
plt.rcParams["font.size"] = 10

# our_method
csv_files = sorted(glob.glob('data/*.csv'))
if not csv_files:
    raise RuntimeError("No CSV files found")

ours_min_wtr = []
for path in csv_files:
    df = pd.read_csv(path)
    columns_lower = {c.lower(): c for c in df.columns}
    tm_col = columns_lower['negative_tm_score']
    wt_col = columns_lower['recovery']
    plddt_col = columns_lower.get('negative_plddt')
    seq_col = columns_lower.get('query_sequence')
    df = df.drop_duplicates(subset=seq_col, keep='first')
    df = df[df[tm_col] <= -0.90]
    df = df[df[plddt_col] <= -90]
    
    ours_min_wtr.extend(df[wt_col].tolist())

df_all = [pd.DataFrame({"Method": "Ours (temp=0.3)", "Recovery_score": ours_min_wtr})]

# ProteinMPNN variants
for temp, label in [
    ("03", "ProteinMPNN (temp=0.3)"),
    ("07", "ProteinMPNN (temp=0.7)"),
    ("10", "ProteinMPNN (temp=1.0)"),
]:
    files = sorted(glob.glob(f'./output_mpnn_{temp}.csv'))
    pmpnn_wtr = []
    for path in files:
        df = pd.read_csv(path)
        columns_lower = {c.lower(): c for c in df.columns}
        wt_col = columns_lower.get('recovery')
        tm_col = columns_lower.get('negative_tm_score')
        plddt_col = columns_lower.get('negative_plddt')
        seq_col = columns_lower.get('query_sequence')
        df = df.drop_duplicates(subset=seq_col, keep='first')
        df = df[df[tm_col] <= -0.90]
        df = df[df[plddt_col] <= -90]
        
        pmpnn_wtr.extend(df[wt_col].tolist())

    df_all.append(pd.DataFrame({"Method": label, "Recovery_score": pmpnn_wtr}))

df = pd.concat(df_all)

method_order = [
    "Ours (temp=0.3)",
    "ProteinMPNN (temp=0.3)",
    "ProteinMPNN (temp=0.7)",
    "ProteinMPNN (temp=1.0)",
]
colors = ["#56B4E9", "#009E73", "#E69F00", "#CC79A7"]
palette = dict(zip(method_order, colors))

plt.figure(figsize=(5.5, 3))

sns.violinplot(
    data=df,
    x="Method",
    y="Recovery_score",
    order=method_order,
    palette=palette,
    width=0.7
)

plt.xlabel("")
plt.ylabel(r"$\mathrm{f}_{\text{recovery}}$", fontsize=12)
plt.tight_layout()
plt.savefig("./plot/fig1_all_multi.png", dpi=400)
plt.show()