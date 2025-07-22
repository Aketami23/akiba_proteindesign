import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scienceplots # noqa: F401
plt.style.use(['science', 'nature'])

plt.rcParams["font.size"] = 10

csv_files = sorted(glob.glob('./data/*.csv'))
if not csv_files:
    raise RuntimeError("No CSV files found")

ours_min_wtr = []

for path in csv_files:
    df = pd.read_csv(path)
    columns_lower = {c.lower(): c for c in df.columns}
    tm_col = columns_lower['negative_tm_score']
    wt_col = columns_lower['recovery']
    plddt_col = columns_lower.get('plddt', None)
    seq_col = columns_lower.get('query_sequence', None)

    if seq_col:
        df = df.drop_duplicates(subset=seq_col, keep='first')

    df = df[df[tm_col] <= -0.953] if tm_col else df
    df = df[df[plddt_col] <= -91.907] if plddt_col else df

    min_val = df[wt_col].min() if wt_col in df.columns else None
    if min_val is not None:
        ours_min_wtr.append(min_val)

pmpnn_files = sorted(glob.glob('./pMPNNdata/*.csv'))
pmpnn_wtr = []
for path in pmpnn_files:
    df = pd.read_csv(path)
    if 'recovery' in df.columns and not df.empty:
        pmpnn_wtr.extend(df['recovery'].tolist())

df_ours = pd.DataFrame({"Method": "Ours", "Recovery score": ours_min_wtr})
df_pmpnn = pd.DataFrame({"Method": "ProteinMPNN", "Recovery score": pmpnn_wtr})
df = pd.concat([df_ours, df_pmpnn], ignore_index=True)

sns.stripplot(data=df, x="Method", y="Recovery score", hue="Method", size=4, jitter=0.05)
sns.boxplot(data=df, x="Method", y="Recovery score", color='w', width=0.4, showfliers=False)

plt.xlabel("")
plt.ylabel(r"$\mathrm{f}_{\text{recovery}}$", fontsize=12)
plt.tight_layout()
plt.savefig("./plot/fig1.png", dpi=300)
plt.close()