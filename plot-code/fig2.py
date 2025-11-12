import dataclasses
import glob
from natsort import natsorted

import pandas as pd
import matplotlib.pyplot as plt

import multiprocessing
import itertools
from tqdm import tqdm
from Bio import SeqIO

import numpy as np
from sklearn.manifold import TSNE

@dataclasses.dataclass
class ExprimentRun:
    label: str
    csv_path: str
    method: str
    seed: int

def simple_match_ratio(s1, s2):
    n = max(len(s1), len(s2))
    if n == 0:
        return 1.0
    matches = sum(c1 == c2 for c1, c2 in zip(s1, s2))
    return 1 - matches / n

def SimpleMatchMat(records):
    rec_lst = records.copy()
    score_lst_lst = []
    N_rec = len(rec_lst)
    pool = multiprocessing.Pool(processes=threads)

    for i in tqdm(range(N_rec)):
        ref = rec_lst.pop(0)
        sims = pool.starmap(simple_match_ratio, zip(itertools.repeat(ref), rec_lst))
        score_lst_lst.append(sims)

    pool.close()
    pool.join()

    print('finished generating the positional match ratio matrix')
    return score_lst_lst

def convert_DM_tofull(sparse_matrix):
    elem = len(sparse_matrix)
    full_matrix = [[0 for i in range(elem)] for j in range(elem)]
    for i in range(len(sparse_matrix)):
        for j in range(len(sparse_matrix[i])):
            full_matrix[i][j+1+i] = sparse_matrix[i][j]
            full_matrix[j+1+i][i] = sparse_matrix[i][j]
        
    return full_matrix

def calc_reduction(DM):
    model = TSNE(n_components=2, metric="precomputed", init="random")
    embedding = model.fit_transform(DM)
    return embedding

def plot_scatter_modified(embedding, grouping_path):
    palette_25 = [
        "#3819e6",
        '#3cb44b',
        '#ffe119',
        "#8fc800",
        '#f58231',
        '#911eb4',
        "#72efef",
        '#f032e6',
        '#d2f53c',
        '#008080',
        "#edbe89",
        '#800000',
        "#4D4D08",
        '#000080',
        '#808080',
        '#000000',
        "#bb3800",
        "#435cd8",
        '#9a6324',
        '#ffd700',
        "#d1c300",
        '#ff1493',
        '#6a3d9a',
        "#736056",
        "#ed1717",
        "#ff7f00",
        "#60f471",
        "#ff69b4",
    ]
    with open(grouping_path, 'r') as f:
        groups = [line.strip() for line in f]
    unique_groups = list(dict.fromkeys(groups))
    color_palette = palette_25
    group_to_color = dict(zip(unique_groups, color_palette))
    print(f"Unique groups: {unique_groups}")

    plt.figure()
    for group in unique_groups:
        group_mask = np.array(groups) == group
        group_points = embedding[group_mask]

        marker_style = (
            ',' if group == 'Ref_sequence'
            else 'x' if group in ['ProteinMPNN (temp=0.3)', 'ProteinMPNN (temp=0.7)', 'ProteinMPNN (temp=1.0)']
            else 'o'
        )

        point_size = (50 if group == 'Ref_sequence'
                      else 10 if group in ['ProteinMPNN (temp=0.3)', 'ProteinMPNN (temp=0.7)', 'ProteinMPNN (temp=1.0)']
                      else 5)

        plt.scatter(
            group_points[:, 0], group_points[:, 1],
            c=[group_to_color[group]],
            s=point_size,
            label=group,
            marker=marker_style,
            lw=0.5,
            alpha=0.6,
        )
    plt.xticks([])
    plt.yticks([])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    # plt.savefig(".fig2.png", bbox_inches="tight", dpi=400)
    plt.show()

# our_method 系
csv_files = natsorted(glob.glob('./data/*.csv'))
runs = [
    ExprimentRun(
        label=f"Our_method_{i}",
        csv_path=path,
        method="our_method",
        seed=i
    )
    for i, path in enumerate(csv_files)
]

# add proteinMPNN
mpnn_settings = [
    # ("ProteinMPNN (temp=1.0)", "../output_mpnn_10.csv", "mpnn"),
    # ("ProteinMPNN (temp=0.7)", "../output_mpnn_07.csv", "mpnn"),
    # ("ProteinMPNN (temp=0.3)", "../output_mpnn_03.csv", "mpnn"),
    ("Ref_sequence", "./req_seq.csv", "raw"),
]

runs.extend(
    ExprimentRun(label=label, csv_path=path, method=method, seed=0)
    for label, path, method in mpnn_settings
)

fasta_path = "./filtered_sequences.fasta"
groupname_path = "./seq_groupnames.txt"

records = []

with open(fasta_path, "w") as fasta, open(groupname_path, "w") as groupfile:

    for run in runs:
        df = pd.read_csv(run.csv_path)
        columns_lower = {c.lower(): c for c in df.columns}
        seq_col = columns_lower.get('query_sequence')
        df = df.drop_duplicates(subset=seq_col, keep='first')
        tm_col = columns_lower.get('negative_tm_score')
        plddt_col = columns_lower.get('negative_plddt')
        re_col = columns_lower.get('recovery')
        df = df[df[tm_col] <= -0.90]
        df = df[df[plddt_col] <= -90.0]
        n = len(df)
        sample_size = min(300, n)
        print(f"Run: {run.label}, Total: {n}, Sampling: {sample_size}")
        selected_indices = df.sample(n=sample_size, random_state=42).index.tolist()

        seq_col = columns_lower.get('query_sequence')
        for idx in selected_indices:
            tm = -df.loc[idx, tm_col]
            plddt = -df.loc[idx, plddt_col]
            records.append({"method": run.method, "label": run.label, "tm": tm, "plddt": plddt})
            seq = df.loc[idx, seq_col]
            label = f"{run.label}_{idx}"
            fasta.write(f">{label}\n{seq}\n")
            groupfile.write(f"{run.label}\n")

threads = 0
if threads == 0:
    threads = multiprocessing.cpu_count()

multiprocessing.set_start_method('fork', force=True)

records = [str(rec.seq) for rec in SeqIO.parse(fasta_path, "fasta")]

DM = SimpleMatchMat(records)

fullDM = convert_DM_tofull(DM)
fullDM = pd.DataFrame(fullDM)

np.save("./DM-checkpoint.npy", fullDM.to_numpy(), allow_pickle=True)

embeddings = calc_reduction(fullDM)

plot_scatter_modified(embeddings, groupname_path)
