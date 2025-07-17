import os
import pandas as pd
import glob
import subprocess
from natsort import natsorted


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

csv_files = natsorted(glob.glob('./data/*.csv'))
csv_files.append('./pMPNNdata/proteinMPNN03.csv')
file_basenames = [os.path.splitext(os.path.basename(p))[0] for p in csv_files]
MAX_PER_FILE = 100
output_fasta = "./SSNplot/pareto_sequences.fasta"
groupname_path = "./SSNplot/seq_groupnames.txt"

with open(output_fasta, "w") as fasta, open(groupname_path, "w") as groupfile:
    for path, basename in zip(csv_files, file_basenames):
        df = pd.read_csv(path)
        columns_lower = {c.lower(): c for c in df.columns}
        seq_col = columns_lower.get('query_sequence', None)
        if not seq_col:
            continue

        if basename == "proteinMPNN03":
            df = df.drop_duplicates(subset=seq_col, keep='first')
            selected = df[seq_col][:MAX_PER_FILE]
        else:
            df = df.drop_duplicates(subset=seq_col, keep='first')
            tm_col = columns_lower['negative_tm_score']
            wt_col = columns_lower['recovery']
            plddt_col = columns_lower['negative_plddt']

            df = df[df[tm_col] <= -0.9]
            df = df[df[plddt_col] <= -90]
            if df.empty:
                continue

            values1 = df[tm_col].values
            values2 = df[wt_col].values
            pareto_fronts = fast_non_dominated_sort(values1, values2)

            selected_idx = []
            for front in pareto_fronts:
                if len(selected_idx) + len(front) <= MAX_PER_FILE:
                    selected_idx.extend(front)
                else:
                    remain = MAX_PER_FILE - len(selected_idx)
                    selected_idx.extend(front[:remain])
                    break

            selected = df.iloc[selected_idx][seq_col]

        for idx, seq in enumerate(selected):
            label = f"{basename}_{idx+1}"
            fasta.write(f">{label}\n{seq}\n")
            groupfile.write(f"{basename}\n")

print(f"FASTA and groupnames list written to {output_fasta} and {groupname_path}")

cmd = [
    "python",
    "./SSNplot/pySSN_wrapper.py",
    "-i", "./SSNplot/pareto_sequences.fasta",
    "--metric", "Levenshtein",
    "--reducer", "tSNE",
    "--grouping", "./SSNplot/seq_groupnames.txt"
]

print("\n--- Running pySSN.py with UMAP visualization... ---\n")
result = subprocess.run(cmd)
if result.returncode == 0:
    print("pySSN.py ran successfully!")
else:
    print(f"pySSN.py failed with return code: {result.returncode}")


