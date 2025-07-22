#Calculation of SSNs
#Author: Paul Jannis Zurek, pjz26@cam.ac.uk
#17/03/2021 v 1.0
# Calculates SSW alignments or Levensthein distance matrices
# Calculates tSNEs or UMAPs
# Generated checkpoints along the way

from Bio import SeqIO
from skbio.alignment import StripedSmithWaterman
import multiprocessing
import itertools
import numpy as np
from matplotlib import pyplot as plt
import argparse
from tqdm import tqdm
from Levenshtein import distance as LevenDist
import pandas as pd
import seaborn as sns
import umap
from sklearn.manifold import TSNE
import matplotlib


if __name__ == "__main__":
    sns.set(style='white', context='notebook', rc={'figure.figsize':(6,6)})

    parser = argparse.ArgumentParser(description="""Calculation of SSNs.
                                    Author: Paul Zurek (pjz26@cam.ac.uk).
                                    Version 1.0""")
    parser.add_argument('-T', '--threads', type=int, default=0, help='Number of threads to execute in parallel. Defaults to CPU count.')
    parser.add_argument('-i', '--input', type=str, help="""Please provide one of the following input files:\n 
                                                        FASTA: List of records to calculate distance matrix from.\n
                                                        CSV: Distance matrix checkpoint.
                                                        NPY: Reducer embeddings checkpoint.
                                                        """, required=True)
    parser.add_argument('-v', '--version', action='version', version='1.0')
    parser.add_argument('--metric', type=str, choices=['Levenshtein','Alignment'], default='Alignment', help='Metic used for distance calculation: Levenshtein or Alignment. Use Levenshtein for close sequences and Alignment for less homologous sequences.')
    parser.add_argument('--grouping', type=str, help='TXT file for group information. Can be used to color the SSN.')
    parser.add_argument('--reducer', type=str, choices=['UMAP','tSNE'], default="UMAP", help='Choice of dimensionality reduction method: UMAP or tSNE. Defaults to UMAP.')

    #Parse arguments
    args = parser.parse_args()
    grouping = args.grouping
    input_file = args.input
    metric = args.metric
    threads = args.threads
    reducer = args.reducer

    if threads == 0:
        threads = multiprocessing.cpu_count()

    name = ".".join(input_file.split(".")[:-1])
    ftype = input_file.split(".")[-1].lower()

    ##############
    # DEFINITIONS

    def all_scores(query_nr, record):   #Calculates all scores for distance_matrix
        global query_lst
        aln = query_lst[query_nr](record)
        #Alignment score
        score = aln.optimal_alignment_score
        #Calculate query coverage
        query_length = len(aln.query_sequence)
        aligned_query_length = aln.query_end - aln.query_begin + 1
        coverage = aligned_query_length / query_length
        #Calculate %identity
        aln_query = aln.aligned_query_sequence
        aln_target = aln.aligned_target_sequence
        aln_length = len(aln_query)
        same_aa = sum(e1 == e2 for e1, e2 in zip(aln_query, aln_target))
        ident = same_aa / aln_length
        return [score, ident, coverage]

    def distance_matrix(records):   #Calculates sparse distance matrix, based on %mismatches after SSW alignment
        rec_lst = records.copy()
        global query_lst
        query_lst = [StripedSmithWaterman(rec) for rec in rec_lst]
        pool = multiprocessing.Pool(processes=threads)
        score_lst_lst = []
        N_rec = len(rec_lst)
        for i in tqdm(range(N_rec)):
            rec_lst.pop(0)
            score_lst = pool.starmap(all_scores, zip(itertools.repeat(i), rec_lst))
            identlst = [1-elem[1] for elem in score_lst] #1-identity = dissimilarity
            score_lst_lst.append(identlst)
        pool.close()
        print('finished generating the alignment score matrix')
        return score_lst_lst

    def convert_DM_tofull(sparse_matrix):   #Converts sparse distance matrix to full matrix
        elem = len(sparse_matrix)
        full_matrix = [[0 for i in range(elem)] for j in range(elem)]
        for i in range(len(sparse_matrix)):
            for j in range(len(sparse_matrix[i])):
                full_matrix[i][j+1+i] = sparse_matrix[i][j]
                full_matrix[j+1+i][i] = sparse_matrix[i][j]
        return full_matrix

    def LevDistMat(records):
        rec_lst = records.copy()
        score_lst_lst = []
        N_rec = len(rec_lst)
        pool = multiprocessing.Pool(processes=threads)
        max_dist = 0
        for i in tqdm(range(N_rec)):
            ref = rec_lst.pop(0)
            score_lst = pool.starmap(LevenDist, zip(itertools.repeat(ref), rec_lst))
            m = max(score_lst, default=0)
            if m > max_dist:
                max_dist = m
            score_lst_lst.append(score_lst)
        pool.close()
        #Scale 0-1
        for i in range(len(score_lst_lst)):
            for j in range(len(score_lst_lst[i])):
                score_lst_lst[i][j] = score_lst_lst[i][j] / max_dist
        print('finished generating the levenshtein score matrix')
        return score_lst_lst

    def calc_reduction(DM):
        if reducer == "UMAP":
            model = umap.UMAP(metric="precomputed")
        elif reducer == "tSNE":
            model = TSNE(n_components=2, metric="precomputed", init="random")
        embedding = model.fit_transform(DM)
        ## metricをcheckpointの命名に追加
        np.save(f"{name}-{metric}-{reducer}-checkpoint.npy", embedding, allow_pickle=True)
        return embedding

    def plot_scatter_colored(embedding):
        colors = ["k" for _ in range(len(embedding[:,1]))]
        if grouping is not None:
            with open(grouping, 'r') as f:
                lines = f.readlines()
            if len(lines) != len(colors):
                raise ValueError("Number of groupings does not match number of embedded sequences.")
            colors = [_i.strip("\n") for _i in lines]
            try:
                colors = [int(c) for c in colors]
            except Exception:
                pass
        plt.figure()
        plt.scatter(embedding[:,0], embedding[:,1], c=colors, s=1)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f"{name}-{metric}-{reducer}.png", bbox_inches="tight", dpi=600)

    """
    plot_scatter_coloredでは、embeddingの色を書いたgroupingファイルを読み込みます。
    groupingファイルに、グループ分けを書いて、色は自動で振り分ける関数をかく。
    以下の関数plot_scatter_modified以外はほぼ元のスクリプトからコピペです
    """
    palette_25 = [
        '#e6194b',  # red
        '#3cb44b',  # green
        '#ffe119',  # yellow
        '#0082c8',  # strong blue
        '#f58231',  # orange
        '#911eb4',  # purple
        '#46f0f0',  # cyan
        '#f032e6',  # magenta
        '#d2f53c',  # lime
        '#008080',  # teal
        '#aa6e28',  # brown
        '#800000',  # maroon
        '#808000',  # olive
        '#000080',  # navy
        '#808080',  # gray
        '#000000',  # black
        '#bcf60c',  # bright lime
        '#4363d8',  # deep blue
        '#9a6324',  # darker brown
        '#ffd700',  # gold
        '#00ced1',  # dark turquoise
        '#ff1493',  # deep pink
        '#1f78b4',  # medium blue (from ColorBrewer)
        '#6a3d9a',  # deep purple (from ColorBrewer)
        '#b15928'   # reddish brown (from ColorBrewer)
    ]




    def plot_scatter_modified(embedding):
        with open(grouping, 'r') as f:
            groups = [line.strip() for line in f]
        unique_groups = list(dict.fromkeys(groups))

        # seabornで視認性の高いカラーパレット生成
        color_palette = palette_25
        group_to_color = dict(zip(unique_groups, color_palette))
        print(f"Unique groups: {unique_groups}")

        plt.figure()
        for group in unique_groups:
            group_mask = np.array(groups) == group
            group_points = embedding[group_mask]

            marker_style = '^' if group == 'proteinMPNN03' else 'o'
            point_size = 15 if group == 'proteinMPNN03' else 5

            plt.scatter(group_points[:, 0], group_points[:, 1],
                        c=[group_to_color[group]], s=point_size, label=group, marker=marker_style)

        plt.xticks([])
        plt.yticks([])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.savefig(f"{name}-{metric}-{reducer}.png", bbox_inches="tight", dpi=600)


    if ftype == 'fasta':
        records = [str(rec.seq) for rec in SeqIO.parse(input_file, "fasta")]
        if metric == 'Alignment':
            DM = distance_matrix(records)
        elif metric == 'Levenshtein':
            DM = LevDistMat(records)
        fullDM = convert_DM_tofull(DM)
        fullDM = pd.DataFrame(fullDM)
        fullDM.to_csv(f"{name}-{metric}-{reducer}-DM-checkpoint.csv") 
        embeddings = calc_reduction(fullDM)
        plot_scatter_modified(embeddings)

    elif ftype == "csv":  #From DM checkpoint
        print("Loading precomputed distance matrix")
        fullDM = pd.read_csv(input_file, index_col=0)
        print("calculating embeddings")
        embeddings = calc_reduction(fullDM)
        plot_scatter_modified(embeddings)

    elif ftype == "npy":  #From embeddings checkpoint (e.g. to re-color without re-calculation)
        embeddings = np.load(input_file, allow_pickle=True)
        plot_scatter_modified(embeddings)