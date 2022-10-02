"""
Tools for generating gene signatures and transforming data into cell state space.
This is pulled directly from scrnaseq_tools/cell_state_space/generate_gene_sig.py
12/11/2020
"""

import numpy as np
import scipy as sp
import pandas as pd
from random import seed
from random import random
from random import gauss
import numpy.random as nr
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns

# import mazebox.plotting._archetype_plots as archetype_plots

seed(1)

## read in reference data and clean it up
## cells is a list of cell IDs in the same order as file columns
## genes is a list of gene names in the same order as file rows
## ref_states is cluster labels, if available


def read_ref(
    file,
    cells=False,
    genes=False,
    ref_states=False,
    delim="\t",
    transpose=False,
    unlog=False,
    normalize=False,
):
    print(
        "WARNING: read_ref assumes that data is in TPM format (or log TPM). "
        "If data is not in this format, please change data before running."
    )
    ref_map = sc.read_text(file, delimiter=delim, first_column_names=False)
    ref_map.var_names_make_unique()
    ref_map.obs_names_make_unique()
    if transpose == True:
        print("transposing...")
        ref_map = ref_map.T
    sc.pp.filter_genes(ref_map, min_cells=10)
    sc.pp.filter_cells(ref_map, min_genes=1)
    if type(unlog) == str:
        print("un-logging...")
        if unlog == "log2":
            ref_map.X = np.exp2(ref_map.X)
        elif unlog == "loge":
            ref_map.X = np.exp(ref_map.X)
        elif unlog == "log1p":
            ref_map.X = np.expm1(ref_map.X)
    if type(normalize) != bool:
        print(f"normalizing data by {normalize} method...")

        if normalize == "scale":
            x = ref_map.X.transpose()  # to normalize each cell, not each gene
            x = (x - x.min()) / (x.max() - x.min())
            ref_map.X = x.transpose()  # transpose back

    if type(cells) != bool:
        ref_map.obs_names = cells
    if type(genes) != bool:
        ref_map.var_names = genes
    if type(ref_states) != bool:
        print(ref_map.shape)
        if type(ref_states) == pd.DataFrame or type(ref_states) == pd.Series:
            if type(ref_states.index[1]) == np.int64:
                print(
                    "Changing ref_states indices to string to match ref_map obs_names..."
                )
                ref_states.index = [str(i) for i in ref_states.index]
            else:
                ref_states.index = [str(i) for i in ref_map.obs_names.values]
                print(ref_states.head())

        ref_map.obs["State"] = ref_states
        ref_map.obs["State"] = ref_map.obs["State"].astype("category").values
        print(ref_map.obs["State"].head())
    elif ref_states == False:
        print("Please add cell state IDs to the reference data.")
    return ref_map


# deprecated; normalization now part of transform_tumor_space
def normalize(data, normalize="scale", transpose=True):
    if type(normalize) != bool:
        if normalize == "scale":
            if transpose == True:
                print("normalizing...")
                x = data.X.transpose()
                x = (x - x.min()) / (x.max() - x.min())  # assumes cells are columns
                data.X = x.transpose()
            else:
                data.X = (data.X - data.X.min()) / (
                    data.X.max() - data.X.min()
                )  # assumes cells are columns

    return data


## generate a gene signature using Scanpy's rank gene groups
def genesig_rank_groups(ref_map, ref_states, n_genes=4, state_id="State"):
    print(
        f"Creating gene signature using rank_gene_groups in Scanpy with {n_genes} genes..."
    )
    sc.tl.rank_genes_groups(ref_map, groupby=state_id)
    # sc.pl.rank_genes_groups(ref_map, n_genes=n_genes)
    plt.show()
    glist_keep = []
    for i in range(n_genes):
        for j in ref_map.uns["rank_genes_groups"]["names"][i]:
            glist_keep.append(j)
    ref_map = ref_map[:, glist_keep]
    data = pd.DataFrame(ref_map.X, columns=ref_map.var_names, index=ref_map.obs_names)
    ref_states_df = pd.DataFrame(ref_states, index=ref_map.obs_names)
    ref_states_df.columns = ["cluster"]
    gene_sig = pd.DataFrame(
        0, index=list(set(ref_map.obs[state_id])), columns=data.columns
    )
    gene_sig = gene_sig.transpose()
    for i, clust in enumerate(set(ref_map.obs[state_id])):
        sig = [0] * len(data.columns)
        nsig = 0
        for j, r in data.iterrows():
            if ref_states_df.loc[j, "cluster"] == clust:  # was [j, cluster]
                sig += r
                nsig += 1
        asig = sig / nsig
        gene_sig[clust] = asig
    # print("Gene signatures: ",gene_sig.head())
    plt.figure()
    plt.show()
    return ref_map, gene_sig, ref_states_df


## generate a gene signature using the average expression for each cell type of a list of genes
def genesig_ave(ref_map, glist, ref_states, clust_name="cluster"):
    print("Generating gene signatures by ave method...")
    glist_keep = []
    for i in glist:
        if i in ref_map.var_names:
            glist_keep.append(i)
    ref_map = ref_map[:, glist_keep]
    data = pd.DataFrame(ref_map.X, columns=ref_map.var_names, index=ref_map.obs_names)
    ref_states_df = pd.DataFrame(ref_states, index=ref_map.obs_names)
    gene_sig = pd.DataFrame(0, index=list(set(ref_states)), columns=data.columns)
    gene_sig = gene_sig.transpose()
    for i, clust in enumerate(set(ref_map.obs["State"])):
        sig = [0] * len(data.columns)
        nsig = 0
        for j, r in data.iterrows():
            if ref_states_df.loc[j, clust_name] == clust:
                sig += r
                nsig += 1
        asig = [x / nsig for x in sig]
        gene_sig[clust] = asig
    # gene_sig = gene_sig/np.linalg.norm(gene_sig) ### ADDED TODO
    print("Gene signatures: ")
    print(gene_sig.head())
    return ref_map, gene_sig, ref_states_df


## generate a gene signature using a given list of genes with values
def genesig_given(ref_map, genes):
    glist = genes.index.values
    glist_keep = []
    for i in glist:
        if i in ref_map.var_names:
            glist_keep.append(i)
    ref_map = ref_map[:, glist_keep]
    gene_sig = genes.loc[genes.index.isin(glist_keep)]
    print("Gene signatures: ", gene_sig.head())
    return ref_map, gene_sig


def transform_ref_space(ref_map, gene_sig, ref_state_df, norm=False, scale=False):
    print(
        "Shape of b (ref data) is: ",
        ref_map.shape,
        " and shape of A (gene sig matrix) is: ",
        gene_sig.shape,
    )
    print("Solving equation Ax = b...")
    ref_map.X = (ref_map.X.T / np.linalg.norm(ref_map.X, axis=1)).T
    df_inv, resid, rank, sing_values = np.linalg.lstsq(
        gene_sig, ref_map.X.T, rcond=1
    )  # FutureWarning without rcond statement
    cells = pd.DataFrame(
        (df_inv.T[:, 0 : len(set(ref_state_df["cluster"]))]), index=ref_state_df.index
    )
    if norm == True:
        print("Calculating norm of b...")
        norm_b = []
        for i in ref_map.X:
            norm_b.append(np.linalg.norm(i))
        data = cells
        for j in cells.columns:
            data[j] = cells[j] / norm_b
    else:
        data = cells
    if scale == True:
        scaling = data.max()
        data = data / scaling
        return data, scaling
    # data = np.arcsinh((data))
    else:
        return data


def transform_tumor_space_csv(
    tumor,
    gene_sig,
    unlog=False,
    type=None,
    scale=False,
    scaling=None,
    spliced=False,
    eps=0.001,
):
    if unlog == True:
        tumor = np.expm1(tumor)
    print("Transforming tumor data...")
    glist_keep = []
    for num, i in enumerate(gene_sig.index.values):
        if i in tumor.columns:
            glist_keep.append(i)
    gene_sig = gene_sig.loc[gene_sig.index.isin(glist_keep)]
    if np.all(np.linalg.norm(gene_sig, axis=0) != 1 - eps):
        print("Renormalizing gene signature")
        gene_sig = gene_sig / np.linalg.norm(gene_sig, axis=0)
    print("Gene signature matrix now has shape: ", gene_sig.shape)
    tumor = tumor[gene_sig.index.values]

    lanorm = np.linalg.norm(tumor, axis=1)
    tumorx = (tumor.T / np.linalg.norm(tumor, axis=1)).T
    data = pd.DataFrame(tumorx, columns=tumor.columns, index=tumor.index)

    df_inv, resid, rank, sing_values = np.linalg.lstsq(gene_sig, np.array(data).T)
    out = pd.DataFrame(df_inv.T)
    # out = np.arcsinh((out))
    if scale == True:
        out = out / scaling
    return out, gene_sig, tumor, lanorm


def transform_tumor_space(
    adata,
    gene_sig,
    unlog=False,
    type=None,
    scale=False,
    scaling=None,
    spliced=False,
    eps=0.001,
):
    tumor = adata.copy()
    if unlog == True:
        tumor.X = np.expm1(tumor.X)
    print("Transforming tumor data...")
    glist_keep = []
    for num, i in enumerate(gene_sig.index.values):
        if i in tumor.var_names:
            glist_keep.append(i)
        # if num % 1000 == 0:
        #     print(num)
    gene_sig = gene_sig.loc[gene_sig.index.isin(glist_keep)]
    if np.all(np.linalg.norm(gene_sig, axis=0) != 1 - eps):

        gene_sig = gene_sig / np.linalg.norm(gene_sig, axis=0)
    print("Gene signature matrix now has shape: ", gene_sig.shape)
    tumor = tumor[:, gene_sig.index.values]
    print("Tumor data shortened to genes in gene list...")
    if spliced == False:
        sc.pp.filter_cells(tumor, min_genes=1)
        print(tumor.X.shape)
        # (x.T/norm(x, axis = 1)).T should equal x/norm(x.T, axis = 0)

        if type == "csr":
            tumorx = sp.sparse.csr_matrix.todense(tumor.X)
        else:
            tumorx = tumor.X
    else:
        sc.pp.filter_cells(tumor, min_counts=1)

        print(tumor.layers["spliced"].shape)

        if type == "csr":
            tumorx = sp.sparse.csr_matrix.todense(tumor.layers["spliced"])
        else:
            tumorx = tumor.layers["spliced"]
    lanorm = np.linalg.norm(tumorx, axis=1)
    tumorx = (tumorx.T / np.linalg.norm(tumorx, axis=1)).T
    print("Tumor data subsetting complete.")
    data = pd.DataFrame(tumorx, columns=tumor.var_names, index=tumor.obs_names)
    # if norm == True:
    #     print("Calculating the norm of cells...")
    #     print(data.head())
    #     for c in data:
    #         data[c] = data[c]/(np.linalg.norm(data[c]))
    #     print(data.head())
    # print("Shape of b (tumor data) is: ", np.array(tumor).shape, " and shape of A (gene sig matrix) is: ", gene_sig.shape)
    df_inv, resid, rank, sing_values = np.linalg.lstsq(gene_sig, np.array(data).T)
    out = pd.DataFrame(df_inv.T)
    # out = np.arcsinh((out))
    if scale == True:
        out = out / scaling
    return out, gene_sig, tumor, lanorm


def transform_vel(
    tumor, gene_sig, norm=False, scale=False, scaling=None, normalize=False, lanorm=None
):
    # Make dataframe for velocity vectors of tumor data
    if type(lanorm) != type(None):
        tumor.layers["velocity"] = (tumor.layers["velocity"].T / lanorm).T
    else:
        tumor.layers["velocity"] = (
            tumor.layers["velocity"].T
            / np.linalg.norm(tumor.layers["velocity"], axis=1)
        ).T

    data_vel = pd.DataFrame(
        tumor.layers["velocity"], columns=tumor.var_names, index=tumor.obs_names
    )
    if type(normalize) != bool:
        if normalize == "scale":
            data_vel = data_vel.transpose()
            data_vel = (data_vel) / np.abs(tumor.X.transpose().max())
            data_vel = data_vel.dropna(0)
            data_vel = data_vel.transpose()

    df_inv, resid, rank, sing_values = np.linalg.lstsq(gene_sig, np.array(data_vel).T)
    cells = pd.DataFrame(df_inv.T)
    if norm == True:
        print("Calculating norm of b...")
        norm_b = []
        for i in np.array(data_vel):
            norm_b.append(np.linalg.norm(i))
        out = cells
        for j in cells:
            out[j] = cells[j] / norm_b
    else:
        out = cells
    if scale == True:
        out = out / scaling
    # out = np.arcsinh(out)
    return out, gene_sig, tumor


def phenotyping_recipe(
    adata,
    sig_matrix,
    groupby,
    save_as=None,
    subtypes=["SCLC-A", "SCLC-A2", "SCLC-N", "SCLC-P", "SCLC-Y"],
    unlog=False,
    type=None,
    scale=False,
    eps=0.001,
    velocity=False,
    save=False,
):
    """
    :param adata: AnnData object
    :param sig_matrix: signature matrix used to transform
    :param groupby: used to set groupby category in adata_small as dtype category
    :param subtypes: list of subtypes
    :param unlog:
    :param type: "csr" if sparse matrix, otherwise None
    :param scale: if signature scores should be rescaled
    :param eps:
    :return:
        adata: original adata object with added attributes: Phenotype, x_Score, x_Score_t1, and x_Score_pos for each x in list of subtypes
        adata_small:
    """
    sig_matrix = sig_matrix / np.linalg.norm(sig_matrix, axis=0)
    data, sig_matrix1, adata_small, lanorm = transform_tumor_space(
        adata, sig_matrix, unlog=unlog, scale=scale, type=type
    )
    data.index = adata_small.obs_names
    data.columns = sig_matrix.columns
    for s in data:
        adata_small.obs[f"{s}_Score"] = data[s]
        adata.obs[f"{s}_Score"] = data[s]

    if velocity:
        data_vel, sig_matrix2, adata_small = transform_vel(
            adata_small, sig_matrix1, scale=scale, lanorm=lanorm
        )
        data_vel.index = adata_small.obs_names
        data_vel.columns = sig_matrix.columns
        for s in data_vel:
            adata_small.obs[f"{s}_Score_t1"] = data_vel[s] + data[s]
            adata.obs[f"{s}_Score_t1"] = data_vel[s] + data[s]
    else:
        sig_matrix2 = sig_matrix

    # filtered phenotype
    pheno = []

    for i, r in data.iterrows():
        test = r * (r > 0)
        if test.max() < 0.1:
            pheno.append("None")

        elif test.max() > 0.5:
            #     if (test/test.sum()).max()>.9:
            pheno.append(data.columns[test.argmax()])
        else:
            pheno.append("Generalist")
    adata_small.obs["Phenotype"] = pheno
    adata_small.obs[groupby] = adata_small.obs[groupby].astype("category")
    adata.obs["Phenotype"] = adata_small.obs["Phenotype"]

    for i in subtypes:
        adata.obs[f"{i}_Score_pos"] = adata.obs[f"{i}_Score"] * (
            adata.obs[f"{i}_Score"] > 0
        )
        plt.figure(figsize=(4, 4))
        ax = plt.subplot()
        sns.boxplot(data=adata.obs, x=groupby, y=f"{i}_Score_pos")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        if save:
            plt.savefig(f"./figures/{save_as}/{i}.pdf")
    # cp = ['#fc8d62', '#66c2a5', '#FFD43B', '#8da0cb', '#e78ac3']
    # color_dict = {'SCLC-Y': cp[4], 'SCLC-A': cp[0], 'SCLC-A2': cp[1], 'SCLC-N': cp[2], 'SCLC-P': cp[3],
    #               'Generalist': 'darkgray', 'None': 'lightgrey'}
    # archetype_plots.archetype_diagrams(adata, sig_matrix, color_dict=color_dict, groupby='None', color='Phenotype',
    #                          grid=False, mean_only=True,
    #                          order=['SCLC-Y', 'SCLC-A', 'SCLC-P', 'SCLC-N', 'SCLC-A2'], norm='None', num_steps=40,
    #                          multiplier=1, figsize=(4, 4), score_name='_Score', alpha=.8, s=4, sizes=20, arrows=True)

    return adata, adata_small, sig_matrix2
