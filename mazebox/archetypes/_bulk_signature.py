# for generating gene signature from bulk RNA-seq data
# Note: this code is currently used in generate_gene_sig.py in Groves-CellSys2021

import os
import numpy as np
import pandas as pd
import time as tm
import scipy.stats as ss
import statsmodels.stats.multitest as mt

# sort_by_ascending is used if a lower number in sort_by is better and should be kept.
# For sort_by = logfc, A higher logfc is more significant so sort_by_ascending = False.
def generate_signature(
    archetypes,
    sig_df,
    range_min=20,
    range_max=200,
    range_step=2,
    norm=2,
    threshold=0.05,
    subset_by="p_adj",
    sort_by="logfc",
    print_c=False,
    sort_by_ascending=False,
):
    """

    :param archetypes:
    :param sig_df:
    :param range_min:
    :param range_max:
    :param range_step:
    :param norm:
    :param threshold:
    :param subset_by:
    :param sort_by:
    :param print_c:
    :param sort_by_ascending:
    :return:
    """
    cond = np.inf
    n_genes = 0
    df_keep = None

    # choose best number of genes
    for n in range(range_min, range_max, range_step):
        c, df = _condition_number(
            archetypes,
            sig_df,
            n_genes=n,
            threshold=threshold,
            norm=norm,
            subset_by=subset_by,
            sort_by=sort_by,
            sort_by_ascending=sort_by_ascending,
        )
        if print_c == True:
            print("C: ", c, "n: ", n)
        if c < cond:
            cond = c
            n_genes = n
            df_keep = df

    # get condition number, genes, and matrix of best n_genes
    _condition_number(
        archetypes,
        sig_df,
        n_genes=n_genes,
        threshold=threshold,
        norm=norm,
        subset_by=subset_by,
        sort_by=sort_by,
        sort_by_ascending=sort_by_ascending,
        print_g=True,
    )
    return df_keep, cond, n_genes


def _condition_number(
    archetypes,
    sig_df,
    threshold=0.05,
    n_genes=200,
    subset_by="p_adj",
    sort_by="logfc",
    norm=2,
    sort_by_ascending=False,
    print_g=False,
):
    genes = sig_df.loc[sig_df[subset_by] < threshold].sort_values(
        by=["subtype", subset_by], ascending=True
    )
    small_genes = pd.DataFrame()

    for p in sorted(list(set(genes["subtype"]))):
        g = genes.loc[genes["subtype"] == p]
        g = g.sort_values(by=[sort_by], ascending=sort_by_ascending)
        g = g.iloc[0:n_genes]
        small_genes = small_genes.append(
            g
        )  # only take top n genes from each phenotype and add to one df
        if print_g == True:
            print(p, g.index)
    cond = np.linalg.cond(
        x=np.matrix(archetypes.loc[small_genes.index], dtype=float), p=norm
    )
    return cond, archetypes.loc[small_genes.index]


def anova(data, labels, threshold=0.05, label="label", test="kruskal"):
    if data.max().max() > 1000:
        print(
            "It appears data has not been log-transformed. Transforming now using log1p."
        )
        data = np.log1p(data)
    arrays = [labels[label], labels.index.values]
    index = pd.MultiIndex.from_arrays(arrays, names=("Phenotype", "Sample"))
    data = data[labels.index]
    df = pd.DataFrame(np.array(data).T, columns=data.index.values, index=index)
    df_grouped = df.groupby(level="Phenotype")
    anova_df = pd.DataFrame(index=data.index)
    stats = []
    pvals = []
    logfcs = []
    high_group = []
    var = []
    for g in data.index:
        values_per_group = [col for col_name, col in df_grouped[g]]
        logfc = df_grouped[g].mean().max() / (
            df_grouped[g].mean().mean() - df_grouped[g].mean().max() / 6
        )
        high_group.append(df_grouped[g].mean().idxmax())
        #         var.append(df_grouped[g].mean().var())
        logfcs.append(logfc)
        if test == "kruskal":
            stat, pval = ss.kruskal(*values_per_group)

        elif test == "f_oneway":
            stat, pval = ss.f_oneway(*values_per_group)
        elif test == "t-test":
            pval_old = 10
            stat_old = 0
            for v in values_per_group:
                if v.index[0][0] == df_grouped[g].mean().idxmax():
                    type1 = v
            type2 = []
            for i in values_per_group:
                for j in i:
                    type2.append(j)
            stat, pval = ss.ttest_ind(a=type1, b=type2, equal_var=False)
            if pval < pval_old:
                pval_old = pval
                stat_old = stat

            pval = pval_old
            stat = stat_old
        stats.append(stat)
        pvals.append(pval)
    anova_df["stats"] = stats
    anova_df["pvals"] = pvals
    anova_df["logfc"] = logfcs
    anova_df["subtype"] = high_group

    _r, adj_p, _a, _b = mt.multipletests(anova_df["pvals"], method="fdr_bh")
    anova_df["p_adj"] = adj_p

    # if adj_p is less than threshold, add gene to list
    # sig_genes is a Boolean vector length of genes
    return anova_df


def generate_arc_sig_df(
    genes_df,
    features_name="",
    logfc_name="",
    subtype_name="",
    subtype_dict=None,
    p_adj_name="",
):

    # convert genes dataframe to one shaped like anova_df: index = genes, with columns: 'logfc','subtype','p_adj'

    arc_sig_df = pd.DataFrame(index=sorted(list(set(genes_df[features_name]))))
    p_adjs = []
    logfcs = []  # mean_diff
    subtypes = []

    for gene in arc_sig_df.index:
        _df = genes_df.loc[genes_df[features_name] == gene]
        _gene = _df.sort_values(by=logfc_name, ascending=False).reset_index().loc[0]
        p_adjs.append(_gene[p_adj_name])
        logfcs.append(_gene[logfc_name])
        if type(subtype_dict) == type(dict):
            s = _gene[subtype_name]
            subtypes.append(subtype_dict[s])
        else:
            subtypes.append(_gene[subtype_name])

    arc_sig_df["p_adj"] = p_adjs
    arc_sig_df["logfc"] = logfcs
    arc_sig_df["subtype"] = subtypes

    return arc_sig_df
