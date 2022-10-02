import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
from scvelo.preprocessing.moments import get_connectivities
import seaborn as sns
from sklearn.utils import resample
from scipy.stats import entropy
import random
from scipy.stats import percentileofscore as POS
from statsmodels.stats import multitest


def permutation_enrichment_test(
    adata,
    score,
    arc_label="specialists_ParetoTI_S",
    n_permutations=1000,
    plot=True,
    stat="median",
    verbose=True,
    sample_equally=None,
    correction=None,
    figsize=(3, 2),
    dpi=200,
    save=None,
):
    score_tmp = score.copy()
    if score_tmp.isna().sum().sum() > 0:
        print(
            "Warning: you have NAs in your score matrix. Ignoring NAs for calculation of statistic."
        )
    score_tmp["spec"] = adata.obs[arc_label]
    mean_by_arc = score_tmp.groupby("spec").mean().idxmax(axis=1)
    p_values = pd.DataFrame(
        columns=score.columns, index=adata.obs[arc_label].cat.categories
    )
    for cat in adata.obs[arc_label].cat.categories:
        print(f"Archetype: {cat}")
        closest = list(adata[adata.obs[arc_label] == cat].obs_names)
        far = list(adata[adata.obs[arc_label] != cat].obs_names)
        if verbose:
            print(
                f"Bulk archetype with highest average score at archetype {cat}: {mean_by_arc[cat]}"
            )

        for s in score.columns:
            median_specialists = score.loc[closest, s].median(skipna=True)
            median_non = score.loc[far, s].median(skipna=True)
            mean_specialists = score.loc[closest, s].mean(skipna=True)
            mean_non = score.loc[far, s].mean(skipna=True)
            random_medians = []
            for r in range(n_permutations):
                if type(sample_equally) == type(None):
                    random_cells = random.choices(score.loc[far, s], k=len(closest))
                else:
                    n_sample_each = int(
                        np.floor(
                            len(closest) / len(np.unique(adata.obs[sample_equally]))
                        )
                    )
                    score_tmp[sample_equally] = adata.obs[sample_equally]
                    random_cells = (
                        score_tmp.loc[far]
                        .groupby(sample_equally)
                        .sample(n_sample_each, replace=True)[s]
                    )

                # print(np.median(random.choices(score.loc[far, s], k=len(closest))))
                if stat == "median":
                    random_medians.append(np.nanmedian(random_cells))
                elif stat == "mean":
                    random_medians.append(np.nanmean(random_cells))
            if plot:
                plt.figure(figsize=figsize, dpi=dpi)
                sns.distplot(random_medians, color="orange", rug=True, hist=False)

                if stat == "median":
                    plt.vlines(
                        x=median_specialists,
                        ymin=plt.ylim()[0],
                        ymax=plt.ylim()[1],
                        colors="k",
                    )
                elif stat == "mean":
                    plt.vlines(
                        x=mean_specialists,
                        ymin=plt.ylim()[0],
                        ymax=plt.ylim()[1],
                        colors="k",
                    )

                plt.title(f"{s} enrichment at {cat}")
                if type(save) != type(None):
                    plt.savefig(f"./figures/{cat}_{s}_{save}")
                plt.show()
            # sns.kdeplot(score.loc[closest, s], color = 'red')
            # sns.kdeplot(score.loc[far, s], color = 'blue')
            # plt.show()

            if stat == "median":
                p_val = 1 - POS(random_medians, median_specialists) / 100

            elif stat == "mean":
                p_val = 1 - POS(random_medians, mean_specialists) / 100
            print(f"\t p-value for {s}: {p_val}")
            p_values.loc[cat, s] = p_val

            if verbose:
                print(f"\t Fold change in mean for {s}: {mean_specialists/mean_non}")
    if type(correction) == type(None):
        return p_values
    else:
        p_values["id"] = p_values.index
        p_values_melt = p_values.melt(id_vars="id", value_vars=p_values.columns)
        p_values_melt["corr_p"] = multitest.multipletests(
            p_values_melt["value"], method=correction
        )[1]
        p_values_corrected = p_values_melt.pivot(
            index="id", columns="variable", values="corr_p"
        )
        return p_values_corrected, p_values


def lstsq_scoring(
    adata,
    sig_matrix,
    score_name="_Score",
    col_name="_Col",
    pheno_name="Phenotype_unfiltered",
    groupby="cline",
    log=False,
    expm1=False,
    smooth=True,
    save=True,
    save_name="",
    basis="umap",
    mouse=False,
):
    # score_name is also used for saving the plots
    # if gene signature is based on un-log-transformed data, use default setting log == False and adata.raw
    # else, use adata.X and log-transformed signature
    if log == True:
        sig_matrix = np.log1p(sig_matrix)

    print(sig_matrix.head())

    adata_ = adata.copy()
    if mouse == True:
        sig_matrix.index = [i.capitalize() for i in sig_matrix.index]
    sig_matrix = sig_matrix.loc[
        [i for i in set(sig_matrix.index).intersection(adata_.var_names)]
    ]
    sig_matrix = sig_matrix / np.linalg.norm(sig_matrix, axis=0)

    adata_ = adata_[:, list(sig_matrix.index)]

    if sp.sparse.issparse(adata_.X.transpose()):
        adata_ = adata_.X.transpose().todense()
        print("csc matrix to dense")
    else:
        adata_ = adata_.X.transpose()
    print(sig_matrix.shape, adata_.shape)

    if expm1 == True:
        x, res, rank, s = np.linalg.lstsq(
            np.expm1(np.array(sig_matrix)), np.expm1(adata_)
        )

    else:
        x, res, rank, s = np.linalg.lstsq(np.array(sig_matrix), adata_)

    pheno_score = pd.DataFrame(
        x.transpose(), index=adata.obs_names, columns=sig_matrix.columns
    )
    col_score = pd.DataFrame(index=adata.obs_names)
    for i in sig_matrix:
        adata.obs[str(i) + score_name] = pheno_score[i]
        col = adata.obs[str(i) + score_name]
        #         col /= np.max(np.abs(col))
        ax = scv.pl.scatter(
            adata, color="lightgrey", show=False, alpha=1, size=200, basis=basis
        )
        if save == True:
            if smooth == True:
                scv.pl.scatter(
                    adata,
                    color=col,
                    basis=basis,
                    fontsize=24,
                    size=80,
                    smooth=True,
                    color_map="RdBu_r",
                    colorbar=True,
                    title=i,
                    ax=ax,
                    zorder=2,
                    save=f"{save_name}_{i}{score_name}.pdf",
                )
                c = get_connectivities(adata).dot(col)
            else:
                scv.pl.scatter(
                    adata,
                    color=col,
                    basis=basis,
                    fontsize=24,
                    size=80,
                    color_map="RdBu_r",
                    colorbar=True,
                    title=i,
                    ax=ax,
                    zorder=2,
                    save=f"{save_name}_{i}{score_name}.pdf",
                )
                c = col
        else:
            if smooth == True:
                scv.pl.scatter(
                    adata,
                    color=col,
                    basis=basis,
                    fontsize=24,
                    size=80,
                    smooth=True,
                    color_map="RdBu_r",
                    colorbar=True,
                    title=i,
                    ax=ax,
                    zorder=2,
                )
                c = get_connectivities(adata).dot(col)
            else:
                scv.pl.scatter(
                    adata,
                    color=col,
                    basis=basis,
                    fontsize=24,
                    size=80,
                    color_map="RdBu_r",
                    colorbar=True,
                    title=i,
                    ax=ax,
                    zorder=2,
                )
                c = col
        adata.obs[str(i) + col_name] = c
        col_score[str(i)] = c
        if save == True:
            sc.pl.violin(
                adata,
                groupby=groupby,
                keys=str(i) + col_name,
                save=f"_{i}{score_name}.pdf",
            )
        else:
            sc.pl.violin(adata, groupby=groupby, keys=str(i) + col_name)
            plt.hlines(y=0, xmin=-1, xmax=11)

        plt.close()

    pheno = []
    maxes = []
    for i, r in col_score.iterrows():
        pheno.append(r.idxmax())
        maxes.append(r.max())
    adata.obs[pheno_name] = pheno
    adata.obs["max"] = maxes
    return adata


def lstsq_scoring_background(
    adata, sig_matrix, log=False, expm1=False, smooth=True, mouse=False
):
    # score_name is also used for saving the plots
    # if gene signature is based on un-log-transformed data, use default setting log == False and adata.raw
    # else, use adata.X and log-transformed signature
    adata_shuffled = adata.copy()
    random = []
    try:
        for r in np.asarray(adata.X.todense()):
            random.append(resample(r))
    except AttributeError:
        for r in np.asarray(adata.X):
            random.append(resample(r))
    adata_shuffled.X = np.asarray(random)

    adata_ = adata_shuffled.copy()
    if mouse == True:
        sig_matrix.index = [i.capitalize() for i in sig_matrix.index]
    sig_matrix = sig_matrix.loc[
        [i for i in set(sig_matrix.index).intersection(adata_.var_names)]
    ]
    adata_ = adata_[:, list(sig_matrix.index)]
    if sp.sparse.issparse(adata_.X.transpose()):
        adata_ = adata_.X.transpose().todense()
        print("csc matrix to dense")
    else:
        adata_ = adata_.X.transpose()
    if log == True:
        x, res, rank, s = np.linalg.lstsq(np.log1p(np.array(sig_matrix)), adata_)

    elif expm1 == True:
        x, res, rank, s = np.linalg.lstsq(
            np.expm1(np.array(sig_matrix)), np.expm1(adata_)
        )

    else:
        x, res, rank, s = np.linalg.lstsq(np.array(sig_matrix), adata_)

    pheno_score = pd.DataFrame(
        x.transpose(), index=adata.obs_names, columns=sig_matrix.columns
    )
    col_score = pd.DataFrame(index=adata.obs_names)
    if type(get_connectivities(adata)) == type(None):
        sc.pp.neighbors(adata)
    for i in sig_matrix:
        adata.obs[str(i) + "score"] = pheno_score[i]
        col = adata.obs[str(i) + "score"]
        #         col /= np.max(np.abs(col))
        if smooth == True:
            c = get_connectivities(adata).dot(col)
        else:
            c = col

        adata.obs[str(i) + "col"] = c
        col_score[str(i)] = c

    pheno = []
    maxes = []
    for i, r in col_score.iterrows():
        pheno.append(r.idxmax())
        maxes.append(r.max())

    del adata_shuffled
    del adata_
    return col_score


def signature_scoring(
    adata,
    sig_matrix,
    method="lstsq",
    score_name="_Score",
    col_name="_Col",
    pheno_name="Phenotype_unfiltered",
    groupby="cline",
    smooth=True,
    save=False,
    basis="umap",
    num_bg=20,
    mouse=False,
    threshold=0.05,
    correction=None,
    log=False,
    expm1=False,
    dt=1,
):
    print("Scoring individual cells...")
    if method == "lstsq":
        adata_scores = lstsq_scoring(
            adata,
            sig_matrix,
            score_name=score_name,
            col_name=col_name,
            mouse=mouse,
            basis=basis,
            groupby=groupby,
            pheno_name=pheno_name,
            smooth=smooth,
            save=save,
            log=log,
            expm1=expm1,
        )

        adata_t1 = adata.copy()

        adata_t1.X = adata.X + np.nan_to_num(adata.layers["velocity"]) * dt
        adata_t1_scores = lstsq_scoring(
            adata_t1,
            sig_matrix,
            score_name=f"{score_name}_t1",
            col_name=col_name,
            mouse=mouse,
            basis=basis,
            groupby=groupby,
            pheno_name=pheno_name,
            smooth=smooth,
            save=save,
            log=log,
            expm1=expm1,
        )

        for i in sig_matrix:
            adata.obs[str(i) + f"{score_name}_t1"] = adata_t1.obs[
                str(i) + f"{score_name}_t1"
            ]

        # Plotting
        custom_palette = [
            "#fc8d62",
            "#66c2a5",
            "#FFD43B",
            "#8da0cb",
            "#e78ac3",
            "#a6d854",
        ]
        plt.figure(figsize=[12, 8])
        sc.pl.scatter(
            adata_scores, color=pheno_name, palette=custom_palette, basis=basis
        )
        plt.show()

        # generate background
        print("Generating background...")
        col_score_shuffled = pd.DataFrame()
        for rand in range(num_bg):
            print(rand)
            col_score = lstsq_scoring_background(
                adata, sig_matrix, mouse=mouse, smooth=smooth, log=log, expm1=expm1
            )
            col_score_shuffled = col_score_shuffled.append(col_score, ignore_index=True)

        thresholds = dict()
        means = dict()
        sds = dict()
        for i in col_score_shuffled:
            means[i] = np.mean(col_score[i])
            sds[i] = np.std(col_score[i])
            if type(correction) == type(None):
                thresholds[i] = np.percentile(col_score[i], 100 - threshold * 100)

            elif correction == "Bonferroni":
                thresholds[i] = np.percentile(
                    col_score[i], 100 - (threshold * 100 / len(col_score[i]))
                )  # Bonferroni correction
            elif correction == "Positive":
                thresholds[i] = means[i]

            sns.kdeplot(col_score_shuffled[i])
        plt.show()

    print("Correcting individual cell scores...")
    plt.rcParams["figure.figsize"] = [8, 6]

    col_score_filtered = pd.DataFrame(index=adata.obs_names)

    for i in sig_matrix:
        # sig score is any score above threshold
        # z score is standardized score for any over threshold
        # for generalist/specialist distinction, I want to standardize each score and then choose the positive ones
        adata.obs[f"{i}_Sig{score_name}"] = adata.obs[f"{i}{col_name}"] * (
            adata.obs[f"{i}{col_name}"] > thresholds[i]
        )
        adata.obs[f"{i}_Z{score_name}"] = [
            (x - means[i]) / sds[i] for x in adata.obs[f"{i}{col_name}"]
        ] * (adata.obs[f"{i}{col_name}"] > thresholds[i])

        ax = scv.pl.scatter(
            adata, color="lightgrey", show=False, alpha=1, size=200, basis=basis
        )
        scv.pl.scatter(
            adata,
            color=f"{i}_Z{score_name}",
            basis=basis,
            fontsize=24,
            size=80,
            vmin=0,
            smooth=smooth,
            color_map="RdBu_r",
            colorbar=True,
            title=i,
            ax=ax,
            zorder=2,
        )
        col = adata.obs[f"{i}_Z{score_name}"]
        #     c = get_connectivities(adata).dot(col)
        col_score_filtered[i] = col
        sc.pl.violin(adata, groupby=groupby, keys=str(i) + f"_Z{score_name}")
        plt.close()
        plt.figure()
        ax = sc.pl.violin(adata, groupby=groupby, keys=str(i) + col_name, show=False)
        ax.hlines(y=thresholds[i], xmin=-0.5, xmax=7.5, linestyle="--")
        plt.xticks(rotation=90)
        plt.show()
        plt.close()


def subtype_cells(
    adata,
    sig_matrix,
    basis="umap",
    score_name="Z_Score",
    plot=True,
    pheno_name="Phenotype_filtered",
    mix_filter=1,
    nonnegative=True,
):
    pheno = []
    maxs = []
    score_entropy = []
    score_df = pd.DataFrame(index=adata.obs_names)
    for i in sig_matrix:
        if nonnegative:
            adata.obs[f"{i}_{score_name}"][adata.obs[f"{i}_{score_name}"] < 0] = 0
            if f"{i}_{score_name}_t1" in adata.obs.columns:
                adata.obs[f"{i}_{score_name}_t1"][
                    adata.obs[f"{i}_{score_name}_t1"] < 0
                ] = 0

        score_df[i] = adata.obs[f"{i}_{score_name}"]

    for i, r in score_df.iterrows():
        if r.sum() == 0:
            pheno.append("None")
            score_entropy.append(None)
        else:
            if (r / r.sum()).max() < mix_filter:
                pheno.append("Mixed")
            else:
                pheno.append(r.idxmax())
            score_entropy.append(entropy(r.T / r.sum()))

        maxs.append(r.max())

    adata.obs[pheno_name] = pheno
    adata.obs["max_filtered"] = maxs
    adata.obs["score_entropy"] = score_entropy
    custom_palette = [
        "grey",
        "lightgrey",
        "#fc8d62",
        "#66c2a5",
        "#FFD43B",
        "#8da0cb",
        "#e78ac3",
    ]
    if plot:
        plt.figure(figsize=[10, 12])
        ax = scv.pl.scatter(
            adata,
            color="lightgrey",
            show=False,
            alpha=1,
            size=40,
            basis=basis,
            figsize=(8, 6),
        )

        scv.pl.scatter(
            adata,
            color=pheno_name,
            palette=custom_palette,
            basis=basis,
            zorder=2,
            ax=ax,
            size=40,
            legend_loc="on right",
        )
        plt.show()
