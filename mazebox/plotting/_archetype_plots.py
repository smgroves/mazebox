import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import seaborn as sns
from collections import Counter
from sklearn.manifold import LocallyLinearEmbedding
import os.path as op


def archetype_diagrams(
    adata,
    sig_matrix,
    color_dict,
    groupby,
    color="Phenotype_filtered",
    score_name="_Z_Score",
    multiplier=1000,
    mean_only=False,
    order=None,
    figsize=(4, 4),
    arrows=False,
    sizes=None,
    alpha=0.3,
    s=30,
    norm="max",
    scale=1,
    cmap="viridis",
    vmin=-1,
    vmax=1,
    grid=True,
    num_steps=50,
    density=False,
):
    # cp = custom_palette[2:]
    # color_dict = {'SCLC-Y': cp[4], 'SCLC-A': cp[0], 'SCLC-A2': cp[1], 'SCLC-N': cp[2], 'SCLC-P': cp[3],
    #               'Mixed': 'darkgray', 'None': 'white'}
    n_types = len(sig_matrix.columns)
    subtypes = sig_matrix.columns
    if groupby != "None":
        for g in adata.obs[groupby].cat.categories:
            print(g)
            _adata = adata[adata.obs[groupby] == g]
            n_samples = len(_adata.obs_names)

            if arrows:
                _archetype_diagram_with_arrows(
                    _adata,
                    score_name,
                    subtypes,
                    color,
                    figsize,
                    n_types,
                    color_dict,
                    order,
                    multiplier,
                    g=g,
                    mean_only=mean_only,
                    cmap=cmap,
                    groupby=groupby,
                    alpha=alpha,
                    s=s,
                    norm=norm,
                    sizes=sizes,
                    grid=grid,
                    num_steps=num_steps,
                )
            else:
                _archetype_diagram_no_arrows(
                    _adata,
                    score_name,
                    subtypes,
                    color,
                    figsize,
                    n_types,
                    order,
                    n_samples,
                    multiplier,
                    g=g,
                    sizes=sizes,
                    alpha=alpha,
                    s=s,
                    color_dict=color_dict,
                    norm=norm,
                    cmap=cmap,
                    vmax=vmax,
                    vmin=vmin,
                    density=density,
                )
    else:
        _adata = adata
        n_samples = len(_adata.obs_names)

        if arrows:
            _archetype_diagram_with_arrows(
                _adata,
                score_name,
                subtypes,
                color,
                figsize,
                n_types,
                color_dict,
                order,
                multiplier,
                scale=scale,
                mean_only=mean_only,
                cmap=cmap,
                groupby=groupby,
                alpha=alpha,
                s=s,
                norm=norm,
                sizes=sizes,
                grid=grid,
                num_steps=num_steps,
            )
        else:
            _archetype_diagram_no_arrows(
                _adata,
                score_name,
                subtypes,
                color,
                figsize,
                n_types,
                order,
                n_samples,
                multiplier,
                sizes=sizes,
                alpha=alpha,
                s=s,
                color_dict=color_dict,
                norm=norm,
                cmap=cmap,
                vmax=vmax,
                vmin=vmin,
                density=density,
            )


def _archetype_diagram_with_arrows(
    _adata,
    score_name,
    subtypes,
    color,
    figsize,
    n_types,
    color_dict,
    order,
    multiplier,
    g="All",
    cmap="viridis",
    norm="max",
    alpha=0.3,
    s=3,
    width=0.0005,
    sizes=3,
    grid=True,
    num_steps=50,
    scale=1,
    mean_only=False,
    groupby="timepoint",
):
    # for arrows, we need to subtype a second timepoint based off of velocity.

    X = _adata.obs[[f"{x}{score_name}" for x in subtypes]]
    X_t1 = _adata.obs[[f"{x}{score_name}_t1" for x in subtypes]]
    top = X.sum().sort_values(ascending=False)[:3]  # return a list of n largest element
    print(top.index)

    # normalize X to sum to 1 (makes sure that the data fits within the archetype diagram with I matrix = vertices
    X_norm = pd.DataFrame(columns=X.columns)
    X_t1_norm = pd.DataFrame(columns=X_t1.columns)

    if norm == "scale":
        for i, r in X.iterrows():
            if r.sum() == 0:
                pass
            # X_norm = X_norm.append(r)
            # X_t1_norm = X_t1_norm.append(X_t1.loc[i])

            else:
                X_norm = X_norm.append(r / r.sum())
                X_t1_norm = X_t1_norm.append(X_t1.loc[i] / X_t1.loc[i].sum())

    elif norm == "max":
        max = X.max().max()
        X_norm = X / max
        X_t1_norm = X_t1 / max
    elif norm == "None":
        X_norm = X
        X_t1_norm = X_t1

    X = X_norm.fillna(0)
    X_t1 = X_t1_norm.fillna(0)
    # Order of sig_matrix, or specified order, for plotting vertices
    if type(order) != type(None):
        X = X[[f"{x}{score_name}" for x in order]]
        X_t1 = X_t1[[f"{x}{score_name}_t1" for x in order]]
        subtypes = order
    else:
        order = subtypes

    ident = np.identity(n_types)

    for i, x in enumerate(subtypes):
        X = X.append(pd.DataFrame(np.array(ident[i]), index=X.columns, columns=[x]).T)

    embedding = LocallyLinearEmbedding(
        n_components=2, n_neighbors=4, method="modified", eigen_solver="dense"
    )
    X_transformed = embedding.fit_transform(
        X[-n_types:]
    )  # fit and transform vertcies (ident[i])
    X_transformed = X_transformed.T

    # transform data
    X_df_data = embedding.transform(X[:-n_types])
    X_df_data = X_df_data.T

    X_t1 = embedding.transform(X_t1)
    X_t1 = X_t1.T

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot()
    colors = _adata[X_norm.index, :].obs[color].values
    if type(colors[0]) == str:
        c = [color_dict[i] for i in colors]
        data = pd.DataFrame(index=_adata.obs_names)
        data["X"] = pd.Series(X_df_data[0], index=_adata.obs_names)
        data["Y"] = pd.Series(X_df_data[1], index=_adata.obs_names)
        data["C"] = pd.Series(c, index=_adata.obs_names)
        data = data.sort_values(by="C", ascending=False)

        if mean_only:
            for g in _adata.obs[groupby].cat.categories:
                sub = _adata[_adata.obs[groupby] == g].obs_names
                _d = data.loc[sub]
                plt.scatter(
                    _d["X"].mean(), _d["Y"].mean(), c="k", zorder=3, alpha=alpha, s=s
                )

        else:
            plt.scatter(data["X"], data["Y"], c=data["C"], zorder=3, alpha=alpha, s=s)

    else:
        c = colors
        if mean_only:
            plt.scatter(
                X_df_data[0].mean(),
                X_df_data[1].mean(),
                c=c.mean(),
                zorder=3,
                alpha=alpha,
                s=s,
            )

        else:
            plt.scatter(
                X_df_data[0], X_df_data[1], c=c, zorder=3, alpha=alpha, s=s, cmap=cmap
            )
    plt.scatter(
        X_transformed[0, -n_types:],
        X_transformed[1, -n_types:],
        c=[color_dict[i] for i in order],
        zorder=2,
        s=sizes,
    )

    # add arrows from each point in X_df_data to X_df_t1
    # format for plt.arrow() is (x,y,dx,dy)

    X_df_t1 = pd.DataFrame(X_t1.T)

    if grid:
        grid_df = pd.DataFrame()
        grid_df["u"] = X_df_t1[0] - X_df_data[0]
        grid_df["v"] = X_df_t1[1] - X_df_data[1]
        grid_df["x"] = X_df_data[0]
        grid_df["y"] = X_df_data[1]

        x = np.linspace(grid_df["x"].min(), grid_df["x"].max(), num=num_steps)
        y = np.linspace(grid_df["y"].min(), grid_df["y"].max(), num=num_steps)

        X, Y = np.meshgrid(x, y)

        grid_df["binned_x"] = pd.cut(grid_df["x"], bins=x)
        grid_df["binned_y"] = pd.cut(grid_df["y"], bins=y)

        binned_grid = grid_df.groupby(["binned_x", "binned_y"])
        binned_grid = binned_grid.mean()
        binned_grid = binned_grid.dropna()
        binned_grid = binned_grid.reset_index()

        binned_grid["binned_x"] = binned_grid["binned_x"].apply(lambda x: x.mid)
        binned_grid["binned_y"] = binned_grid["binned_y"].apply(lambda x: x.mid)

        plt.quiver(
            binned_grid["binned_x"].values,
            binned_grid["binned_y"].values,
            binned_grid["u"].values,
            binned_grid["v"].values,
            angles="xy",
            scale_units="xy",
            zorder=4,
            headlength=2,
            headaxislength=2,
            headwidth=3,
            scale=scale,
            alpha=alpha,
            width=0.003,
            linestyle="solid",
        )

    else:
        dx = X_df_t1[0] - X_df_data[0]
        dy = X_df_t1[1] - X_df_data[1]
        if mean_only:
            dx.index = _adata.obs_names
            dy.index = _adata.obs_names

            for g in _adata.obs[groupby].cat.categories:
                sub = _adata[_adata.obs[groupby] == g].obs_names
                _d = data.loc[sub]
                plt.arrow(
                    _d["X"].mean(),
                    _d["Y"].mean(),
                    dx.loc[sub].mean() * multiplier,
                    dy.loc[sub].mean() * multiplier,
                    fc="black",
                    ec="black",
                    alpha=alpha,
                    width=width,
                )
        else:
            for i, r in X_df_t1.iterrows():
                # print(X_df_data.T[i][0], X_df_data.T[i][1])
                # ax.annotate('', (X_df_data.T[i][0], X_df_data.T[i][1]),(r[0]*multiplier, r[1]*multiplier))

                plt.arrow(
                    X_df_data.T[i][0],
                    X_df_data.T[i][1],
                    dx[i] * multiplier,
                    dy[i] * multiplier,
                    fc="lightgrey",
                    ec="lightgrey",
                    alpha=alpha,
                    width=width,
                )

    # plt.legend()
    plt.title(g)
    plt.axis("tight")
    plt.show()
    plt.close()


def _archetype_diagram_no_arrows(
    _adata,
    score_name,
    subtypes,
    color,
    figsize,
    n_types,
    order,
    n_samples,
    multiplier,
    g="All",
    sizes=None,
    alpha=0.3,
    s=30,
    color_dict=None,
    norm="scale",
    cmap="viridis",
    vmin=-1,
    vmax=1,
    density=False,
):
    # X = subtype scores for each subtype (pd dataframe)
    X = _adata.obs[[f"{x}{score_name}" for x in subtypes]]
    top = X.sum().sort_values(ascending=False)[:3]  # return a list of n largest element
    print(top.index)

    # normalize X to sum to 1 (makes sure that the data fits within the archetype diagram with I matrix = vertices
    X_norm = pd.DataFrame(columns=X.columns)
    if norm == "scale":
        for i, r in X.iterrows():
            if r.sum() == 0:
                X_norm = X_norm.append(r)
            else:
                X_norm = X_norm.append(r / r.sum())
    elif norm == "max":
        max = X.max().max()
        X_norm = X / max
    elif norm == "None":
        X_norm = X

    X = X_norm.fillna(0)
    if type(order) != type(None):
        X = X[[f"{x}{score_name}" for x in order]]
        subtypes = order
    else:
        order = subtypes

    ident = np.identity(n_types)
    for i, x in enumerate(subtypes):
        X = X.append(pd.DataFrame(np.array(ident[i]), index=X.columns, columns=[x]).T)

    embedding = LocallyLinearEmbedding(
        n_components=2, n_neighbors=4, method="modified", eigen_solver="dense"
    )
    X_transformed = embedding.fit_transform(X[-n_types:])
    X_transformed = X_transformed.T

    X_df_data = embedding.transform(X[:-n_types])
    X_df_data = X_df_data.T

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot()
    colors = _adata.obs[color].values
    if type(colors[0]) == str:
        c = [color_dict[i] for i in colors]
        data = pd.DataFrame(index=_adata.obs_names)
        data["X"] = pd.Series(X_df_data[0], index=_adata.obs_names)
        data["Y"] = pd.Series(X_df_data[1], index=_adata.obs_names)
        data["C"] = pd.Series(c, index=_adata.obs_names)
        data = data.sort_values(by="C", ascending=False)

        plt.scatter(data["X"], data["Y"], c=data["C"], zorder=3, alpha=alpha, s=s)
        # plt.legend()
    else:
        c = colors
        plt.scatter(
            X_df_data[0],
            X_df_data[1],
            c=c,
            zorder=3,
            alpha=alpha,
            s=s,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar()

    if type(sizes) == int:
        plt.scatter(
            X_transformed[0, -n_types:],
            X_transformed[1, -n_types:],
            c="black",
            zorder=2,
            s=sizes,
        )
    else:
        sizes = [Counter(colors)[i] * multiplier / n_samples for i in order]
        print(sizes)
        plt.scatter(
            X_transformed[0, -n_types:],
            X_transformed[1, -n_types:],
            c=[color_dict[i] for i in order],
            zorder=2,
            s=sizes,
        )
    plt.title(g)
    plt.axis("tight")
    plt.show()
    plt.close()
    if density:
        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot()
        colors = _adata.obs[color].values
        if type(colors[0]) == str:
            c = [color_dict[i] for i in colors]
        else:
            c = colors
        sns.kdeplot(X_df_data[0], X_df_data[1], cmap="Reds", shade=True, bw=0.15)
        if type(sizes) == int:
            plt.scatter(
                X_transformed[0, -n_types:],
                X_transformed[1, -n_types:],
                c="k",
                zorder=2,
                s=sizes,
            )
        else:
            sizes = [Counter(colors)[i] * multiplier / n_samples for i in order]
            print(sizes)
            plt.scatter(
                X_transformed[0, -n_types:],
                X_transformed[1, -n_types:],
                c=[color_dict[i] for i in order],
                zorder=2,
                s=sizes,
            )
        plt.title(g)
        plt.axis("tight")
        plt.show()
        plt.close()


custom_palette = [
    "grey",
    "lightgrey",
    "#fc8d62",
    "#66c2a5",
    "#FFD43B",
    "#8da0cb",
    "#e78ac3",
]


def subtype_bar(
    adata,
    groupby,
    fname,
    pheno_name="Phenotype_filtered",
    custom_palette=custom_palette,
):
    import matplotlib.pyplot as plt

    plt.rcParams["figure.figsize"] = [8, 4]
    (
        (
            pd.crosstab(adata.obs[groupby], adata.obs[pheno_name]).T
            / pd.crosstab(adata.obs[groupby], adata.obs[pheno_name]).sum(axis=1)
        )
        * 100
    ).T.plot(kind="bar", stacked=True, linewidth=0, color=custom_palette)
    plt.savefig(f"./figures/barplot_{fname}.pdf")


def archetype_umap():
    print("hi")


def twodimplot_tumor(
    data,
    veldata,
    adata,
    gene_sig,
    celltype1,
    celltype2,
    save=False,
    onecluster=False,
    cluster=None,
    tumor_clust_name="louvain",
    axes=False,
    meanonly=False,
    dir_prefix=None,
    palette=sns.color_palette("Paired"),
):
    if type(dir_prefix) == type(None) and save != False:
        print("must define out directory using argument 'dir_prefix' to save figure")

    # data = tumor data to plot
    # corresponding velocity data
    # celltype1 = reference cell types to plot as axes
    # celltype2 = y axis
    # onecluster = only plot one of the louvain clusters from the tumor
    # cluster = if onecluster == True, this is the cluster to plot
    clust_names = list(gene_sig.columns)

    if type(celltype1) == str:
        clust_dict = {}
        for i, c in enumerate(clust_names):
            clust_dict[c] = i
        celltype1 = clust_dict[celltype1]
    if type(celltype2) == str:
        clust_dict = {}
        for i, c in enumerate(clust_names):
            clust_dict[c] = i
        celltype2 = clust_dict[celltype2]

    tumor_clust_names = sorted(list(set(adata.obs[tumor_clust_name])))
    if tumor_clust_name != "louvain":
        tumor_clust_dict = {}
        for i, c in enumerate(tumor_clust_names):
            tumor_clust_dict[c] = i
        adata.obs[tumor_clust_name] = [
            tumor_clust_dict[i] for i in list(adata.obs[tumor_clust_name].values)
        ]

    customPalette = palette
    # add data points
    if onecluster == False:
        if meanonly == True:
            plt.figure(figsize=(20, 20), dpi=300)

            plt.axhline(y=0, color="gray", alpha=0.5, linestyle="--")
            plt.axvline(x=0, color="gray", alpha=0.5, linestyle="--")
            if axes != False:
                plt.xlim(axes[0])
                plt.ylim(axes[1])
            else:
                plt.xlim(
                    [
                        np.array([data[celltype1].min(), data[celltype2].min()]).min(),
                        np.array([data[celltype1].max(), data[celltype2].max()]).max(),
                    ]
                )
                plt.ylim(
                    [
                        np.array([data[celltype1].min(), data[celltype2].min()]).min(),
                        np.array([data[celltype1].max(), data[celltype2].max()]).max(),
                    ]
                )

            labels = pd.DataFrame(
                adata.obs[tumor_clust_name].values, columns=["label"], index=data.index
            )
            for label in list(set(adata.obs[tumor_clust_name])):
                plt.annotate(
                    label,
                    data.loc[labels["label"] == label, [celltype1, celltype2]].mean(),
                    horizontalalignment="center",
                    verticalalignment="center",
                    size=18,
                    alpha=1,
                    color=customPalette[int(label)],
                )
                plt.annotate(
                    "",
                    xy=(
                        data.loc[labels["label"] == label, celltype1].mean()
                        + veldata.loc[labels["label"] == label, celltype1].mean(),
                        data.loc[labels["label"] == label, celltype2].mean()
                        + veldata.loc[labels["label"] == label, celltype2].mean(),
                    ),
                    xytext=(
                        data.loc[
                            labels["label"] == label, [celltype1, celltype2]
                        ].mean()
                    ),
                    arrowprops=dict(
                        facecolor="black",
                        edgecolor="black",
                        width=1,
                        headlength=3,
                        headwidth=3,
                        alpha=1,
                    ),
                )
            plt.title([clust_names[celltype1], " vs. ", clust_names[celltype2]])
            plt.tight_layout()
            if save == True:
                plt.savefig(
                    op.join(
                        dir_prefix,
                        f"{clust_names[celltype1]} vs. {clust_names[celltype2]}.png",
                    )
                )
                plt.close()
            elif type(save) == str:
                plt.savefig(op.join(dir_prefix, f"{save}.png"))
                plt.close()
            else:
                plt.show()
                plt.close()
        else:
            plt.figure(figsize=(20, 20), dpi=200)
            plt.scatter(
                x=data[celltype1],
                y=data[celltype2],
                s=10,
                color=[customPalette[int(i)] for i in adata.obs[tumor_clust_name]],
                alpha=1,
            )
            if type(veldata) != type(None):
                for x, y in zip(
                    range(len(data[celltype1])), range(len(data[celltype1]))
                ):
                    plt.annotate(
                        "",
                        xy=(
                            data[celltype1][x] + veldata[celltype1][x],
                            data[celltype2][y] + veldata[celltype2][y],
                        ),
                        xytext=(data[celltype1][x], data[celltype2][y]),
                        arrowprops=dict(
                            facecolor="black",
                            width=1,
                            headlength=3,
                            headwidth=3,
                            alpha=0.2,
                        ),
                    )
            plt.axhline(y=0, color="gray", alpha=0.5, linestyle="--")
            plt.axvline(x=0, color="gray", alpha=0.5, linestyle="--")
            if axes != False:
                plt.xlim(axes[0])
                plt.ylim(axes[1])
            labels = pd.DataFrame(
                adata.obs[tumor_clust_name].values, columns=["label"], index=data.index
            )
            for label in list(set(adata.obs[tumor_clust_name])):
                plt.annotate(
                    label,
                    data.loc[labels["label"] == label, [celltype1, celltype2]].mean(),
                    horizontalalignment="center",
                    verticalalignment="center",
                    size=12,
                    alpha=0.5,
                    color="black",
                    backgroundcolor=customPalette[int(label)],
                )
            plt.title([clust_names[celltype1], " vs. ", clust_names[celltype2]])
            if save == True:
                plt.savefig(
                    op.join(
                        dir_prefix,
                        f"{clust_names[celltype1]} vs. {clust_names[celltype2]}.png",
                    )
                )
                plt.close()
            elif type(save) == str:
                plt.savefig(op.join(dir_prefix, f"{save}.png"))
                plt.close()
            else:
                plt.show()
                plt.close()
    else:
        if cluster == None:
            print(
                'You must specify a cluster from the test data for onecluster == True. Use argument "cluster."'
            )
        else:
            plt.figure(figsize=(20, 20))

            list_c = (adata.obs[tumor_clust_name] == cluster).values
            print(len(data.loc[list_c, celltype1]))
            plt.scatter(
                x=data.loc[list_c, celltype1],
                y=data.loc[list_c, celltype2],
                s=10,
                alpha=1,
            )
            plt.axhline(y=0, color="gray", alpha=0.5, linestyle="--")
            plt.axvline(x=0, color="gray", alpha=0.5, linestyle="--")
            for x, y in zip(
                data.loc[list_c, celltype1].index.values,
                data.loc[list_c, celltype1].index.values,
            ):
                if veldata != None:
                    plt.annotate(
                        "",
                        xy=(
                            data.loc[list_c, celltype1][x]
                            + veldata.loc[list_c, celltype1][x],
                            data.loc[list_c, celltype2][y]
                            + veldata.loc[list_c, celltype2][y],
                        ),
                        xytext=(
                            data.loc[list_c, celltype1][x],
                            data.loc[list_c, celltype2][y],
                        ),
                        arrowprops=dict(
                            facecolor="black",
                            width=1,
                            headlength=5,
                            headwidth=5,
                            alpha=0.2,
                        ),
                    )

            plt.title([clust_names[celltype1], " vs. ", clust_names[celltype2]])
            if save == True:
                plt.savefig(
                    op.join(
                        dir_prefix,
                        f"{clust_names[celltype1]} vs. {clust_names[celltype2]}.png",
                    )
                )
                plt.close()
            elif type(save) == str:
                plt.savefig(op.join(dir_prefix, f"{save}.png"))
                plt.close()
            else:
                plt.show()
                plt.close()


def archetype_score_boxplots(
    adata,
    groupby,
    fname,
    cp=["#fc8d62", "#66c2a5", "#FFD43B", "#8da0cb", "#e78ac3"],
    subtypes=["SCLC-A", "SCLC-A2", "SCLC-N", "SCLC-P", "SCLC-Y"],
):

    for x, i in enumerate(subtypes):
        adata.obs[f"{i}_Score_pos"] = adata.obs[f"{i}_Score"] * (
            adata.obs[f"{i}_Score"] > 0
        )
        plt.figure(figsize=(3, 3))
        sns.boxplot(data=adata.obs, x=groupby, y=f"{i}_Score_pos", color=cp[x])
        plt.savefig(f"./figures/{i}_{fname}.pdf")
