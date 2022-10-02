from sklearn import preprocessing as pp
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns

#    bulk = pd.read_csv('../../../data/bulk-rna-seq/SCLC_combined_Minna_CCLE_batch_corrected_wo_lowgenes.csv', header = 0, index_col = 0)
# arc = pd.read_csv(
#     "../../../data/bulk-rna-seq/arc_gene_space_no_lowgenes-10-21-20.csv",
#     header=None,
#     index_col=None,
# )
# arc_genes = pd.read_csv(
#     "../../../data/bulk-rna-seq/geneNames_wo_lowgenes.csv",
#     header=None,
#     index_col=None,
# )
def fit_bulk_pca(
    adata_var_names,
    bulk,
    arc,
    arc_genes,
    species="human",
    calendar_genes=[
        "MARC1",
        "MARC2",
        "MARCH1",
        "MARCH10",
        "MARCH11",
        "MARCH2",
        "MARCH3",
        "MARCH4",
        "MARCH5",
        "MARCH6",
        "MARCH7",
        "MARCH8",
        "MARCH9",
        "SEPT10",
        "SEPT11",
        "SEPT12",
        "SEPT14",
        "SEPT1",
        "SEPT2",
        "SEPT3",
        "SEPT4",
        "SEPT5",
        "SEPT6",
        "SEPT7",
        "SEPT8",
        "SEPT9",
    ],
):
    bulk = bulk.loc[[i not in calendar_genes for i in bulk.index]]
    if species == "mouse":
        bulk.index = [i.capitalize() for i in bulk.index]
    shared_genes = sorted(list(set(adata_var_names).intersection(bulk.index)))

    bulk = bulk.loc[shared_genes]
    bulk = pd.DataFrame(pp.scale(bulk), columns=bulk.columns, index=bulk.index)
    # bulk = bulk / np.linalg.norm(bulk, axis=0)

    pca = PCA(n_components=20)
    data_pca = pca.fit_transform(bulk.T)
    pca_df = pd.DataFrame(data_pca)

    arc.columns = arc_genes[0]
    arc = arc.T
    arc.columns = ["SCLC-Y", "SCLC-P", "SCLC-N", "SCLC-A2", "SCLC-A"]
    if species == "mouse":
        arc.index = [i.capitalize() for i in arc.index]
    arc = arc.loc[shared_genes]
    arc = pd.DataFrame(pp.scale(arc), columns=arc.columns, index=arc.index)

    # arc = arc / np.linalg.norm(arc, axis=0)

    arc_pca = pca.transform(arc.T)
    arc_pca_df = pd.DataFrame(arc_pca, index=arc.columns)
    arc_pca_df["color"] = arc.columns

    return pca, pca_df, arc_pca_df, shared_genes


def transform_data(adata, pca, shared_genes, layer="X", species="human"):
    if species == "mouse":
        shared_genes = [i.capitalize() for i in shared_genes]
    tmp = adata[:, shared_genes].copy()
    if layer == "X":
        adata_df = pd.DataFrame(tmp.X, index=tmp.obs_names, columns=tmp.var_names)
    else:
        adata_df = pd.DataFrame(
            tmp.layers[layer].todense(), index=tmp.obs_names, columns=tmp.var_names
        )

    adata_df = adata_df.T
    adata_df = pd.DataFrame(
        pp.scale(adata_df), columns=adata_df.columns, index=adata_df.index
    )

    # adata_df = adata_df / np.linalg.norm(adata_df, axis=0)

    adata_pca = pca.transform(adata_df.T)
    adata_pca_df = pd.DataFrame(adata_pca, index=adata_df.columns)
    tot_var_full = adata_df.T.var().sum()
    varexpl = adata_pca_df.var()  # var explained by bulk PCA
    return adata_df, adata_pca_df, {"tot_var_full": tot_var_full, "varexpl": varexpl}


def EV_shuffled_fit(
    adata_var_names,
    adata_df,
    bulk,
    n_components=20,
    n_shuffles=10,
    species="human",
    calendar_genes=[
        "MARC1",
        "MARC2",
        "MARCH1",
        "MARCH10",
        "MARCH11",
        "MARCH2",
        "MARCH3",
        "MARCH4",
        "MARCH5",
        "MARCH6",
        "MARCH7",
        "MARCH8",
        "MARCH9",
        "SEPT10",
        "SEPT11",
        "SEPT12",
        "SEPT14",
        "SEPT1",
        "SEPT2",
        "SEPT3",
        "SEPT4",
        "SEPT5",
        "SEPT6",
        "SEPT7",
        "SEPT8",
        "SEPT9",
    ],
):
    bulk = bulk.loc[[i not in calendar_genes for i in bulk.index]]
    if species == "mouse":
        bulk.index = [i.capitalize() for i in bulk.index]
    shared_genes = sorted(list(set(adata_var_names).intersection(bulk.index)))
    bulk = bulk.loc[shared_genes]
    bulk = pd.DataFrame(pp.scale(bulk), columns=bulk.columns, index=bulk.index)
    ev = []
    for i in range(n_shuffles):
        ran = shuffle(bulk)
        ran.index = bulk.index
        ran_pca = PCA(n_components=n_components)
        ran_data_pca = ran_pca.fit_transform(ran.T)
        ran_pca_df = pd.DataFrame(ran_data_pca)
        adata_pca_rand = ran_pca.transform(adata_df.T)
        adata_pca_rand_df = pd.DataFrame(adata_pca_rand, index=adata_df.columns)
        ev.append(list(adata_pca_rand_df.var()))

    return ev


def plot_EV(adata, bulk, arc, arc_genes, species="human", groupby="timepoint"):
    pca, pca_df, arc_pca_df, shared_genes = fit_bulk_pca(
        adata.var_names, bulk, arc, arc_genes, species=species
    )
    adata_df, adata_pca_df, adata_var_dict = transform_data(
        adata, pca, shared_genes, species=species
    )
    adata_var_dict["shuffled_fit_EV"] = EV_shuffled_fit(
        adata.var_names, adata_df, bulk, species=species
    )
    plt.figure(figsize=(5, 5))
    plt.plot(100 * adata_var_dict["varexpl"] / adata_var_dict["tot_var_full"])
    for i in range(10):
        plt.plot(
            [
                100 * x
                for x in adata_var_dict["shuffled_fit_EV"][i]
                / adata_var_dict["tot_var_full"]
            ],
            c="grey",
        )
    plt.figure(figsize=(12, 10))
    sns.scatterplot(-pca_df[0], pca_df[1])
    sns.scatterplot(
        -adata_pca_df[0], adata_pca_df[1], hue=adata.obs[groupby], alpha=0.5
    )
    sns.scatterplot(-arc_pca_df[0], arc_pca_df[1], color="k", s=200)

    plt.title("Explained Variance for Single Cell Data in PCA on Bulk Data")
    plt.ylabel("Percent Explained Variance")
    plt.xlabel("PC")
    plt.show()

    # ev_df for seaborn
    flat_list = [
        item for sublist in adata_var_dict["shuffled_fit_EV"] for item in sublist
    ]
    ev_df = pd.DataFrame(10 * list(range(20)), columns=["PC"])
    ev_df["ev"] = [100 * i / adata_var_dict["tot_var_full"] for i in flat_list]

    plt.figure(figsize=(5, 5))
    plt.plot(100 * adata_var_dict["varexpl"] / adata_var_dict["tot_var_full"])
    # for i in range(10):
    #     plt.plot([100*x for x in ev[i]], c = 'grey')
    sns.lineplot(data=ev_df, x="PC", y="ev", color="grey")
    plt.ylim(ev_df["ev"].min(), 2 * ev_df["ev"].max())
    plt.title(
        "Explained Variance for Single Cell Data in PCA on Bulk Data \n Zoomed in to show EV for 10 shuffled datasets"
    )
    plt.ylabel("Percent Explained Variance")
    plt.xlabel("PC")
    plt.show()
