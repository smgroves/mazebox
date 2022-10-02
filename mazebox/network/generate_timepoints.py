# magic_recipe
import magic
import scvelo as scv
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import os
import os.path as op


def magic_recipe(adata, genes, clusters, fname="", directory="out"):
    try:
        os.mkdir(op.join(f"./{directory}/"))
    except FileExistsError:
        pass
    genes = list(set(genes).intersection(set(adata.var_names)))
    magic_operator = magic.MAGIC(verbose=True, solver="approximate")
    adata.layers["t0"] = magic_operator.fit_transform(adata.layers["Ms"])
    adata.layers["t1"] = magic_operator.transform(
        scv.utils.get_extrapolated_state(adata)
    )
    t0 = pd.DataFrame(
        adata[:, genes].layers["t0"], index=adata.obs_names, columns=genes
    )
    t0.to_csv(f"./{directory}/t0_{fname}.csv")

    t1 = pd.DataFrame(
        adata[:, genes].layers["t1"], index=adata.obs_names, columns=genes
    )
    t1.to_csv(f"./{directory}/t1_{fname}.csv")

    network_clusters = pd.DataFrame(adata.obs[clusters])
    network_clusters.to_csv(f"./{directory}/{fname}_clusters.csv")


def moments_recipe(adata, genes, clusters, fname="", directory="out"):
    try:
        os.mkdir(op.join(f"./{directory}/"))
    except FileExistsError:
        pass
    genes = sorted(list(set(genes).intersection(set(adata.var_names))))
    adata.layers["t0"] = adata.layers["Ms"]
    adata.layers["t1"] = scv.utils.get_extrapolated_state(adata)
    t0 = pd.DataFrame(
        adata[:, genes].layers["t0"], index=adata.obs_names, columns=genes
    )
    t0.to_csv(f"./{directory}/t0_{fname}.csv")

    t1 = pd.DataFrame(
        adata[:, genes].layers["t1"], index=adata.obs_names, columns=genes
    )
    t1.to_csv(f"./{directory}/t1_{fname}.csv")

    network_clusters = pd.DataFrame(adata.obs[clusters])
    network_clusters.to_csv(f"./{directory}/{fname}_clusters.csv")
