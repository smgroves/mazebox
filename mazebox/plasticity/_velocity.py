import scanpy as sc
import scvelo as scv
import matplotlib.pyplot as plt


def velocity(adata, mode="deterministic", basis="umap", color=None, groupby=None):
    scv.pp.moments(adata)
    scv.tl.velocity(adata, mode=mode, groupby=groupby)
    scv.tl.velocity_graph(adata)
    scv.pl.velocity_embedding_grid(
        adata, basis=basis, color=color, figsize=(6, 4), alpha=0.3
    )
    plt.show()
    scv.tl.velocity_graph(adata)
    scv.tl.velocity_confidence(adata)
    keys = "velocity_length", "velocity_confidence"
    scv.pl.scatter(adata, c=keys, cmap="coolwarm", perc=[5, 95], basis=basis)
    plt.show()


def velocity_paga(adata, groups="leiden", vkey="velocity", basis="umap", color=None):
    adata.uns["neighbors"]["distances"] = adata.obsp["distances"]
    adata.uns["neighbors"]["connectivities"] = adata.obsp["connectivities"]
    scv.tl.paga(adata, groups=groups, vkey=vkey)
    scv.pl.paga(adata, basis=basis, color=color, palette="tab20")
