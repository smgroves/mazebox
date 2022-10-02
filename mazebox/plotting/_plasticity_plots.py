import scanpy as sc
import scvelo as scv
import matplotlib.pyplot as plt
import seaborn as sns


def ctrp_groups(adata, groups, basis="umap", ctrp="ctrp", vmax=None):
    for r in sorted(adata.obs[groups].cat.categories):
        _adata = adata.copy()
        _adata = _adata[
            _adata.obs[groups] == r,
        ]
        ax = scv.pl.scatter(
            adata, color="lightgrey", show=False, alpha=1, size=100, basis=basis
        )

        scv.pl.scatter(
            _adata,
            color=ctrp,
            basis=basis,
            fontsize=24,
            size=40,
            vmin=0,
            vmax=vmax,
            color_map="jet",
            title=r,
            alpha=0.5,
            ax=ax,
            zorder=2,
            legend_loc="right margin",
        )


def ctrp_hist(
    adata,
    groups=None,
    bins=100,
    fname=None,
    ctrp="ctrp",
    figsize=(8, 6),
    custom_palette=sns.color_palette("muted"),
):
    plt.figure(figsize=figsize)
    sns.reset_defaults()
    sns.set_palette(custom_palette)
    for r in adata.obs[groups].cat.categories:
        _adata = adata.copy()
        _adata = _adata[
            _adata.obs[groups] == r,
        ]
        sns.distplot(sorted(_adata.obs[ctrp]), bins=bins)
        print(r, _adata.obs[ctrp].mean())
        plt.ylabel("Density")
        plt.xlabel("Transport Potential")
    if type(fname) == type(None):
        plt.show()
    else:
        plt.savefig(f"./figures/{fname}.pdf")


def ctrp(adata, basis="umap", ctrp="ctrp", figsize=(6, 6)):
    _adata = adata[
        adata.obs["absorbing"] == True,
    ]
    ax = scv.pl.scatter(
        adata,
        color=ctrp,
        show=False,
        color_map="jet",
        alpha=0.5,
        basis=basis,
        dpi=350,
        zorder=1,
        figsize=figsize,
    )
    scv.pl.scatter(
        _adata, color="black", size=30, ax=ax, zorder=2, alpha=1, basis=basis
    )
