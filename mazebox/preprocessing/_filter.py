import dropkick as dk
import scvelo as scv
import scanpy as sc

# from _settings import settings
import numpy as np
import scrublet
import matplotlib.pyplot as plt


def dropkick_filter(adata, filter=True, verbose=False, plot=True):
    adata_rc = dk.dropkick(adata, n_jobs=4, verbose=verbose)
    if plot:
        dk.qc_summary(adata)
    if filter:
        adata = adata[(adata.obs.dropkick_label == 1), :].copy()
    return adata


def dropkick_recipe(
    adatas,
    batch_key,
    filter=True,
    plot=True,
    write=False,
    fname="",
    verbose=False,
    n_hvgs=2000,
    min_genes=100,
    min_counts=3,
    retain_genes=None,
    batch_categories=None,
    mito_names="^mt-|^MT-",
    n_ambient=10,
    target_sum=None,
    X_final="arcsinh_norm",
    n_comps=50,
):
    print("Running dropkick on each sample and filtering...")
    if len(adatas) == 1:
        adata = adatas[0]
        if "dropkick_score" not in adata.obs.keys():
            adata_rc = dk.dropkick(adata, n_jobs=4, verbose=verbose)
        if plot:
            dk.qc_summary(adata)
        if filter:
            adata = adata[(adata.obs.dropkick_label == 1), :].copy()
    else:
        adatas = [
            dropkick_filter(adata, verbose=verbose, plot=plot, filter=filter)
            for adata in adatas
            if "dropkick_score" not in adata.obs.keys()
        ]
        a1 = adatas.pop(0)
        adata = a1.concatenate(
            adatas, batch_key=batch_key, batch_categories=batch_categories
        )
        print(adata)

    print("Filtering and normalizing concatenated data...")
    # remove cells and genes with zero total counts
    orig_shape = adata.shape

    # store raw counts before manipulation
    adata.layers["raw_counts"] = adata.X.copy()

    # filter cells and genes
    sc.pp.filter_cells(adata, min_genes=min_genes)
    scv.pp.filter_genes(adata, min_counts=min_counts, retain_genes=retain_genes)
    if verbose:
        if adata.shape[0] != orig_shape[0]:
            print(
                "Ignoring {} barcodes with less than {} genes detected".format(
                    orig_shape[0] - adata.shape[0], min_genes
                )
            )
        if adata.shape[1] != orig_shape[1]:
            print(
                "Ignoring {} genes with zero total counts".format(
                    orig_shape[1] - adata.shape[1]
                )
            )
    adata.obs.drop(columns=["n_genes"], inplace=True)
    print(adata)

    # identify mitochondrial genes
    adata.var["mito"] = adata.var_names.str.contains(mito_names)
    # identify putative ambient genes by lowest dropout pct (top n_ambient)
    adata.var["pct_dropout_by_counts"] = np.array(
        (1 - (adata.X.astype(bool).sum(axis=0) / adata.n_obs)) * 100
    ).squeeze()
    lowest_dropout = adata.var.pct_dropout_by_counts.nsmallest(n=n_ambient).min()
    highest_dropout = adata.var.pct_dropout_by_counts.nsmallest(n=n_ambient).max()
    adata.var["ambient"] = adata.var.pct_dropout_by_counts <= highest_dropout
    # reorder genes by dropout rate
    adata = adata[:, np.argsort(adata.var.pct_dropout_by_counts)].copy()
    if verbose:
        print(
            "Top {} ambient genes have dropout rates between {} and {} percent:\n\t{}".format(
                len(adata.var_names[adata.var.ambient]),
                round(lowest_dropout, 3),
                round(highest_dropout, 3),
                adata.var_names[adata.var.ambient].tolist(),
            )
        )
    # calculate standard qc .obs and .var
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mito", "ambient"], inplace=True, percent_top=None
    )

    # other arcsinh-transformed metrics
    adata.obs["arcsinh_total_counts"] = np.arcsinh(adata.obs["total_counts"])
    adata.obs["arcsinh_n_genes_by_counts"] = np.arcsinh(adata.obs["n_genes_by_counts"])

    # normalize counts before transforming
    # with target_sum = None, this is equivalent to scv.pp.normalize_per_cell
    sc.pp.normalize_total(adata, target_sum=target_sum, layers=None, layer_norm=None)
    adata.layers["norm_counts"] = adata.X.copy()

    # arcsinh-transform normalized counts (adata.layers["arcsinh_norm"])
    adata.X = np.arcsinh(adata.layers["norm_counts"])
    sc.pp.scale(adata)  # scale genes for feeding into model
    adata.layers[
        "arcsinh_norm"
    ] = adata.X.copy()  # save arcsinh scaled counts in .layers

    adata.X = np.log1p(adata.layers["norm_counts"])
    sc.pp.scale(adata)  # scale genes for feeding into model
    adata.layers["log1p_norm"] = adata.X.copy()  # save log1p scaled counts in .layers

    # HVGs
    if n_hvgs is not None:
        if verbose:
            print("Determining {} highly variable genes".format(n_hvgs))
        # log1p transform for HVGs (adata.layers["log1p_norm"])
        # sc.pp.log1p(adata)
        # adata.layers["log1p_norm"] = adata.X.copy()  # save to .layers
        sc.pp.highly_variable_genes(
            adata, n_top_genes=n_hvgs  # , n_bins=20, flavor="seurat"
        )
        adata.var.drop(columns=["dispersions", "dispersions_norm"], inplace=True)

    # remove unneeded stuff
    del adata.layers["norm_counts"]

    # set .X as desired for downstream processing; default raw_counts
    if (X_final != "raw_counts") & verbose:
        print("Setting {} layer to .X".format(X_final))
    adata.X = adata.layers[X_final].copy()

    scv.tl.score_genes_cell_cycle(adata)
    if n_hvgs is not None:
        sc.tl.pca(adata, n_comps=n_comps, use_highly_variable=True)
    else:
        sc.tl.pca(adata, n_comps=n_comps, use_highly_variable=False)

    # if plot:
    #     sc.pl.pca(adata, color=[ "pct_counts_mito", "phase", batch_key])
    # if write:
    #     adata.write_h5ad(f"{settings.writedir}adata_dropkick{fname}.{settings.file_format_data}")
    return adata


def scanpy_recipe(adata, retain_genes, min_genes=100, min_cells=3):
    sc.pp.filter_cells(adata, min_genes=min_genes)
    scv.pp.filter_and_normalize(
        adata, min_cells=min_cells, retain_genes=retain_genes, log=True
    )
    return adata


def doublet_detections(
    adata, copy=False, layer="raw_counts", plot=False, log_transform=False
):
    if copy:
        _adata = adata.copy()
        counts = _adata.layers[layer]
        scrub = scrublet.Scrublet(counts)
        doublet_scores, predicted_doublets = scrub.scrub_doublets(
            log_transform=log_transform
        )
        _adata.obs["doublet_scores"] = doublet_scores
        _adata.obs["predicted_doublets"] = predicted_doublets
        return _adata
    else:
        counts = adata.layers[layer]
        scrub = scrublet.Scrublet(counts)
        doublet_scores, predicted_doublets = scrub.scrub_doublets(
            log_transform=log_transform
        )
        adata.obs["doublet_scores"] = doublet_scores
        adata.obs["predicted_doublets"] = predicted_doublets
    if plot:
        scrub.plot_histogram()
        print("Running UMAP...")
        scrub.set_embedding(
            "UMAP", scrublet.get_umap(scrub.manifold_obs_, 10, min_dist=0.3)
        )

        # # Uncomment to run tSNE - slow
        # print('Running tSNE...')
        # scrub.set_embedding('tSNE', scr.get_tsne(scrub.manifold_obs_, angle=0.9))

        # # Uncomment to run force layout - slow
        # print('Running ForceAtlas2...')
        # scrub.set_embedding('FA', scr.get_force_layout(scrub.manifold_obs_, n_neighbors=5. n_iter=1000))

        print("Done.")
        scrub.plot_embedding("UMAP", order_points=True)
        plt.show()
