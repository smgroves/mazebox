import scanorama
import scanpy as sc
import scvelo as scv
import pandas as pd
import numpy as np

def scanorama_recipe(adata, copy = False, groups = 'timepoints', correct = False, basis = 'umap', random_state = 0):

    batches = sorted(adata.obs[groups].cat.categories)

    alldata = {}
    for batch in batches:
        print(batch)
        alldata[batch] = adata[adata.obs[groups] == batch,]

    # convert to list of AnnData objects
    adatas = list(alldata.values())
    # integrated, corrected = scanorama.correct_scanpy(adatas, return_dimred=True)

    #updated to be compatible with scanorama v1.7, which only returns corrected
    corrected = scanorama.correct_scanpy(adatas, return_dimred=True)
    all_ind = sc.concat(adatas).obs_names
    all_s = np.concatenate(corrected.obs['X_scanorama'])
    print(all_s.shape)

    all_sc = pd.DataFrame(all_s, index=all_ind)
    all_sc = all_sc.loc[adata.obs_names]
    adata.obsm["X_SC"] = np.array(all_sc)
    print('added attribute X_SC with integrated data')
    if correct:
        first = corrected.pop(0)
        adata_SC = first.concatenate(corrected, batch_key=groups,
                                            batch_categories=batches)
        sc.tl.pca(adata_SC, random_state = random_state)
        sc.pp.neighbors(adata_SC, random_state = random_state)
        if basis == 'umap':
            sc.tl.umap(adata_SC, random_state = random_state)
        elif basis == 'tsne':
            sc.tl.tsne(adata_SC, random_state = random_state)
        else:
            print('Basis must be one of ["umap","tsne"]')

        sc.pl.scatter(adata_SC, basis=basis, color=groups)
        adata.obsm['X_pca_sc'] = np.array(pd.DataFrame(adata_SC.obsm['X_pca'], index=all_ind).loc[adata.obs_names])
        if basis in ['umap','tsne']:
            adata.obsm[f'X_{basis}_sc'] = np.array(pd.DataFrame(adata_SC.obsm[f'X_{basis}'], index=all_ind).loc[adata.obs_names])
            print(f"added attributes X_pca_sc and X_{basis}_sc")
    if copy: return adata.copy()


