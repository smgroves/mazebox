from typing import Optional, Union

import numpy as np
from anndata import AnnData
from sklearn.utils import check_random_state, check_array

from scanpy.tools._utils import get_init_pos_from_paga, _choose_representation
from scanpy import logging as logg
from scanpy._settings import settings
from scanpy._compat import Literal
from scanpy._utils import AnyRandom, NeighborsView

_InitPos = Literal["paga", "spectral", "random"]


def umap_update(
    adata: AnnData,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    maxiter: Optional[int] = None,
    alpha: float = 1.0,
    gamma: float = 1.0,
    negative_sample_rate: int = 5,
    init_pos: Union[_InitPos, np.ndarray, None] = "spectral",
    random_state: AnyRandom = 0,
    a: Optional[float] = None,
    b: Optional[float] = None,
    copy: bool = False,
    method: Literal["umap", "rapids"] = "umap",
    neighbors_key: Optional[str] = None,
) -> Optional[AnnData]:

    adata = adata.copy() if copy else adata

    if neighbors_key is None:
        neighbors_key = "neighbors"

    if neighbors_key not in adata.uns:
        raise ValueError(
            f'Did not find .uns["{neighbors_key}"]. Run `sc.pp.neighbors` first.'
        )
    start = logg.info("computing UMAP")

    neighbors = NeighborsView(adata, neighbors_key)

    if "params" not in neighbors or neighbors["params"]["method"] != "umap":
        logg.warning(
            f'.obsp["{neighbors["connectivities_key"]}"] have not been computed using umap'
        )
    from umap.umap_ import find_ab_params, simplicial_set_embedding

    if a is None or b is None:
        a, b = find_ab_params(spread, min_dist)
    else:
        a = a
        b = b
    adata.uns["umap"] = {"params": {"a": a, "b": b}}
    if isinstance(init_pos, str) and init_pos in adata.obsm.keys():
        init_coords = adata.obsm[init_pos]
    elif isinstance(init_pos, str) and init_pos == "paga":
        init_coords = get_init_pos_from_paga(
            adata, random_state=random_state, neighbors_key=neighbors_key
        )
    else:
        init_coords = init_pos  # Let umap handle it
    if hasattr(init_coords, "dtype"):
        init_coords = check_array(init_coords, dtype=np.float32, accept_sparse=False)

    if random_state != 0:
        adata.uns["umap"]["params"]["random_state"] = random_state
    random_state = check_random_state(random_state)

    neigh_params = neighbors["params"]
    X = _choose_representation(
        adata,
        neigh_params.get("use_rep", None),
        neigh_params.get("n_pcs", None),
        silent=True,
    )
    if method == "umap":
        # the data matrix X is really only used for determining the number of connected components
        # for the init condition in the UMAP embedding
        n_epochs = 0 if maxiter is None else maxiter
        X_umap = simplicial_set_embedding(
            X,
            neighbors["connectivities"].tocoo(),
            n_components,
            alpha,
            a,
            b,
            gamma,
            negative_sample_rate,
            n_epochs,
            init_coords,
            random_state,
            neigh_params.get("metric", "euclidean"),
            neigh_params.get("metric_kwds", {}),
            verbose=settings.verbosity > 3,
        )
    elif method == "umap_original":
        import umap

        X_contiguous = np.ascontiguousarray(X, dtype=np.float32)
        n_epochs = 200 if maxiter is None else maxiter
        n_neighbors = neighbors["params"]["n_neighbors"]

        umap = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            n_epochs=n_epochs,
            learning_rate=alpha,
            init=init_coords,
            min_dist=min_dist,
            spread=spread,
            negative_sample_rate=negative_sample_rate,
            a=a,
            b=b,
            verbose=settings.verbosity > 3,
            random_state=random_state,
        )
        X_umap = umap.fit_transform(X_contiguous)

    # elif method == 'rapids':
    #     metric = neigh_params.get('metric', 'euclidean')
    #     if metric != 'euclidean':
    #         raise ValueError(
    #             f'`sc.pp.neighbors` was called with `metric` {metric!r}, '
    #             "but umap `method` 'rapids' only supports the 'euclidean' metric."
    #         )
    #     from cuml import UMAP
    #     n_neighbors = neighbors['params']['n_neighbors']
    #     n_epochs = 500 if maxiter is None else maxiter # 0 is not a valid value for rapids, unlike original umap
    #     X_contiguous = np.ascontiguousarray(X, dtype=np.float32)
    #     umap = UMAP(
    #         n_neighbors=n_neighbors,
    #         n_components=n_components,
    #         n_epochs=n_epochs,
    #         learning_rate=alpha,
    #         init=init_pos,
    #         min_dist=min_dist,
    #         spread=spread,
    #         negative_sample_rate=negative_sample_rate,
    #         a=a,
    #         b=b,
    #         verbose=settings.verbosity > 3,
    #     )
    #     X_umap = umap.fit_transform(X_contiguous)
    adata.obsm["X_umap_orig"] = X_umap  # annotate samples with UMAP coordinates
    logg.info(
        "    finished",
        time=start,
        deep=("added\n" "    'X_umap', UMAP coordinates (adata.obsm)"),
    )
    return (adata, umap) if copy else umap
