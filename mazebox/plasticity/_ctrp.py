import os.path as op
import scipy as sp
from scipy.spatial import distance
import pandas as pd
import scanpy as sc
from scvelo import settings
from scvelo import logging as logg
from scvelo.tools.utils import scale, groups_to_bool, strings_to_categoricals
from random import seed
from scipy.sparse import linalg, csr_matrix, issparse
import numpy as np
from matplotlib.colors import ListedColormap, to_rgba
import scvelo as scv
from scvelo.preprocessing.neighbors import get_connectivities
from scipy.sparse import csr_matrix


def write_to_obs(adata, key, vals, cell_subset=None):
    if cell_subset is None:
        adata.obs[key] = vals
    else:
        vals_all = (
            adata.obs[key].copy() if key in adata.obs.keys() else np.zeros(adata.n_obs)
        )
        vals_all[cell_subset] = vals
        adata.obs[key] = vals_all


# Would like to add in a check if the user wants to separate by groups. If some of the groups do not have an endpoint,
# suggest combining groups to find transitions across groups.
def ctrp(
    data,
    vkey="velocity",
    groupby=None,
    groups=None,
    self_transitions=False,
    basis=None,
    weight_diffusion=0,
    scale_diffusion=1,
    eps=1e-2,
    copy=False,
    adata_dist="X_pca",
    random_seed=1,
):
    """
    :param data: AnnData object
    :param vkey: velocity key to use for transition matrix
    :param groupby: if string, analysis will be run on each category separately
    :param groups: groups for cell subsetting
    :param self_transitions:
    :param basis: basis for transitions to be restricted to. None means that the transition matrix will contain all transitions,
    'pca' means only transitions within the pca reduced space are found, etc.
    :param weight_diffusion: weight of the diffusion kernel for transition matrix calculation
    :param scale_diffusion: scale of diffusion kernel for transition matrix calculation
    :param eps: tolerance for eigenvalue calculation
    :param copy: if True, return copy of AnnData object with updated parameters
    :param adata_dist: data to be used for distance calculation. Default X_pca means distances between cells will be
    calculated in PCA space
    :param random_seed: Default 1
    :return: adata if copy == True, else None and adata is updated with the obs attributes:
    ctrp: plasticity value for each cell
    absorbing: Boolean value detailing if cell is an end state
    """

    seed(random_seed)
    adata = data.copy() if copy else data
    logg.info("computing terminal states", r=True)

    strings_to_categoricals(adata)
    if groupby is not None:
        logg.warn(
            "Only set groupby, when you have evident distinct clusters/lineages,"
            " each with an own root and end point."
        )

    categories = [None]
    if groupby is not None and groups is None:
        categories = adata.obs[groupby].cat.categories

    # groupby = 'cell_fate' if groupby is None and 'cell_fate' in adata.obs.keys() else groupby
    # categories = adata.obs[groupby].cat.categories if groupby is not None and groups is None else [None]
    for c, cat in enumerate(categories):
        groups = cat if cat is not None else groups
        cell_subset = groups_to_bool(adata, groups=groups, groupby=groupby)
        _adata = adata if groups is None else adata[cell_subset]
        connectivities = get_connectivities(_adata, "distances")

        T = scv.tl.transition_matrix(
            _adata,
            vkey=vkey,
            basis=basis,
            weight_diffusion=weight_diffusion,
            scale_diffusion=scale_diffusion,
            self_transitions=self_transitions,
            backward=False,
        )

        eigvecs_ends = eigs(T, eps=eps, perc=[2, 98])[1]
        ends = csr_matrix.dot(connectivities, eigvecs_ends).sum(1)
        # ends = eigvecs_ends.sum(1)

        ends = scale(np.clip(ends, 0, np.percentile(ends, 98)))
        write_to_obs(adata, "end_points", ends, cell_subset)

        _adata.obs["col"] = ends

        n_ends = eigvecs_ends.shape[1]
        groups_str = " (" + groups + ")" if isinstance(groups, str) else ""

        logg.info("    identified " + str(n_ends) + " end points" + groups_str)
        print("Dropping absorbing rows for fundamental matrix...")

        absorbing = np.where(ends >= 1.0 - eps)[0]
        T = dropcols_coo(T, np.where(ends >= 1.0 - eps)[0])
        if not np.all(T.getnnz(1) == 0):
            dropped = absorbing
        else:
            dropped = np.concatenate((absorbing, np.where(T.getnnz(1) == 0)[0]))
            print("Dropping rows with all zeroes...")
            T = dropcols_coo(T, np.where(T.getnnz(1) == 0)[0])

        I = np.eye(T.shape[0])
        print(T.shape)

        # TODO: still under development. If det(I-T) = 0, inspect transition matrix.
        if np.abs(np.linalg.det(I - T)) == 0:
            print("Determinant equals 0. Solving for fate using pseudoinverse.")
            fate = np.linalg.pinv(I - T)
        else:
            print("Calculating fate...")
            fate = np.linalg.inv(I - T)  # fundamental matrix of markov chain
        if issparse(T):
            fate = fate.A
        fate = fate / fate.sum(axis=1)  # each row is a different starting cell

        if adata_dist is None:
            try:
                adataX = dropcols_coo(_adata.X, dropped)
                adataX = sp.sparse.csr_matrix.todense(np.expm1(adataX))
            except AttributeError:
                adataX = _adata.X
                mask = np.ones(len(adataX), dtype=bool)
                mask[dropped] = False
                adataX = adataX[mask, :]

        elif adata_dist == "raw":
            adataX = dropcols_coo(_adata.raw.X, dropped)

            adataX = sp.sparse.csr_matrix.todense(adataX)
        else:
            adataX = _adata.obsm[adata_dist]
            mask = np.ones(len(adataX), dtype=bool)
            mask[dropped] = False
            adataX = adataX[mask, :]
        print("Calculating distances...")
        dist = distance.cdist(adataX, adataX, "euclidean")
        print("Calculating inner product...")
        ip = np.einsum("ij,ij->i", fate, dist)
        if cell_subset is not None:
            cs = cell_subset.copy()
            new_subset = pd.Series(cs, index=adata.obs_names.values)
            true_subset = np.where(cs == True)[0]
            for i in list(dropped):
                new_subset.iloc[true_subset[i]] = False
        else:
            new_subset = pd.Series(
                [True] * len(adata.obs_names.values), index=adata.obs_names.values
            )
            for i in list(dropped):
                new_subset.iloc[i] = False
            if "ctrp" in adata.obs.keys():
                adata.obs["ctrp"] = np.zeros(adata.n_obs)
        write_to_obs(adata, "ctrp", np.log1p(ip), new_subset)

    adata.obs["absorbing"] = [i for i in adata.obs["end_points"] >= 1.0 - eps]

    return adata if copy else None


def eigs(T, k=100, eps=1e-3, perc=None, print_val=True):
    try:
        eigvals, eigvecs = linalg.eigs(
            T.T, k=k, which="LR", v0=T[0]
        )  # find k eigs with largest real part

        p = np.argsort(eigvals)[::-1]  # sort in descending order of eigenvalues
        eigvals = eigvals.real[p]
        eigvecs = eigvecs.real[:, p]

        idx = (
            eigvals >= 1 - eps
        )  # select eigenvectors with eigenvalue of 1 within eps tolerance
        eigvals = eigvals[idx]
        eigvecs = np.absolute(eigvecs[:, idx])
        if print_val:
            print("Eigenvalues: ", eigvals)
            # print(eigvecs.shape)
        if perc is not None:
            lbs, ubs = np.percentile(eigvecs, perc, axis=0)
            eigvecs[eigvecs < lbs] = 0
            eigvecs = np.clip(eigvecs, 0, ubs)
            eigvecs /= eigvecs.max(0)

    except:
        eigvals, eigvecs = np.empty(0), np.zeros(shape=(T.shape[0], 0))
    return eigvals, eigvecs


def dropcols_coo(M, idx_to_drop):
    idx_to_drop = np.unique(idx_to_drop)
    C = M.tocoo()
    keep = ~np.in1d(C.col, idx_to_drop)
    C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
    C.col -= idx_to_drop.searchsorted(C.col)  # decrement column indices
    C.row -= idx_to_drop.searchsorted(C.row)  # decrement column indices
    C._shape = (C.shape[0] - len(idx_to_drop), C.shape[1] - len(idx_to_drop))
    return C.tocsr()


# transition matrix P =
# |---------------|
# |   Q   |   R   |
# |---------------|
# |   0   |   I   |
# |---------------|


def extract_Q(P, connectivities=None, eps=0.001, eigen=False, eps_eig=1e-3):
    """
    For an absorbing Markov chain with transition rate matrix P this computes the
    matrix Q.

    Note that this does not assume that P is in the required format. It
    identifies the rows and columns that have a 1 in the diagonal and removes
    them.

    If using on real data, set eigen = True to use the method used in scvelo.
    """
    if eigen:
        eigvecs_ends = eigs(P, eps=eps_eig, perc=[2, 98])[1]
        n_ends = eigvecs_ends.shape[1]
        logg.info("    identified " + str(n_ends) + " end points")
        ends = csr_matrix.dot(connectivities, eigvecs_ends).sum(1)

        ends = scale(np.clip(ends, 0, np.percentile(ends, 98)))

        indices_without_1_in_diagonal = np.where(ends < 1.0 - eps)[0]
        Q = P[
            indices_without_1_in_diagonal.reshape(-1, 1), indices_without_1_in_diagonal
        ]
        return Q, ends
    else:
        indices_without_1_in_diagonal = np.where(P.diagonal() < 1 - eps)[0]
        Q = P[
            indices_without_1_in_diagonal.reshape(-1, 1), indices_without_1_in_diagonal
        ]
        return Q


def extract_R(P, connectivities=None, eps=0.001, eigen=False, eps_eig=1e-3):
    if eigen:
        eigvecs_ends = eigs(P, eps=eps_eig, perc=[2, 98], print_val=False)[1]
        ends = csr_matrix.dot(connectivities, eigvecs_ends).sum(1)

        ends = scale(np.clip(ends, 0, np.percentile(ends, 98)))
        indices_without_1_in_diagonal = np.where(ends < 1.0 - eps)[0]
        indices_with_1_in_diagonal = np.where(ends >= 1.0 - eps)[0]
    else:
        indices_without_1_in_diagonal = np.where(P.diagonal() < 1 - eps)[0]
        indices_with_1_in_diagonal = np.where(P.diagonal() >= 1 - eps)[0]

    R = P[indices_without_1_in_diagonal.reshape(-1, 1), indices_with_1_in_diagonal]
    return R


def compute_N(Q):
    """
    For an absorbing Markov chain with transition rate matrix P that gives
    matrix Q this computes the fundamental matrix N.
    """
    number_of_rows, _ = Q.shape
    N = np.linalg.inv(np.eye(number_of_rows) - Q)
    return N


def compute_t(P, eps=0.001):
    """
    For an absorbing Markov chain with transition rate matrix this computes the
    vector t which gives the expected number of steps until absorption.

    Note that this does not assume P is in the required format.
    """
    Q = extract_Q(P, eps)
    N = compute_N(Q)
    number_of_rows, _ = Q.shape
    return N @ np.ones(number_of_rows)


def get_long_run_state(pi, k, P):
    """
    For a Markov chain with transition matrix P and starting state vector pi,
    obtain the state distribution after k steps.
    """
    return pi @ np.linalg.matrix_power(P, k)


def expected_steps_fast(P, eps=0.001):
    Q = extract_Q(P, eps)
    I = np.identity(Q.shape[0])
    o = np.ones(Q.shape[0])
    np.linalg.solve(I - Q, o)


def p_absorbing(P, eps=0.001):
    """
    For an absorbing Markov chain with transition rate matrix, this computes
    the absorbing probabilities whose entries (i,j) are the probability of being absorbing in the absorbing state
    j when starting from transient state i.

    """
    Q = extract_Q(P, eps)
    N = compute_N(Q)

    R = extract_R(P, eps)
    return N.dot(R)


def p_transient(P, eps=0.001):
    """
    For an absorbing Markov chain with transition rate matrix, this computes
    the transient probabilities whose entries (i,j) are the probability of visiting the transient state
    j when starting from transient state i.

    """
    Q = extract_Q(P, eps)
    N = compute_N(Q)
    number_of_rows, _ = Q.shape
    I = np.identity(number_of_rows)
    Ndg = np.diagflat(np.diag(N))
    return (N - I) @ np.linalg.inv(Ndg)


def step_variance(P, eps=0.001):
    """
    For an absorbing Markov chain with transition rate matrix, this computes
    the variance on the number of steps before being absorbed when starting in transient state i.
    """
    Q = extract_Q(P, eps)
    N = compute_N(Q)
    number_of_rows, _ = Q.shape
    I = np.identity(number_of_rows)
    t = compute_t(P, eps)
    return (2 * N - I) @ t - np.multiply(t, t)


def calculate_ctrp(N, D):
    fate = (
        N.T / N.sum(axis=1)
    ).T  # each row is a different starting cell; fate is proportion of time spent in each state j when starting from i
    return np.einsum("ij,ij->i", fate, D)


def return_t_r_indices(P, connectivities=None, eps=0.001, eigen=False, eps_eig=1e-3):
    """
    :param P: transition matrix
    :param connectivities: connectivities from adata; only set if eigen == True
    :param eps: tolerance
    :param eigen: use eigenvalues to find absorbing states
    :return: [t,r]: indices of transient states and absorbing states, respectively
    """
    if eigen:
        eigvecs_ends = eigs(P, eps=eps_eig, perc=[2, 98], print_val=False)[1]
        n_ends = eigvecs_ends.shape[1]
        ends = csr_matrix.dot(connectivities, eigvecs_ends).sum(1)
        ends = scale(np.clip(ends, 0, np.percentile(ends, 98)))

        indices_without_1_in_diagonal = np.where(ends < 1.0 - eps)[0]
        indices_with_1_in_diagonal = np.where(ends >= 1.0 - eps)[0]
    else:
        indices_without_1_in_diagonal = np.where(P.diagonal() < 1 - eps)[0]
        indices_with_1_in_diagonal = np.where(P.diagonal() >= 1 - eps)[0]

    return indices_without_1_in_diagonal, indices_with_1_in_diagonal


def ctrp_simplified(
    data,
    vkey="velocity",
    groupby=None,
    groups=None,
    self_transitions=True,
    basis=None,
    eps_eig=1e-3,
    weight_diffusion=0,
    scale_diffusion=1,
    eps=1e-2,
    copy=False,
    distance_basis="X_pca",
    random_seed=1,
    T=None,  # only include a T matrix if groupby is also None.
):

    seed(random_seed)
    adata = data.copy() if copy else data
    strings_to_categoricals(adata)
    if groupby is not None:
        logg.warn(
            "Set groupby IFF you have sufficient evidence that each group occupies a different phenotypic landscape."
        )

    categories = [None]
    if groupby is not None and groups is None:
        categories = adata.obs[groupby].cat.categories

    for c, cat in enumerate(categories):
        groups = cat if cat is not None else groups
        cell_subset = groups_to_bool(adata, groups=groups, groupby=groupby)
        _adata = adata if groups is None else adata[cell_subset]
        connectivities = get_connectivities(_adata, "distances")
        if type(T) == type(None):
            ###########################
            ### ALEX LOOK HERE
            T = scv.tl.transition_matrix(
                _adata,
                vkey=vkey,
                basis=basis,
                weight_diffusion=weight_diffusion,
                scale_diffusion=scale_diffusion,
                self_transitions=self_transitions,
                backward=False,
            )
        print("(" + groups + ")" if isinstance(groups, str) else "")
        # try:
        T = T.todense()
        # except AttributeError:
        # pass
        print(T.shape)
        # Calcuate N, the fundamental matrix
        Q, ends = extract_Q(
            T, connectivities=connectivities, eigen=True, eps=eps, eps_eig=eps_eig
        )
        t_ind, r_ind = return_t_r_indices(
            T, connectivities=connectivities, eigen=True, eps=eps, eps_eig=eps_eig
        )
        R = T[t_ind.reshape(-1, 1), r_ind]
        N = compute_N(Q)
        ###########################

        adata.uns[f"{cat}_TM"] = T  ##added 8/6/21

        write_to_obs(adata, "end_points", ends, cell_subset)

        _adata.obs["col"] = ends
        # Calculate the distance matrix on the transient states
        if distance_basis is None:
            adataX = sp.sparse.csr_matrix.todense(np.expm1(_adata.X))
        elif distance_basis == "raw":
            adataX = sp.sparse.csr_matrix.todense(_adata.raw.X)
        else:
            adataX = _adata.obsm[distance_basis]
        D = distance.cdist(adataX, adataX, "euclidean")
        D = D[t_ind.reshape(-1, 1), t_ind]
        ctrp = calculate_ctrp(N, D)
        p_absorbing = N.dot(R)
        print(p_absorbing)
        if cell_subset is not None:
            cs = cell_subset.copy()
            new_subset = pd.Series(cs, index=adata.obs_names.values)
            true_subset = np.where(cs == True)[0]
            for i in list(r_ind):
                new_subset.iloc[true_subset[i]] = False
        else:
            new_subset = pd.Series(
                [True] * len(adata.obs_names.values), index=adata.obs_names.values
            )
            for i in list(r_ind):
                new_subset.iloc[i] = False
            if "ctrp" in adata.obs.keys():
                adata.obs["ctrp"] = np.zeros(adata.n_obs)
        write_to_obs(adata, "ctrp", np.log1p(ctrp), new_subset)

    adata.obs["absorbing"] = [i for i in adata.obs["end_points"] >= 1.0 - eps]

    return adata if copy else None
