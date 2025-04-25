"""Gene expression simulation from SCENIC+ results."""
from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any, Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pycisTopic.signature_enrichment import signature_enrichment
from scipy import sparse
from scipy.stats import norm as normal
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from scenicplus.eregulon_enrichment import get_eRegulons_as_signatures, rank_data

if TYPE_CHECKING:
    import pandas as pd

RF_KWARGS = {
    "n_jobs": 1,
    "n_estimators": 1000,
    "max_features": "sqrt"
}

GBM_KWARGS = {
    "learning_rate": 0.01,
    "n_estimators": 500,
    "max_features": 0.1
}

SKLEARN_REGRESSOR_FACTORY: dict[str, RegressorMixin] = {
    "RF": RandomForestRegressor,
    "GBM": GradientBoostingRegressor
}

def _do_one_round_of_simulation(
        original_matrix: pd.DataFrame,
        perturbed_matrix: pd.DataFrame,
        regressors: dict[str, tuple[list[str], RegressorMixin]]
    ):
    new_exp_mtx = perturbed_matrix.copy()
    for gene in tqdm(regressors.keys(), total = len(regressors.keys()), leave = False):
        TFs, gene_regressor = regressors[gene]
        TF_original_exp = original_matrix[TFs].to_numpy()
        TF_perturbed_exp = perturbed_matrix[TFs].to_numpy()
        if not all((TF_original_exp == TF_perturbed_exp).ravel()):
            gene_predicted_exp_orig = gene_regressor.predict(TF_original_exp)
            gene_predicted_exp_perturbed = gene_regressor.predict(TF_perturbed_exp)
            fc = gene_predicted_exp_perturbed / gene_predicted_exp_orig
            new_exp_mtx[gene] = new_exp_mtx[gene] * fc
        else:
            continue
    return new_exp_mtx

def train_gene_expression_models(
    df_EXP: pd.DataFrame,
    gene_to_TF: dict[str, list[str]],
    genes: list[str] | None = None,
    regressor_type: Literal["GBM", "RF"] = "GBM",
    regressor_kwargs: dict[str, Any] = GBM_KWARGS
) -> dict[str, tuple[list[str], RegressorMixin]]:
    """
    Train a regression model for each gene using TFs as predictores for gene expression.

    Parameters
    ----------
    df_EXP: pd.DataFrame
        pandas DataFrame containing expression matrix (cell x gene).
    gene_to_TF: dict[str, list[str]]
        mapping between gene and the TFs that are predicted to regulate that gene
    genes: list[str] | None
        list of genes for which to predict expression. Default (None) is all genes.
    regressor_type: Literal["GBM", "RF"]
        regressor type to use, either gradient boosting machines (GBM) or Random
        Forrest (RF).
    regressor_kwargs: dict[str, Any]
        Keyword arguments that are bassed to the regressor.

    Returns
    -------
    dict[str, tuple[list[str], RegressorMixin]]
        A dictionary of the form {gene: (TFs, regressor)}

    """
    if regressor_type not in SKLEARN_REGRESSOR_FACTORY:
        raise ValueError(f"Please select a regressor_type from {', '.join(SKLEARN_REGRESSOR_FACTORY.keys())}")
    if genes is None:
        genes = list(gene_to_TF.keys())
    if not all(g in df_EXP.columns for g in genes):
        raise ValueError("Some genes are not in the expression matrix, please check input!")
    regressors: dict[str, tuple[list[str], RegressorMixin]] = {}
    for gene in tqdm(genes, total = len(genes)):
        regressor: RegressorMixin = SKLEARN_REGRESSOR_FACTORY[regressor_type](**regressor_kwargs)
        predictor_TF = gene_to_TF[gene].copy()
        #remove gene itself as predictor
        if gene in predictor_TF:
            predictor_TF.remove(gene)
        if len(predictor_TF) == 0:
            continue
        predictor_TF_exp_v = df_EXP[predictor_TF].to_numpy()
        if len(predictor_TF_exp_v.shape) == 1:
            predictor_TF_exp_v = predictor_TF_exp_v.reshape(-1, 1)
        predictand_target_gene_exp_v = df_EXP[gene].to_numpy()
        regressor.fit(predictor_TF_exp_v, predictand_target_gene_exp_v)
        regressors[gene] = (predictor_TF, regressor)
    return regressors

def simulate_perturbation(
    df_EXP: pd.DataFrame,
    perturbation: dict[str, list[float | int] | int | float],
    keep_intermediate: bool = False,
    n_iter: int = 5,
    regressors = dict[list[str], tuple[list[str], RegressorMixin]]
) -> pd.DataFrame | dict[int, pd.DataFrame]:
    """
    Simulate a perturbation.

    Parameters
    ----------
    df_EXP: pd.DataFrame
        Pandas DataFrame containing expression matrix (cell x gene).
    perturbation: dict[str, list[float | int] | int | float]
        Dictionary specifying perturbation.
        This can be a mapping between a gene and a single value (all cells will be
        set to this value during the perturbation) or a mapping between a gene and
        a list of values containing one value per cell.
    keep_intermediate: bool
        Wether or not to keep intermediate values, if True a dictionary of dataframes
        is returned. Default, False.
    n_iter: int
        Number of iterations, default is 5.
    regressors = dict[list[str], tuple[list[str], RegressorMixin]]
        Regressors from `train_gene_expression_models`.

    Returns
    -------
    pd.DataFrame | dict[int, pd.DataFrame]
        A single pandas dataframe containing predicted values at the last iteration
        in case keep_intermediate is False, otherwise a dictionary of pandas dataframes
        containg predicted values at each iteration.

    """
    if keep_intermediate:
        perturbation_over_iter: dict[int, pd.DataFrame] = {}
    #initialize
    original_matrix = df_EXP.copy()
    perturbed_matrix = original_matrix.copy()
    for gene in perturbation:
        gene_perturbation = perturbation[gene]
        if isinstance(gene_perturbation, list):
            if len(gene_perturbation) != len(perturbed_matrix):
                raise ValueError("When specifying a list of perturbations the length of the list must equal the number of cells in the gene expression matrix")
        perturbed_matrix[gene] = gene_perturbation
    if keep_intermediate:
        perturbation_over_iter[0] = original_matrix
        perturbation_over_iter[1] = perturbed_matrix
    #do several iterations of perturbation
    for i in range(n_iter):
        new_matrix = _do_one_round_of_simulation(original_matrix, perturbed_matrix, regressors)
        original_matrix = perturbed_matrix.copy()
        perturbed_matrix = new_matrix.copy()
        if keep_intermediate:
            perturbation_over_iter[i + 2] = perturbed_matrix
    if keep_intermediate:
        return perturbation_over_iter
    else:
        return perturbed_matrix

def permute_rows_nsign(A: np.ndarray) -> None:
    """Permute in place the entries and randomly switch the sign for each row of a matrix independently. From celloracle."""
    plmi = np.array([+1, -1])
    for i in range(A.shape[0]):
        np.random.shuffle(A[i, :])
        A[i, :] = A[i, :] * np.random.choice(plmi, size=A.shape[1])

def _project_perturbation_in_embedding(
    embedding: np.ndarray,
    perturbed_matrix: np.ndarray,
    original_matrix: np.ndarray,
    sigma_corr: float = 0.05,
    n_cpu: int = 1
) -> np.ndarray:
    try:
        from velocyto.estimation import colDeltaCorpartial
    except Exception as e:
        print("Please install velocyto to project perturbation")
        raise e
    #based on celloracle/velocyto code
    delta_matrix: np.ndarray = perturbed_matrix - original_matrix
    delta_matrix_random =  delta_matrix.copy()
    permute_rows_nsign(delta_matrix_random)
    n_neighbors = int(perturbed_matrix.shape[0] / 5) #default from cell oracle
    nn = NearestNeighbors(n_neighbors = n_neighbors + 1, n_jobs = n_cpu)
    nn.fit(embedding)
    embedding_knn = nn.kneighbors_graph(mode = "connectivity")

    # Pick random neighbours and prune the rest
    neigh_ixs = embedding_knn.indices.reshape((-1, n_neighbors + 1))
    p = np.linspace(0.5, 0.1, neigh_ixs.shape[1])
    p = p / p.sum()

    # There was a problem of API consistency because the random.choice can pick the diagonal value (or not)
    # resulting self.corrcoeff with different number of nonzero entry per row.
    # Not updated yet not to break previous analyses
    # Fix is substituting below `neigh_ixs.shape[1]` with `np.arange(1,neigh_ixs.shape[1]-1)`
    # I change it here since I am doing some breaking changes
    sampling_ixs = np.stack([np.random.choice(neigh_ixs.shape[1],
                                            size=(int(0.3 * (n_neighbors + 1)),),
                                            replace=False,
                                            p=p) for i in range(neigh_ixs.shape[0])], 0)
    neigh_ixs = neigh_ixs[np.arange(neigh_ixs.shape[0])[:, None], sampling_ixs]
    nonzero = neigh_ixs.shape[0] * neigh_ixs.shape[1]
    embedding_knn = sparse.csr_matrix((np.ones(nonzero),
                                        neigh_ixs.ravel(),
                                        np.arange(0, nonzero + 1, neigh_ixs.shape[1])),
                                        shape=(neigh_ixs.shape[0],
                                                neigh_ixs.shape[0]))

    corrcoef = colDeltaCorpartial(perturbed_matrix.T, delta_matrix.T, neigh_ixs,  threads = n_cpu)
    corrcoef[np.isnan(corrcoef)] = 1
    np.fill_diagonal(corrcoef, 0)

    transition_prob = np.exp(corrcoef / sigma_corr) * embedding_knn.A
    transition_prob /= transition_prob.sum(1)[:, None]

    unitary_vectors = embedding.T[:, None, :] - embedding.T[:, :, None]  # shape (2,ncells,ncells)
    unitary_vectors /= np.linalg.norm(unitary_vectors, ord=2, axis=0)  # divide by L2
    np.fill_diagonal(unitary_vectors[0, ...], 0)  # fix nans
    np.fill_diagonal(unitary_vectors[1, ...], 0)

    delta_embedding = (transition_prob * unitary_vectors).sum(2)
    delta_embedding = delta_embedding - ((embedding_knn.A * unitary_vectors).sum(2) / embedding_knn.sum(1).A.T)
    delta_embedding = delta_embedding.T
    return delta_embedding

def _calculate_grid_arrows(
    embedding: np.ndarray,
    delta_embedding: np.ndarray,
    offset_frac: float,
    n_grid_cols: int,
    n_grid_rows: int,
    n_neighbors: int,
    n_cpu: int
):
    #prepare grid
    min_x = min(embedding[:, 0])
    max_x = max(embedding[:, 0])
    min_y = min(embedding[:, 1])
    max_y = max(embedding[:, 1])
    offset_x = (max_x - min_x) * offset_frac
    offset_y = (max_y - min_y) * offset_frac
    #calculate number of points underneath grid points
    x_dist_between_points = (max_x - min_x) / n_grid_cols
    y_dist_between_points = (max_y - min_y) / n_grid_rows
    minimal_distance = np.mean([y_dist_between_points, x_dist_between_points]) #will be used to mask certain points in the grid

    grid_x, grid_y = np.meshgrid(
        np.linspace(min_x + offset_x, max_x - offset_x, n_grid_cols),
        np.linspace(min_y + offset_y, max_y - offset_y, n_grid_rows)
    )
    grid_xy = np.array([np.hstack(grid_x), np.hstack(grid_y)]).T

    #find neighbors of gridpoints
    nn = NearestNeighbors(n_neighbors = n_neighbors, n_jobs = n_cpu)
    nn.fit(embedding)
    dists, neighs = nn.kneighbors(grid_xy)

    std = np.mean([abs(g[1] - g[0]) for g in grid_xy])
    # isotropic gaussian kernel
    gaussian_w = normal.pdf(loc=0, scale=0.5*std, x=dists)
    total_p_mass = gaussian_w.sum(1)

    uv = (delta_embedding[neighs] * gaussian_w[:, :, None]).sum(1) / np.maximum(1, total_p_mass)[:, None]

    #mask points in the grid which don't have points of the embedding underneath them
    mask = dists.min(1) < minimal_distance

    return grid_xy, uv, mask

def plot_perturbation_effect_in_embedding(
    perturbed_matrix: pd.DataFrame,
    original_matrix: pd.DataFrame,
    embedding: np.ndarray,
    AUC_kwargs: dict[str, Any] = {},
    ax: matplotlib.axes | None = None,
    grid_offset_frac: float = 0.005,
    grid_n_cols: int = 25,
    grid_n_rows: int = 25,
    grid_n_neighbors: int = 25,
    eRegulons: pd.DataFrame | None = None,
    n_cpu: int = 1,
    calculate_perturbed_auc_values: bool = True,
):
    level = logging.INFO
    format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger("Perturbation")
    if calculate_perturbed_auc_values:
        eRegulon_signatures = get_eRegulons_as_signatures(eRegulons=eRegulons)
        log.info("Generating ranking based on perturbed matrix.")
        perturbed_ranking = rank_data(perturbed_matrix)
        log.info("Scoring eRegulons.")
        perturbed_ranking = signature_enrichment(
            perturbed_ranking,
            eRegulon_signatures["Gene_based"],
            enrichment_type="gene",
            n_cpu = n_cpu,
            **AUC_kwargs)

    log.info("Projecting perturbation effect in embedding")
    delta_embedding = _project_perturbation_in_embedding(
        embedding = embedding,
        perturbed_matrix = perturbed_matrix.to_numpy(),
        original_matrix = original_matrix.to_numpy(),
        n_cpu = n_cpu
    )

    log.info("Calculating grid of arrows")
    grid_xy, uv, mask = _calculate_grid_arrows(
        embedding=embedding,
        delta_embedding=delta_embedding,
        offset_frac=grid_offset_frac,
        n_grid_cols=grid_n_cols,
        n_grid_rows=grid_n_rows,
        n_neighbors=grid_n_neighbors,
        n_cpu=n_cpu
    )
    distances = np.sqrt((uv**2).sum(1))
    norm = matplotlib.colors.Normalize(vmin=0.15, vmax=0.5, clip=True)
    def scale(X):
        return [(x - min(X)) / (max(X) - min(X)) for x in X]
    uv[np.logical_or(~mask, np.array(scale(distances)) < 0.15)] = np.nan
    log.info("Plotting")
    if ax is None:
        fig, ax = plt.subplots()
    ax.streamplot(
            grid_xy.reshape(grid_n_cols,grid_n_rows, 2)[:, :, 0],
            grid_xy.reshape(grid_n_cols,grid_n_rows, 2)[:, :, 1],
            uv.reshape(grid_n_cols,grid_n_rows, 2)[:, :, 0],
            uv.reshape(grid_n_cols,grid_n_rows, 2)[:, :, 1],
            density = 3,
            color = np.array(scale(distances)).reshape(grid_n_cols, grid_n_rows),
            cmap = "Greys",
            zorder = 10,
            norm = norm,
            linewidth = 0.5)
    return ax
