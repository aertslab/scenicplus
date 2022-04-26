"""Gene expression simulation from SCENIC+ results
"""

from multiprocessing.sharedctypes import Value
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from tqdm import tqdm
import numpy as np
from velocyto.estimation import colDeltaCorpartial
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm as normal
from .dimensionality_reduction import plot_metadata_given_ax
import matplotlib.pyplot as plt
import logging
import sys
import ray
from typing import Dict, List, Mapping, Optional, Sequence
import pandas as pd


RF_KWARGS = {
    'n_jobs': 1,
    'n_estimators': 1000,
    'max_features': 'sqrt'
}

GBM_KWARGS = {
    'learning_rate': 0.01,
    'n_estimators': 500,
    'max_features': 0.1
}

SKLEARN_REGRESSOR_FACTORY = {
    'RF': RandomForestRegressor,
    'GBM': GradientBoostingRegressor
}

DEFAULT_REGRESSOR_PARAMS = {
    'RF': RF_KWARGS,
    'GBM': GBM_KWARGS
}

def _do_one_round_of_simulation(original_matrix, perturbed_matrix, regressors):
    new_exp_mtx = perturbed_matrix.copy()
    for gene in tqdm(regressors.keys(), total = len(regressors.keys())):
        gene_regressor = regressors[gene][-1]
        TF_original_exp = original_matrix[regressors[gene][0: len(regressors[gene]) -1]].to_numpy()
        TF_perturbed_exp = perturbed_matrix[regressors[gene][0: len(regressors[gene]) -1]].to_numpy()
        if not all((TF_original_exp == TF_perturbed_exp).ravel()):
            gene_predicted_exp_orig = gene_regressor.predict(TF_original_exp)
            gene_predicted_exp_perturbed = gene_regressor.predict(TF_perturbed_exp)
            fc = gene_predicted_exp_perturbed / gene_predicted_exp_orig
            new_exp_mtx[gene] = new_exp_mtx[gene] * fc
        else:
            continue
    return new_exp_mtx

def train_gene_expression_models(
    scplus_obj: 'SCENICPLUS', 
    eRegulon_metadata_key: Optional[str] = 'eRegulon_metadata',
    genes: Optional[List] = None, 
    regressor_type: Optional[str] = 'GBM', 
    regressor_kwargs: Optional[dict] = None, 
    eRegulons_to_use: Optional[List] = None) -> List:
    """
    Train a regression model for each gene using eRegulons as predictors

    Parameters
    ---------
    scplus_obj: `class::SCENICPLUS`
        A SCENICPLUS object.
    eRegulon_metadata_key: str
        Key using which to find the eRegulon metadata panda dataframe in scplus_obj.uns
        Default:  'eRegulon_metadata',
    genes: List
        List of genes for which to train the regression models. Default uses all genes.
    regressor_type: str
        Method to use for regression, options are GBM (Gradient Boosting Machine) and RF (Random Forrest).
    regressor_kwargs: dict
        Keyword arguments containing parameters to use for training the regression model.
    eRegulons_to_use: List
        List of eRegulons to consider as predictors. Default uses all eRegulons in scplus_obj.uns[eRegulon_metadata_key]

    Returns
    -------
    A dictionary indexed by gene names containing as values a list of predictor TFs together with the regression model
    """
    #check arguments
    if regressor_type not in SKLEARN_REGRESSOR_FACTORY.keys():
        raise ValueError(f'Please select a regressor_type from {", ".join(SKLEARN_REGRESSOR_FACTORY.keys())}')
    if eRegulon_metadata_key not in scplus_obj.uns.keys():
        raise ValueError(f'key {eRegulon_metadata_key} not found in scplus_obj.keys()')
    if regressor_kwargs is None:
        regressor_kwargs = DEFAULT_REGRESSOR_PARAMS[regressor_type]
    #if genes is set to None, predict expression of all genes
    if genes is None:
        genes = scplus_obj.gene_names
    df_EXP = scplus_obj.to_df('EXP')
    eRegulon_metadata = scplus_obj.uns[eRegulon_metadata_key].copy()
    #subset eRegulon_metadata for selected eRegulons (only these will be used as predictors)
    if eRegulons_to_use is not None:
        idx_to_keep = np.isin(
            [x.split('_(')[0] for x in eRegulon_metadata['Region_signature_name']], 
            [x.split('_(')[0] for x in eRegulons_to_use])
        eRegulon_metadata = eRegulon_metadata.loc[idx_to_keep]
    regressors = {}
    for gene in tqdm(genes, total = len(genes)):
        regressor = SKLEARN_REGRESSOR_FACTORY[regressor_type](**regressor_kwargs)
        predictor_TF = list(set(eRegulon_metadata.query("Gene == @gene")["TF"]))
        if len(predictor_TF) == 0:
            continue
        predictor_TF_exp_v = df_EXP[predictor_TF].to_numpy()
        if len(predictor_TF_exp_v.shape) == 1:
            predictor_TF_exp_v = predictor_TF_exp_v.reshape(-1, 1)
        predictand_target_gene_exp_v = df_EXP[gene].to_numpy()
        regressor.fit(predictor_TF_exp_v, predictand_target_gene_exp_v)
        regressors[gene] = [*predictor_TF, regressor]
    return regressors

        
def simulate_perturbation(
    scplus_obj: 'SCENICPLUS', 
    perturbation: dict, 
    eRegulon_metadata_key: Optional[str] = 'eRegulon_metadata',
    n_iter: Optional[int] = 5, 
    regressors: Optional[dict] = None, 
    genes: Optional[List] = None, 
    regressor_type: Optional[str] = 'GBM',
    regressor_kwargs: Optional[dict] = None, 
    eRegulons_to_use: Optional[List] = None, 
    keep_intermediate: Optional[bool] = False):
    """
    Simulate TF perturbation
    
    Parameters
    ---------
    scplus_obj: `class::SCENICPLUS`
        A SCENICPLUS object.
    perturbation: dict
        A dictionary indexed by TF names with perturbation level as values. 
        e.g. {'SOX10': 0} will simulate a perturbation where the expression level of SOX10 is set to 0 in all cells
    eRegulon_metadata_key: str
        Key using which to find the eRegulon metadata panda dataframe in scplus_obj.uns
        Default:  'eRegulon_metadata'
    n_iter: int
        Number of itertions to simulate. Default is 5
    regressors: dict
        Dictionary of regressors as generated by train_gene_expression_models. 
        If set to None, this dictionary will be generated internally.
    genes: List
        List of genes for which to train the regression models. Default uses all genes.
    regressor_type: str
        Method to use for regression, options are GBM (Gradient Boosting Machine) and RF (Random Forrest).
    regressor_kwargs: dict
        Keyword arguments containing parameters to use for training the regression model.
    eRegulons_to_use: List
        List of eRegulons to consider as predictors. Default uses all eRegulons in scplus_obj.uns[eRegulon_metadata_key]
    keep_intermediate: bool
        If set to True simulated gene expression values for each iteration will be kept

    Returns
    -------
    A single or list of simulated matrices.

    """
    #check arguments
    if regressors is None:
        regressors = train_gene_expression_models(
            scplus_obj = scplus_obj,
            genes = genes,
            regressor_type = regressor_type,
            regressor_kwargs = regressor_kwargs,
            eRegulons_to_use = eRegulons_to_use,
            eRegulon_metadata_key = eRegulon_metadata_key)
    if keep_intermediate:
        perturbation_over_iter = {}
    #initialize
    original_matrix = scplus_obj.to_df('EXP').copy()
    perturbed_matrix = original_matrix.copy()
    for gene in perturbation.keys():
        perturbed_matrix[gene] = perturbation[gene]
    if keep_intermediate:
        perturbation_over_iter['0'] = original_matrix
        perturbation_over_iter['1'] = perturbed_matrix
    #do several iterations of perturbation
    for i in range(n_iter):
        new_matrix = _do_one_round_of_simulation(original_matrix, perturbed_matrix, regressors)
        original_matrix = perturbed_matrix.copy()
        perturbed_matrix = new_matrix.copy()
        if keep_intermediate:
            perturbation_over_iter[str(i + 2)] = perturbed_matrix
    if keep_intermediate:
        return perturbation_over_iter
    else:
        return perturbed_matrix

def permute_rows_nsign(A: np.ndarray) -> None:
    """Permute in place the entries and randomly switch the sign for each row of a matrix independently.
    From celloracle
    """
    plmi = np.array([+1, -1])
    for i in range(A.shape[0]):
        np.random.shuffle(A[i, :])
        A[i, :] = A[i, :] * np.random.choice(plmi, size=A.shape[1])

def _project_perturbation_in_embedding(scplus_obj, perturbed_matrix, reduction_name, sigma_corr = 0.05, n_cpu = 1):
    #based on celloracle/velocyto code
    if reduction_name not in scplus_obj.dr_cell.keys():
        raise ValueError(f'Embbeding "{reduction_name}" not found!')
    original_matrix = scplus_obj.to_df('EXP').copy().to_numpy().astype('double')
    delta_matrix = original_matrix - perturbed_matrix.to_numpy().astype('double')
    delta_matrix_random =  delta_matrix.copy()
    permute_rows_nsign(delta_matrix_random)

    embedding = scplus_obj.dr_cell[reduction_name].to_numpy()
    n_neighbors = int(original_matrix.shape[0] / 5) #default from cell oracle
    nn = NearestNeighbors(n_neighbors = n_neighbors + 1, n_jobs = n_cpu)
    nn.fit(embedding)
    embedding_knn = nn.kneighbors_graph(mode = 'connectivity')

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

    corrcoef = colDeltaCorpartial(original_matrix.T, delta_matrix.T, neigh_ixs,  threads = n_cpu)
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

def _calculate_grid_arrows(embedding, delta_embedding, offset_frac, n_grid_cols, n_grid_rows, n_neighbors, n_cpu):
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
    scplus_obj: 'SCENICPLUS', 
    perturbed_matrix: pd.DataFrame, 
    reduction_name: str, 
    grid_offset_frac: Optional[float] = 0.01,
    grid_n_cols: Optional[int] = 50,
    grid_n_rows: Optional[int] = 50,
    grid_n_neighbors: Optional[int] = 1000,
    arrow_scale: Optional[float] = 3,
    n_cpu: Optional[int] = 1,
    figsize: Optional[tuple] = (6.4, 4.8),
    save: Optional[str] = None,
    **kwargs):
    """
    Plot dimensionality reduction with perturbation arrows in a grid.

    Parameters
    ----------
    scplus_obj: `class::SCENICPLUS`
        A SCENICPLUS object.
    perturbed_matrix: pd.DataFrame
        A pandas dataframe with the perturbed gene expression values, as generated by simulate_perturbation
    reduction_name: str
        name of the cellular dimensionality reduction to use
    grid_offset_frac: float
        Fraction of whitespace to use surounding the plot for plotting the arrows.
    grid_n_cols: int
        Number of columns to plot the grid of arrows
    grid_n_rows: int
        Number of rows to plot the grid of arrows
    grid_n_neighbors: int
        Number of neighbors to consider when calculating the grid of arrows.
    n_cpu: int
        Number of cpus to use.
    figsize: tuple
        Tuple indicating the size of the plot
    save: str
        Path where to save the figure.

    """

    level = logging.INFO
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger('perturbation')

    log.info(f'Projecting perturbation effect in embedding: {reduction_name}')
    delta_embedding = _project_perturbation_in_embedding(
        scplus_obj=scplus_obj,
        perturbed_matrix=perturbed_matrix,
        reduction_name=reduction_name,
        n_cpu = n_cpu)

    log.info('Calculating grid of arrows')
    embedding = scplus_obj.dr_cell[reduction_name].to_numpy()
    grid_xy, uv, mask = _calculate_grid_arrows(
        embedding=embedding, 
        delta_embedding=delta_embedding,
        offset_frac=grid_offset_frac,
        n_grid_cols=grid_n_cols,
        n_grid_rows=grid_n_rows,
        n_neighbors=grid_n_neighbors,
        n_cpu=n_cpu)

    log.info('Plotting')
    fig, ax = plt.subplots(figsize=figsize)
    ax = plot_metadata_given_ax(
        scplus_obj=scplus_obj,
        reduction_name=reduction_name,
        ax = ax,
        **kwargs
    )
    ax.quiver(grid_xy[mask, 0], grid_xy[mask, 1], uv[mask, 0], uv[mask, 1], scale = arrow_scale)
    if save is not None:
        fig.savefig(save)
    else:
        plt.show(fig)
        return ax
