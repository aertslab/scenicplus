"""Link transcription factors (TFs) to genes based on co-expression of TF and target genes.

Both linear methods (spearman or pearson correlation) and non-linear methods (random forrest or gradient boosting) are used to link TF to genes.

The correlation methods are used to seperate TFs which are infered to have a positive influence on gene expression (i.e. positive correlation) 
and TFs which are infered to have a negative influence on gene expression (i.e. negative correlation).

"""


import pandas as pd
import numpy as np
import ray
import logging
import time
import sys

from tqdm import tqdm

from .scenicplus_class import SCENICPLUS
from .utils import _create_idx_pairs, masked_rho4pairs

import scipy.sparse

COLUMN_NAME_TARGET = "target"
COLUMN_NAME_WEIGHT = "importance"
COLUMN_NAME_REGULATION = "regulation"
COLUMN_NAME_CORRELATION = "rho"
COLUMN_NAME_TF = "TF"
COLUMN_NAME_SCORE_1 = "importance_x_rho"
COLUMN_NAME_SCORE_2 = "importance_x_abs_rho"
RHO_THRESHOLD = 0.03

# Create logger
level = logging.INFO
format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
handlers = [logging.StreamHandler(stream=sys.stdout)]
logging.basicConfig(level=level, format=format, handlers=handlers)
log = logging.getLogger('TF2G')

def _inject_TF_as_its_own_target(
    scplus_obj: SCENICPLUS = None,
    TF2G_adj: pd.DataFrame = None, 
    ex_mtx: pd.DataFrame = None,
    rho_threshold = RHO_THRESHOLD, 
    TF2G_key = 'TF2G_adj', 
    out_key = 'TF2G_adj',
    inplace = True,
    increase_importance_by = 0.00001) -> pd.DataFrame:
    if scplus_obj is None and TF2G_adj is None:
        raise ValueError('Either provide a SCENIC+ object of a pd.DataFrame with TF to gene adjecencies!')
    if scplus_obj is not None and TF2G_adj is not None:
        raise ValueError('Either provide a SCENIC+ object of a pd.DataFrame with TF to gene adjecencies! Not both!')

    log.info(f"Warning: adding TFs as their own target to adjecencies matrix. Importance values will be max + {increase_importance_by}")
    
    origin_TF2G_adj = scplus_obj.uns[TF2G_key] if scplus_obj is not None else TF2G_adj
    ex_mtx = scplus_obj.to_df(layer='EXP') if scplus_obj is not None else ex_mtx

    origin_TF2G_adj = origin_TF2G_adj.sort_values('TF')
    max_importances = origin_TF2G_adj.groupby('TF').max()['importance']

    TFs_in_adj = list(set(origin_TF2G_adj['TF'].to_list()))
    TF_to_TF_adj = pd.DataFrame(
                    data = {"TF": TFs_in_adj,
                            "target": TFs_in_adj,
                            "importance": max_importances.loc[TFs_in_adj] + increase_importance_by})
    TF_to_TF_adj = _add_correlation(
            adjacencies=TF_to_TF_adj,
            ex_mtx = ex_mtx,
            rho_threshold=rho_threshold)

    new_TF2G_adj = pd.concat([origin_TF2G_adj, TF_to_TF_adj]).reset_index(drop = True)
    if inplace:
        scplus_obj.uns[out_key] = new_TF2G_adj
        return None
    else:
        return new_TF2G_adj


def load_TF2G_adj_from_file(SCENICPLUS_obj: SCENICPLUS,
                            f_adj: str,
                            inplace=True,
                            key='TF2G_adj',
                            rho_threshold=RHO_THRESHOLD):
    """
    Function to load TF2G adjacencies from file

    Parameters
    ----------
    SCENICPLUS_obj
        An instance of :class:`~scenicplus.scenicplus_class.SCENICPLUS`
    f_adj
        File path to TF2G adjacencies matrix
    inplace
        Boolean specifying wether or not to store adjacencies matrix in `SCENICPLUS_obj` under slot .uns[key].
        Default: True
    key_added
        String specifying where in the .uns slot to store the adjacencies matrix in `SCENICPLUS_obj`
        Default: "TF2G_adj"
    rho_threshold
        A floating point number specifying from which absolute value to consider a correlation coefficient positive or negative.
        Default: 0.03
    """
    log.info(f'Reading file: {f_adj}')
    df_TF_gene_adj = pd.read_csv(f_adj, sep='\t')
    # only keep relevant entries
    idx_to_keep = np.logical_and(np.array([tf in SCENICPLUS_obj.gene_names for tf in df_TF_gene_adj['TF']]),
                                 np.array([gene in SCENICPLUS_obj.gene_names for gene in df_TF_gene_adj['target']]))
    df_TF_gene_adj_subset = df_TF_gene_adj.loc[idx_to_keep]
    if not COLUMN_NAME_CORRELATION in df_TF_gene_adj_subset.columns:
        log.info(f'Adding correlation coefficients to adjacencies.')
        df_TF_gene_adj_subset = _add_correlation(
            adjacencies=df_TF_gene_adj_subset,
            ex_mtx=SCENICPLUS_obj.to_df(layer='EXP'),
            rho_threshold=rho_threshold)
    df_TF_gene_adj_subset = _inject_TF_as_its_own_target(
        TF2G_adj=df_TF_gene_adj_subset, 
        inplace = False, 
        ex_mtx = SCENICPLUS_obj.to_df(layer='EXP'))
    if not COLUMN_NAME_SCORE_1 in df_TF_gene_adj_subset.columns:
        log.info(f'Adding importance x rho scores to adjacencies.')
        df_TF_gene_adj_subset[COLUMN_NAME_SCORE_1] = df_TF_gene_adj_subset[COLUMN_NAME_CORRELATION] * \
            df_TF_gene_adj_subset[COLUMN_NAME_WEIGHT]
    if not COLUMN_NAME_SCORE_2 in df_TF_gene_adj_subset.columns:
        log.info(f'Adding importance x |rho| scores to adjacencies.')
        df_TF_gene_adj_subset[COLUMN_NAME_SCORE_2] = abs(
            df_TF_gene_adj_subset[COLUMN_NAME_CORRELATION]) * abs(df_TF_gene_adj_subset[COLUMN_NAME_WEIGHT])

    if inplace:
        log.info(f'Storing adjacencies in .uns["{key}"].')
        SCENICPLUS_obj.uns[key] = df_TF_gene_adj_subset
    else:
        return df_TF_gene_adj_subset


def _add_correlation(
        adjacencies: pd.DataFrame,
        ex_mtx: pd.DataFrame,
        rho_threshold=RHO_THRESHOLD,
        mask_dropouts=False):
    """
    Add correlation in expression levels between target and factor.

    Parameters
    ----------
    adjacencies: pd.DataFrame
        The dataframe with the TF-target links.
    ex_mtx: pd.DataFrame
        The expression matrix (n_cells x n_genes).
    rho_threshold: float
        The threshold on the correlation to decide if a target gene is activated
        (rho > `rho_threshold`) or repressed (rho < -`rho_threshold`).
    mask_dropouts: boolean
        Do not use cells in which either the expression of the TF or the target gene is 0 when
        calculating the correlation between a TF-target pair.

    Returns
    -------
        The adjacencies dataframe with an extra column.
    """
    assert rho_threshold > 0, "rho_threshold should be greater than 0."

    # Calculate Pearson correlation to infer repression or activation.
    if mask_dropouts:
        ex_mtx = ex_mtx.sort_index(axis=1)
        col_idx_pairs = _create_idx_pairs(adjacencies, ex_mtx)
        rhos = masked_rho4pairs(ex_mtx.values, col_idx_pairs, 0.0)
    else:
        genes = list(set(adjacencies[COLUMN_NAME_TF]).union(
            set(adjacencies[COLUMN_NAME_TARGET])))
        ex_mtx = ex_mtx[ex_mtx.columns[ex_mtx.columns.isin(genes)]]
        corr_mtx = pd.DataFrame(
            index=ex_mtx.columns, columns=ex_mtx.columns, data=np.corrcoef(ex_mtx.values.T))
        rhos = np.array([corr_mtx[s2][s1]
                        for s1, s2 in zip(adjacencies.TF, adjacencies.target)])

    regulations = (rhos > rho_threshold).astype(
        int) - (rhos < -rho_threshold).astype(int)
    return pd.DataFrame(
        data={
            COLUMN_NAME_TF: adjacencies[COLUMN_NAME_TF].values,
            COLUMN_NAME_TARGET: adjacencies[COLUMN_NAME_TARGET].values,
            COLUMN_NAME_WEIGHT: adjacencies[COLUMN_NAME_WEIGHT].values,
            COLUMN_NAME_REGULATION: regulations,
            COLUMN_NAME_CORRELATION: rhos,
        }
    )


def _run_infer_partial_network(target_gene_name,
                              gene_names,
                              ex_matrix,
                              method_params,
                              tf_matrix,
                              tf_matrix_gene_names):
    """
    A function to call arboreto 
    """
    from arboreto import core as arboreto_core

    target_gene_name_index = _get_position_index([target_gene_name], gene_names)
    target_gene_expression = ex_matrix[:, target_gene_name_index].ravel()

    n = arboreto_core.infer_partial_network(
        regressor_type=method_params[0],
        regressor_kwargs=method_params[1],
        tf_matrix=tf_matrix,
        tf_matrix_gene_names=tf_matrix_gene_names,
        target_gene_name=target_gene_name,
        target_gene_expression=target_gene_expression,
        include_meta=False,
        early_stop_window_length=arboreto_core.EARLY_STOP_WINDOW_LENGTH,
        seed=666)
    return(n)


@ray.remote
def _run_infer_partial_network_ray(target_gene_name,
                                  gene_names,
                                  ex_matrix,
                                  method_params,
                                  tf_matrix,
                                  tf_matrix_gene_names):
    """
    A function to call arboreto with ray
    """

    return _run_infer_partial_network(target_gene_name,
                                     gene_names,
                                     ex_matrix,
                                     method_params,
                                     tf_matrix,
                                     tf_matrix_gene_names)


def calculate_TFs_to_genes_relationships(scplus_obj: SCENICPLUS,
                                         tf_file: str,
                                         method: str = 'GBM',
                                         ray_n_cpu: int = 1,
                                         key: str = 'TF2G_adj',
                                         **kwargs):
    """
    A function to calculate TF to gene relationships using arboreto and correlation

    Parameters
    ----------
    scplus_obj
        An instance of :class:`~scenicplus.scenicplus_class.SCENICPLUS`
    tf_file
        Path to a file specifying with genes are TFs
    method
        Whether to use Gradient Boosting Machines (GBM) or random forest (RF)
    ray_n_cpu
        Number of cpus to use
    key
        String specifying where in the .uns slot to store the adjacencies matrix in :param:`SCENICPLUS_obj`
        default: "TF2G_adj"
    **kwargs
        Parameters to pass to ray.init
    """
    from arboreto.utils import load_tf_names
    from arboreto.algo import _prepare_input
    from arboreto.core import SGBM_KWARGS, RF_KWARGS
    from arboreto.core import to_tf_matrix

    if(method == 'GBM'):
        method_params = [
            'GBM',      # regressor_type
            SGBM_KWARGS  # regressor_kwargs
        ]
    elif(method == 'RF'):
        method_params = [
            'RF',       # regressor_type
            RF_KWARGS   # regressor_kwargs
        ]

    gene_names = scplus_obj.gene_names
    cell_names = scplus_obj.cell_names
    ex_matrix = scplus_obj.X_EXP

    tf_names = load_tf_names(tf_file)
    ex_matrix, gene_names, tf_names = _prepare_input(
        ex_matrix, gene_names, tf_names)
    tf_matrix, tf_matrix_gene_names = to_tf_matrix(
        ex_matrix, gene_names, tf_names)
    
    #convert ex_matrix, tf_matrix to np.array if necessary
    if isinstance(ex_matrix, np.matrix):
        ex_matrix = np.array(ex_matrix)
    elif scipy.sparse.issparse(ex_matrix):
        ex_matrix = ex_matrix.toarray()
        
    if isinstance(tf_matrix, np.matrix):
        tf_matrix = np.array(tf_matrix)
    elif scipy.sparse.issparse(tf_matrix):
       tf_matrix = tf_matrix.toarray()

    ray.init(num_cpus=ray_n_cpu, **kwargs)
    log.info(f'Calculating TF to gene correlation, using {method} method')
    start_time = time.time()
    try:
        jobs = []
        for gene in tqdm(gene_names, total=len(gene_names), desc='initializing'):
            jobs.append(_run_infer_partial_network_ray.remote(gene,
                                                             gene_names,
                                                             ex_matrix,
                                                             method_params,
                                                             tf_matrix,
                                                             tf_matrix_gene_names))
        # add progress bar, adapted from: https://github.com/ray-project/ray/issues/8164

        def to_iterator(obj_ids):
            while obj_ids:
                finished_ids, obj_ids = ray.wait(obj_ids)
                for finished_id in finished_ids:
                    yield ray.get(finished_id)
        tfs_to_genes = []
        for adj in tqdm(to_iterator(jobs),
                        total=len(jobs),
                        desc=f'Running using {ray_n_cpu} cores',
                        smoothing=0.1):
            tfs_to_genes.append(adj)
    except Exception as e:
        print(e)
    finally:
        ray.shutdown()

    log.info('Took {} seconds'.format(time.time() - start_time))
    start_time = time.time()
    log.info(f'Adding correlation coefficients to adjacencies.')
    adj = pd.concat(tfs_to_genes).sort_values(by='importance', ascending=False)
    ex_matrix = pd.DataFrame(
        scplus_obj.X_EXP, index=scplus_obj.cell_names, columns=scplus_obj.gene_names)
    adj = _add_correlation(adj, ex_matrix)
    adj = _inject_TF_as_its_own_target(
        TF2G_adj=adj, 
        inplace = False, 
        ex_mtx = scplus_obj.to_df(layer='EXP'))
    log.info(f'Adding importance x rho scores to adjacencies.')
    adj[COLUMN_NAME_SCORE_1] = adj[COLUMN_NAME_CORRELATION] * \
        adj[COLUMN_NAME_WEIGHT]
    adj[COLUMN_NAME_SCORE_2] = abs(
        adj[COLUMN_NAME_CORRELATION]) * abs(adj[COLUMN_NAME_WEIGHT])
    log.info('Took {} seconds'.format(time.time() - start_time))
    scplus_obj.uns[key] = adj


def _get_position_index(query_list, target_list):
    """
    Helper function to grep an instance in a list
    """
    d = {k: v for v, k in enumerate(target_list)}
    index = (d[k] for k in query_list)
    return list(index)
