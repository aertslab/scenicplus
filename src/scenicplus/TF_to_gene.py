"""Link transcription factors (TFs) to genes based on co-expression of TF and target genes.

Both linear methods (spearman or pearson correlation) and non-linear methods (random forrest or gradient boosting) are used to link TF to genes.

The correlation methods are used to seperate TFs which are infered to have a positive influence on gene expression (i.e. positive correlation) 
and TFs which are infered to have a negative influence on gene expression (i.e. negative correlation).

"""


import logging
import os
import sys
import joblib
import numpy as np
import pandas as pd
from arboreto.algo import _prepare_input
from arboreto.core import (EARLY_STOP_WINDOW_LENGTH, RF_KWARGS, SGBM_KWARGS,
                           infer_partial_network, to_tf_matrix)
from tqdm import tqdm
from scenicplus.utils import _create_idx_pairs, masked_rho4pairs
from typing import Literal, List, Union
import pathlib

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
    scplus_obj = None,
    TF2G_adj: pd.DataFrame = None, 
    ex_mtx: pd.DataFrame = None,
    rho_threshold = RHO_THRESHOLD, 
    TF2G_key = 'TF2G_adj', 
    out_key = 'TF2G_adj',
    inplace = True,
    increase_importance_by = 0.00001) -> Union[None, pd.DataFrame]:
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


def load_TF2G_adj_from_file(SCENICPLUS_obj,
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
    if COLUMN_NAME_CORRELATION not in df_TF_gene_adj_subset.columns:
        log.info('Adding correlation coefficients to adjacencies.')
        df_TF_gene_adj_subset = _add_correlation(
            adjacencies=df_TF_gene_adj_subset,
            ex_mtx=SCENICPLUS_obj.to_df(layer='EXP'),
            rho_threshold=rho_threshold)
    df_TF_gene_adj_subset = _inject_TF_as_its_own_target(
        TF2G_adj=df_TF_gene_adj_subset, 
        inplace = False, 
        ex_mtx = SCENICPLUS_obj.to_df(layer='EXP'))
    if COLUMN_NAME_SCORE_1 not in df_TF_gene_adj_subset.columns:
        log.info('Adding importance x rho scores to adjacencies.')
        df_TF_gene_adj_subset[COLUMN_NAME_SCORE_1] = df_TF_gene_adj_subset[COLUMN_NAME_CORRELATION] * \
            df_TF_gene_adj_subset[COLUMN_NAME_WEIGHT]
    if COLUMN_NAME_SCORE_2 not in df_TF_gene_adj_subset.columns:
        log.info('Adding importance x |rho| scores to adjacencies.')
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

def calculate_TFs_to_genes_relationships(
        df_exp_mtx: pd.DataFrame,
        tf_names: List[str],
        temp_dir: pathlib.Path,
        method: Literal['GBM', 'RF'] = 'GBM',
        n_cpu: int = 1,
        seed: int = 666) -> pd.DataFrame:
    """
    #TODO: Add docstrings
    """

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

    exp_mtx, gene_names, tf_names = _prepare_input(
        expression_data = df_exp_mtx, gene_names = None, tf_names = tf_names)
    tf_matrix, tf_matrix_gene_names = to_tf_matrix(
        exp_mtx,  gene_names, tf_names)
            
    log.info('Calculating TF-to-gene importance')
    if temp_dir is not None:
        if type(temp_dir) == str:
            temp_dir = pathlib.Path(temp_dir)
        if not temp_dir.exists():
            Warning(f"{temp_dir} does not exist, creating it.")
            os.makedirs(temp_dir)
        
    TF_to_genes = joblib.Parallel(
        n_jobs = n_cpu,
        temp_folder = temp_dir)(
            joblib.delayed(infer_partial_network)(
                target_gene_name = gene,
                target_gene_expression = exp_mtx[:, gene_names.index(gene)],
                regressor_type = method_params[0],
                regressor_kwargs = method_params[1],
                tf_matrix = tf_matrix,
                tf_matrix_gene_names = tf_matrix_gene_names,
                include_meta = False,
                early_stop_window_length = EARLY_STOP_WINDOW_LENGTH,
                seed = seed)
            for gene in tqdm(
                gene_names, 
                total=len(gene_names), 
                desc=f'Running using {n_cpu} cores'))

    adj = pd.concat(TF_to_genes).sort_values(by='importance', ascending=False)
    log.info('Adding correlation coefficients to adjacencies.')
    adj = _add_correlation(adj, df_exp_mtx)
    adj = _inject_TF_as_its_own_target(
        TF2G_adj=adj, 
        inplace = False, 
        ex_mtx = df_exp_mtx)
    return adj