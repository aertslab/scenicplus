# -*- coding: utf-8 -*-
"""Merging, scoring and assessing TF correlations of cistromes

"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from random import sample
import seaborn as sns
from scipy.stats import pearsonr
from typing import List
from scenicplus.utils import p_adjust_bh
from scenicplus.scenicplus_class import SCENICPLUS
import numpy as np
import pandas as pd


# Create pseudobulks
#TODO: fix this function, bug with nr of cells
def generate_pseudobulks(scplus_obj: SCENICPLUS,
                         variable: str,
                         normalize_expression: bool = True,
                         auc_key: str = 'Cistromes_AUC',
                         signature_key: str = 'Unfiltered',
                         nr_cells: int = 10,
                         nr_pseudobulks: int = 100,
                         seed: int = 555):
    """
    Generate pseudobulks based on the cistrome AUC matrix and gene expression

    Parameters
    ---------
    scplus_obj: :class:`SCENICPLUS`
        A :class:`SCENICPLUS` object with scored cistromes (`scplus_obj.uns['Cistromes_AUC'][cistromes_key]`)
    variable: str, optional
        Variable to create pseudobulks by. It must ve a column in `scplus_obj.metadata_cell`
    cistromes_key: str, optional
        Key to store where cistromes AUC values are stored. Cistromes AUC will retrieved from 
        `scplus_obj.uns['Cistromes_AUC'][cistromes_key]` and the pseudobulk matrix will be stored
        at `scplus_obj.uns['Pseudobulk']['Cistromes_AUC'][variable][cistromes_key]` and `scplus_obj.uns['Pseudobulk'][variable]['Expression']`
    nr_cells : int, optional
        Number of cells to include per pseudobulk.
    nr_pseudobulks: int, optional
        Number of pseudobulks to generate per class
    seed: int
        Seed to ensure that pseudobulk are reproducible.
    """
    cell_data = scplus_obj.metadata_cell
    cistromes_auc = scplus_obj.uns[auc_key][signature_key]
    cell_data = cell_data.loc[cistromes_auc.index, :]
    dgem = pd.DataFrame(
        scplus_obj.X_EXP, index=scplus_obj.cell_names, columns=scplus_obj.gene_names).copy()
    categories = list(set(cell_data.loc[:, variable]))
    cistrome_auc_agg_list = list()
    dgem_agg_list = list()
    cell_names = list()
    if normalize_expression:
        dgem = dgem.T / dgem.T.sum(0) * 10**6
        dgem = np.log1p(dgem).T
    for category in categories:
        cells = cell_data[cell_data.loc[:, variable]
                          == category].index.tolist()
        for x in range(nr_pseudobulks):
            random.seed(x)
            sample_cells = sample(cells, nr_cells) #here is the bug
            sub_dgem = dgem.loc[sample_cells, :].mean(axis=0)
            sub_auc = cistromes_auc.loc[sample_cells, :].mean(axis=0)
            cistrome_auc_agg_list.append(sub_auc)
            dgem_agg_list.append(sub_dgem)
            cell_names.append(category + '_' + str(x))
    cistrome_auc_agg = pd.concat(cistrome_auc_agg_list, axis=1)
    cistrome_auc_agg.columns = cell_names
    dgem_agg = pd.concat(dgem_agg_list, axis=1)
    dgem_agg.columns = cell_names
    if not 'Pseudobulk' in scplus_obj.uns.keys():
        scplus_obj.uns['Pseudobulk'] = {}
    if not variable in scplus_obj.uns['Pseudobulk'].keys():
        scplus_obj.uns['Pseudobulk'][variable] = {}
    scplus_obj.uns['Pseudobulk'][variable]['Expression'] = dgem_agg
    if not auc_key in scplus_obj.uns['Pseudobulk'][variable].keys():
        scplus_obj.uns['Pseudobulk'][variable][auc_key] = {}
    scplus_obj.uns['Pseudobulk'][variable][auc_key][signature_key] = cistrome_auc_agg

#TODO: fix multiple uses of pandas concat (generates a lot of warning)
def TF_cistrome_correlation(scplus_obj: SCENICPLUS,
                            variable: str = None,
                            use_pseudobulk: bool = True,
                            auc_key: str = 'Cistromes_AUC',
                            signature_key: str = 'Unfiltered',
                            out_key: str = 'Unfiltered',
                            subset: List[str] = None):
    """
    Get correlation between gene expression and cistrome accessibility

    Parameters
    ---------
    scplus_obj: :class:`SCENICPLUS`
        A :class:`SCENICPLUS` object with pseudobulk matrices (`scplus_obj.uns['Pseudobulk']`)
    variable: str, optional
        Variable used to create the pseudobulks. Must be a key in `scplus_obj.uns['Pseudobulk']`.
        Only required if use_pseudobulk is False.
    use_pseudobulk: bool, optional
        Whether to use pseudobulk matrix or actual values.
    cistromes_key: str, optional
        Key to retrieve the pseudobulk matrices.  Cistrome accessibility will be retrieved from
        `scplus_obj.uns['Pseudobulk'][variable]['Cistromes_AUC'][cistromes_key]` and 
        gene expression from `scplus_obj.uns['Pseudobulk'][variable]['Expression']`.
    out_key : str, optional
        Ouput key. Correlations will be stored at `scplus_obj.uns['TF_cistrome_correlation'][variable][out_key]`.
    subset: List, optional
        Subset of cells to be used to calculate correlations. Default: None (All)
    """
    if use_pseudobulk:
        dgem_agg = scplus_obj.uns['Pseudobulk'][variable]['Expression']
        cistromes_auc_agg = scplus_obj.uns['Pseudobulk'][variable][auc_key][signature_key]
    else:
        dgem_agg = pd.DataFrame(
            scplus_obj.X_EXP, index=scplus_obj.cell_names, columns=scplus_obj.gene_names).copy().T
        cistromes_auc_agg = scplus_obj.uns[auc_key][signature_key].copy().T

    if subset is not None:
        cell_data = pd.DataFrame([x.rsplit('_', 1)[0] for x in cistromes_auc_agg.columns],
                                 index=cistromes_auc_agg.columns).iloc[:, 0]
        subset_cells = cell_data[cell_data.isin(subset)].index.tolist()
        cistromes_auc_agg = cistromes_auc_agg.loc[:, subset_cells]
        dgem_agg = dgem_agg.loc[:, subset_cells]
    corr_df = pd.DataFrame(columns=['TF', 'Cistrome', 'Rho', 'P-value'])
    for tf in cistromes_auc_agg.index:
        # Handle _extended
        tf_rna = tf.split('_')[0]
        if tf_rna in dgem_agg.index:
            cistromes_auc_tf = cistromes_auc_agg.loc[tf, :]
            tf_expr = dgem_agg.loc[tf_rna, :]
            # Exception in case TF is only expressed in 1 cell
            # TFs expressed in few cells could be filtered too
            try:
                corr_1, _1 = pearsonr(tf_expr, cistromes_auc_tf)
                x = {'TF': tf_rna,
                     'Cistrome': tf,
                     'Rho': corr_1,
                     'P-value': _1}
                corr_df = corr_df.append(pd.DataFrame(
                    data=x, index=[0]), ignore_index=True)
            except:
                continue
    corr_df = corr_df.dropna()
    corr_df['Adjusted_p-value'] = p_adjust_bh(corr_df['P-value'])

    if not 'TF_cistrome_correlation' in scplus_obj.uns.keys():
        scplus_obj.uns['TF_cistrome_correlation'] = {}
    if not out_key in scplus_obj.uns['TF_cistrome_correlation'].keys():
        scplus_obj.uns['TF_cistrome_correlation'][out_key] = {}
    scplus_obj.uns['TF_cistrome_correlation'][out_key] = corr_df

#TODO: fix multiple uses of pandas concat (generates a lot of warning)
def eregulon_correlation(scplus_obj: SCENICPLUS,
                         auc_key: str = 'eRegulon_AUC',
                         signature_key1: str = 'Gene_based',
                         signature_key2: str = 'Region_based',
                         nSignif: int = 3,
                         out_key: str = 'Unfiltered',
                         subset_cellids: List[str] = None
                        ):
        """
        Get correlation between gene-based and region-based eRegulon AUC

        Parameters
        ---------
        scplus_obj: :class:`SCENICPLUS`
            A :class:`SCENICPLUS` object
        auc_key: str, optional
            Name of the AUC matrix used to calculate the correlation, normally 'eRegulon_AUC'. 
            Must be a key in `scplus_obj.uns.keys()`.
        signature_key1: str, optional
            Variable used to calculate the correlation, normally 'Gene_based'. 
            Must be a key in `scplus_obj.uns[auc_key].keys()`.
        signature_key2: str, optional
            Variable used to calculate the correlation, normally 'Region_based'. 
            Must be a key in `scplus_obj.uns[auc_key].keys()`.
        nSignif: str, optional
            Number of digits to save.
        out_key : str, optional
            Ouput key. Correlations will be stored at `scplus_obj.uns['eRegulon_correlation'][out_key]`.
        subset_cellids: List, optional
            Subset of cells to be used to calculate correlations. Default: None (All)
        """

        gene_auc = scplus_obj.uns[auc_key][signature_key1].copy().T
        region_auc = scplus_obj.uns[auc_key][signature_key2].copy().T

        if subset_cellids is not None:
            cell_data = pd.DataFrame([x.rsplit('_', 1)[0] for x in gene_auc.columns],
                                     index=gene_auc.columns).iloc[:, 0]
            subset_cells = cell_data[cell_data.isin(subset_cellids)].index.tolist()
            gene_auc = gene_auc.loc[:, subset_cells]
            region_auc = region_auc.loc[:, subset_cells]

        # cistrome naming includes number of genes/regions, so need to create matching names
        # x.rsplit('_', 1) splits at first _ from the right
        gene_auc['id_short'] = gene_auc.index.map(lambda x: x.rsplit('_', 1)[0])
        gene_auc['id_full'] = gene_auc.index
        gene_auc = gene_auc.set_index('id_short')

        region_auc['id_short'] = region_auc.index.map(lambda x: x.rsplit('_', 1)[0])
        region_auc['id_full'] = region_auc.index
        region_auc = region_auc.set_index('id_short')

        corr_df = pd.DataFrame(columns=['id', signature_key1, signature_key2, 'Rho', 'P-value'])

        for tf in gene_auc.index:
            # All TFs should match, but just in case
            if tf in region_auc.index:
                # record orginal cistrome name for results
                signature1_id = gene_auc.loc[tf, 'id_full']
                signature2_id = region_auc.loc[tf, 'id_full']
                # Exception in case TF is only expressed in 1 cell
                # TFs expressed in few cells could be filtered too
                try:
                    corr_1, _1 = pearsonr(gene_auc.loc[tf, gene_auc.columns != 'id_full'],
                                          region_auc.loc[tf, gene_auc.columns != 'id_full'])
                    x = {'id': tf,
                         signature_key1: signature1_id,
                         signature_key2: signature2_id,
                         'Rho': round(corr_1,nSignif),
                         'P-value': _1}
                    corr_df = pd.concat([corr_df,
                                         pd.DataFrame(data=x, index=[0])],
                                        ignore_index=True)
                except:
                    continue
        corr_df = corr_df.dropna()
        corr_df['Adjusted_pValue'] = p_adjust_bh(corr_df['P-value'])
        corr_df['Abs_rho'] = abs(corr_df['Rho'])
        corr_df.sort_values('Abs_rho', ascending=False, inplace=True)

        if not 'eRegulon_correlation' in scplus_obj.uns.keys():
            scplus_obj.uns['eRegulon_correlation'] = {}
        if not out_key in scplus_obj.uns['eRegulon_correlation'].keys():
            scplus_obj.uns['eRegulon_correlation'][out_key] = {}
        scplus_obj.uns['eRegulon_correlation'][out_key] = corr_df