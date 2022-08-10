"""Plot TF expression, motif enrichment and AUC values of target genes and regions in a dotplot.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.cluster import AgglomerativeClustering
from matplotlib.cm import ScalarMappable
import matplotlib.backends.backend_pdf
from typing import Dict, List, Tuple
from typing import Optional
import re
from pycistarget.motif_enrichment_dem import DEM
from plotnine import ggplot, geom_point, aes, scale_fill_distiller, theme_bw, geom_tile, theme, element_text, element_blank
from plotnine.facets import facet_grid
import plotnine


from ..scenicplus_class import SCENICPLUS
from ..utils import flatten_list

def generate_dotplot_df(
    scplus_obj: SCENICPLUS,
    size_matrix: pd.DataFrame,
    color_matrix: pd.DataFrame,
    scale_size_matrix: bool = True,
    scale_color_matrix: bool = True,
    group_variable: str = None,
    subset_eRegulons: list = None) -> pd.DataFrame:
    """
    Function to generate dotplot dataframe from cistrome AUC enrichment

    Parameters
    ----------
    scplus_obj: `class::SCENICPLUS`
        A :class:`SCENICPLUS` object.
    size_matrix: pd.DataFrame
        A pd.DataFrame containing values to plot using size scale.
    color_matrix
        A pd.DataFrame containing values to plot using color scale.
    scale_size_matrix: bool
        Scale size matrix between 0 and 1 along index.
    scale_color_matrix: bool
        Scale color matrix between 0 and 1 along index.
    group_variable: str:
        Variable by which to group cell barcodes by (needed if the index of size or color matrix are cells.)
    subset_eRegulons: List
        List of eRegulons to plot.
    Returns
    -------
    pd.DataFrame
    """
    # Handle matrices which have cell barcodes on one axis
    if all(np.isin(size_matrix.columns, scplus_obj.cell_names)):
        size_matrix = size_matrix.T
    if all(np.isin(size_matrix.index, scplus_obj.cell_names)):
        #calculate mean values
        if group_variable is None:
            raise ValueError('group_variable can not be None when size_matrix is a matrix with cell barcodes on one axis.')
        if group_variable not in scplus_obj.metadata_cell.columns:
            raise ValueError('group variable must be a column in scplus_obj.metadata_cell')
        size_matrix = size_matrix.groupby(scplus_obj.metadata_cell[group_variable]).mean()
    if all(np.isin(color_matrix.columns, scplus_obj.cell_names)):
        color_matrix = color_matrix.T
    if all(np.isin(color_matrix.index, scplus_obj.cell_names)):
        #calculate mean values
        if group_variable is None:
            raise ValueError('group_variable can not be None when color_matrix is a matrix with cell barcodes on one axis.')
        if group_variable not in scplus_obj.metadata_cell.columns:
            raise ValueError('group variable must be a column in scplus_obj.metadata_cell')
        color_matrix = color_matrix.groupby(scplus_obj.metadata_cell[group_variable]).mean()
    if not all(np.isin(size_matrix.index, color_matrix.index)):
        raise ValueError('size_matrix and color_matrix should have the same values as index.')
    size_matrix_features = (
        size_matrix.columns.to_list(),                      #full eRegulon name
        [f.split('_(')[0] for f in size_matrix.columns],    #eRegulon name without number of targets
        [f.split('_')[0] for f in size_matrix.columns])     #tf name
    color_matrix_features = (
        color_matrix.columns.to_list(),                      #full eRegulon name
        [f.split('_(')[0] for f in color_matrix.columns],    #eRegulon name without number of targets
        [f.split('_')[0] for f in color_matrix.columns])     #tf name

    #put features in same order
    #both matrices have eRegulons as features
    def _find_idx(l, e):
        return [i for i, v in enumerate(l) if v == e]
    if all(['_(' in x for x in size_matrix_features[0]]) and all(['_(' in x for x in color_matrix_features[0]]):
        if not all(np.isin(size_matrix_features[1], color_matrix_features[1])):
            raise ValueError('When two matrices are given with eRegulons as features then the names of these features (without number of targets) should match!')
        color_matrix = color_matrix.iloc[
            :, flatten_list([_find_idx(color_matrix_features[1], e) for e in size_matrix_features[1]])]
    #size matrix has eRegulons as features but color matrix not
    elif all(['_(' in x for x in size_matrix_features[0]]):
        color_matrix = color_matrix[size_matrix_features[2]]
    #color matrix has eRegulons as features but size matrix not
    elif all(['_(' in x for x in color_matrix_features[0]]):
        size_matrix = size_matrix[color_matrix_features[2]]
    #none of the matrices have eRegulons as features
    else:
        size_matrix = size_matrix[color_matrix_features[2]]
    
    if subset_eRegulons is not None:
        #change to TF names
        subset_eRegulons = [x.split('_')[0] for x in subset_eRegulons]
        size_matrix = size_matrix[[x for x in size_matrix if x.split('_')[0] in subset_eRegulons]]
        color_matrix = color_matrix[[x for x in color_matrix if x.split('_')[0] in subset_eRegulons]]
    
    if scale_size_matrix:
        size_matrix = (size_matrix - size_matrix.min()) / (size_matrix.max() - size_matrix.min())
    if scale_color_matrix:
        color_matrix = (color_matrix - color_matrix.min()) / (color_matrix.max() - color_matrix.min())
    
    size_matrix_df = size_matrix.stack().reset_index()
    color_matrix_df = color_matrix.stack().reset_index()
    size_matrix_df.columns = ['index', 'size_name', 'size_val']
    color_matrix_df.columns = ['index', 'color_name', 'color_val']
    if all(['_(' in x for x in size_matrix_features[0]]) and all(['_(' in x for x in color_matrix_features[0]]):
        size_matrix_df['eRegulon_name'] = [x.split('_(')[0] for x in size_matrix_df['size_name']]
        color_matrix_df['eRegulon_name'] = [x.split('_(')[0] for x in color_matrix_df['color_name']]
        merged_df = size_matrix_df.merge(color_matrix_df, on = ['index', 'eRegulon_name'])
        merged_df['TF'] = [x.split('_')[0] for x in merged_df['eRegulon_name']]
    else:
        size_matrix_df['TF'] = [x.split('_')[0] for x in size_matrix_df['size_name']]
        color_matrix_df['TF'] = [x.split('_')[0] for x in color_matrix_df['color_name']]
        merged_df = size_matrix_df.merge(color_matrix_df, on = ['index', 'TF'])
        if all(['_(' in x for x in size_matrix_features[0]]):
            merged_df['eRegulon_name'] = merged_df['size_name']
        elif all(['_(' in x for x in color_matrix_features[0]]):
            merged_df['eRegulon_name'] = merged_df['color_name']
        else:
            merged_df['eRegulon_name'] = merged_df['TF']

    #for esthetics
    merged_df = merged_df[['index', 'TF', 'eRegulon_name', 'size_name', 'color_name', 'size_val', 'color_val']]
    return merged_df


def heatmap_dotplot(
    scplus_obj: SCENICPLUS,
    size_matrix: pd.DataFrame,
    color_matrix: pd.DataFrame,
    scale_size_matrix: bool = True,
    scale_color_matrix: bool = True,
    group_variable: str = None,
    subset_eRegulons: list = None,
    sort_by: str = 'color_val',
    index_order: list = None,
    save: str = None,
    figsize: tuple = (5, 8),
    split_repressor_activator: bool = True,
    orientation: str = 'vertical'):
    """
    Function to generate dotplot dataframe from cistrome AUC enrichment

    Parameters
    ----------
    scplus_obj: `class::SCENICPLUS`
        A :class:`SCENICPLUS` object.
    size_matrix: pd.DataFrame
        A pd.DataFrame containing values to plot using size scale.
    color_matrix
        A pd.DataFrame containing values to plot using color scale.
    scale_size_matrix: bool
        Scale size matrix between 0 and 1 along index.
    scale_color_matrix: bool
        Scale color matrix between 0 and 1 along index.
    group_variable: str:
        Variable by which to group cell barcodes by (needed if the index of size or color matrix are cells.)
    subset_eRegulons: List
        List of eRegulons to plot.
    sort_by: str
        Sort by color_val or size_val.
    index_order: list
        Order of index to plot.
    figsize: tuple
        size of the figure (x, y).
    split_repressor_activator: bool
        Wether to split the plot on repressors/activators.
    orientation: str
        Plot in horizontal or vertical orientation
    """
    plotting_df = generate_dotplot_df(
        scplus_obj = scplus_obj,
        size_matrix = size_matrix,
        color_matrix = color_matrix,
        scale_size_matrix = scale_size_matrix,
        scale_color_matrix = scale_color_matrix,
        group_variable = group_variable,
        subset_eRegulons = subset_eRegulons)
    if index_order is not None:
        if len(set(index_order) & set(plotting_df['index'])) != len(set(plotting_df['index'])):
            Warning('not all indices are provided in index_order, order will not be changed!')
        else:
            plotting_df['index'] = pd.Categorical(plotting_df['index'], categories = index_order)
    #sort values
    tmp = plotting_df[['index', 'eRegulon_name', sort_by]
        ].pivot_table(index = 'index', columns = 'eRegulon_name'
        ).fillna(0)['color_val']
    if index_order is not None:
        tmp = tmp.loc[index_order]
    idx_max = tmp.idxmax(axis = 0)
    order = pd.concat([idx_max[idx_max == x] for x in tmp.index.tolist() if len(plotting_df[plotting_df == x]) > 0]).index.tolist()
    plotting_df['eRegulon_name'] = pd.Categorical(plotting_df['eRegulon_name'], categories = order)
    plotnine.options.figure_size = figsize
    if split_repressor_activator:
        plotting_df['repressor_activator'] = ['activator' if '+' in n.split('_')[1] and 'extended' not in n or '+' in n.split('_')[2] and 'extended' in n  else 'repressor' for n in plotting_df['eRegulon_name']]
        if orientation == 'vertical':
            plot = (
                ggplot(plotting_df, aes('index', 'eRegulon_name'))
                + facet_grid(
                    'repressor_activator ~ .', 
                    scales = "free", 
                    space = {'x': [1], 'y': [sum(plotting_df['repressor_activator'] == 'activator'), sum(plotting_df['repressor_activator'] == 'repressor')]})
                + geom_tile(mapping = aes(fill = 'color_val'))
                + scale_fill_distiller(type = 'div', palette = 'RdYlBu')
                + geom_point(
                        mapping = aes(size = 'size_val'),
                        colour = "black")
                + theme(axis_text_x=element_text(rotation=90, hjust=1))
                + theme(axis_title_x = element_blank(), axis_title_y = element_blank()))
        elif orientation == 'horizontal':
            plot = (
                ggplot(plotting_df, aes('eRegulon_name', 'index'))
                + facet_grid(
                    '. ~ repressor_activator', 
                    scales = "free", 
                    space = {'y': [1], 'x': [sum(plotting_df['repressor_activator'] == 'activator'), sum(plotting_df['repressor_activator'] == 'repressor')]})
                + geom_tile(mapping = aes(fill = 'color_val'))
                + scale_fill_distiller(type = 'div', palette = 'RdYlBu')
                + geom_point(
                        mapping = aes(size = 'size_val'),
                        colour = "black")
                + theme(axis_text_x=element_text(rotation=90, hjust=1))
                + theme(axis_title_x = element_blank(), axis_title_y = element_blank()))
    else:
        if orientation == 'vertical':
            plot = (
                ggplot(plotting_df, aes('index', 'eRegulon_name'))
                + geom_tile(mapping = aes(fill = 'color_val'))
                + scale_fill_distiller(type = 'div', palette = 'RdYlBu')
                + geom_point(
                        mapping = aes(size = 'size_val'),
                        colour = "black")
                + theme(axis_title_x = element_blank(), axis_title_y = element_blank()))
        elif orientation == 'horizontal':
            plot = (
                ggplot(plotting_df, aes('eRegulon_name', 'index'))
                + geom_tile(mapping = aes(fill = 'color_val'))
                + scale_fill_distiller(type = 'div', palette = 'RdYlBu')
                + geom_point(
                        mapping = aes(size = 'size_val'),
                        colour = "black")
                + theme(axis_title_x = element_blank(), axis_title_y = element_blank()))
    if save is not None:
        plot.save(save)
    else:
        return plot

# Utils
def _flatten(A):
    """
    Utils function to flatten lists
    """
    rt = []
    for i in A:
        if isinstance(i, list):
            rt.extend(_flatten(i))
        else:
            rt.append(i)
    return rt

# For motif enrichment


def generate_dotplot_df_motif_enrichment(scplus_obj: SCENICPLUS,
                                         enrichment_key: str,
                                         group_variable: str = None,
                                         barcode_groups: dict = None,
                                         subset: list = None,
                                         subset_TFs: list = None,
                                         use_pseudobulk: bool = False,
                                         use_only_direct: bool = False,
                                         normalize_expression: bool = True,
                                         standardize_expression: bool = False):
    """
    DEPRECATED Function to generate dotplot dataframe from motif enrichment results

    Parameters
    ---------
    scplus_obj: `class::SCENICPLUS`
        A :class:`SCENICPLUS` object with motif enrichment results from pycistarget (`scplus_obj.menr`).
    enrichment_key: str
        Key of the motif enrichment result to use.
    group_variable: str, optional
        Group variable to use to calculate TF expression per group. Levels of this variable should match with the entries in the
        selected motif enrichment dictionary. Only required if barcode groups is not provided.
    barcode_groups: Dict, optional
        A dictionary containing cell barcodes per class to calculate TF expression per group. Keys of the dictionary should match
        the keys in the selected motif enrichment dictionary. Only required if group_variable is not provided.
    subset: list, optional
        Subset of classes to use. Default: None (use all)
    subset_TFs: list, optional
        List of TFs to use. Default: None (use all)
    use_pseudobulk: bool, optional
        Whether to use pseudobulk to calculate gene expression. Only available if there is pseudobulk profiles for group_variable.
        Default: False
    use_only_direct: bool, optional
        Whether to only use NES values of directly annotated motifs. Default: False
    normalize_expression: bool, optional
        Whether to log(CPM) normalize gene expression. Default: True
    standardize_expression: bool, optional
        Wheter to standarize gene expression between o and 1. Default: False

    Returns
    ---------
        A data frame with enrichment values per TF and cell group to use as input for `dotplot()`.
    """
    # Get data
    # Gene expression
    # If using pseudobulk
    if use_pseudobulk:
        dgem = scplus_obj.uns['Pseudobulk'][group_variable]['Expression'].copy(
        )
        if group_variable is not None:
            cell_data = pd.DataFrame([x.rsplit('_', 1)[0] for x in dgem.columns],
                                     index=dgem.columns).iloc[:, 0]
    # If using all
    else:
        dgem = pd.DataFrame(scplus_obj.X_EXP, index=scplus_obj.cell_names,
                            columns=scplus_obj.gene_names).copy().T
        if group_variable is not None:
            cell_data = scplus_obj.metadata_cell.loc[scplus_obj.cell_names, group_variable]
    # Should gene expression be normalized?
    if normalize_expression:
        dgem = dgem.T / dgem.T.sum(0) * 10**6
        dgem = np.log1p(dgem).T
    # Checking motif enrichment data
    menr = scplus_obj.menr[enrichment_key]
    if isinstance(menr, DEM):
        menr = scplus_obj.menr[enrichment_key].motif_enrichment.copy()
        menr_df = pd.concat([menr[x] for x in menr.keys()])
        score_keys = ['Log2FC', 'Adjusted_pval']
        columns = _flatten(['Contrast', score_keys])
    else:
        menr = scplus_obj.menr[enrichment_key].copy()
        menr_df = pd.concat([menr[x].motif_enrichment for x in menr.keys()])
        score_keys = ['NES']
        columns = _flatten(['Region_set', score_keys])

    if use_only_direct == True:
        columns = columns + 'Direct_annot'
        menr_df = menr_df[columns]
        menr_df.columns = _flatten(['Region_set', score_keys, 'Direct_annot'])
        menr_df['TF'] = menr_df['Direct_annot']
        menr_df = menr_df.drop(['Direct_annot'])
    else:
        annot_columns = list(filter(lambda x: 'annot' in x, menr_df.columns))
        columns = columns + annot_columns
        menr_df = menr_df[columns]
        menr_df.columns = _flatten(['Region_set', score_keys, annot_columns])
        for column in annot_columns:
            menr_df[column] = menr_df[column].str.split(', ')

        menr_df['TF'] = [menr_df[annot_columns[0]] + menr_df[col]
                         for col in annot_columns[1:]][0]
        menr_df = menr_df[_flatten(['TF', 'Region_set', score_keys])]
        menr_df.columns = _flatten(['TF', 'Group', score_keys])
        menr_df = menr_df.explode('TF')
        menr_df = menr_df.groupby(['TF', 'Group']).max().reset_index()

    # Check for cistrome subsets
    if subset_TFs is None:
        subset_TFs = list(set(menr_df['TF']))
    # Check that cistromes are in the AUC matrix
    subset_TFs = set(subset_TFs).intersection(menr_df['TF'].tolist())
    subset_TFs = set(subset_TFs).intersection(dgem.index.tolist())
    # Subset matrices
    # By cistrome
    tf_expr = dgem.loc[subset_TFs, :]
    menr_df = menr_df.loc[menr_df['TF'].isin(subset_TFs), :]
    # By cells
    if subset is not None:
        if group_variable is not None:
            subset_cells = cell_data[cell_data.isin(subset)].index.tolist()
            cell_data = cell_data.loc[subset_cells]
        if barcode_groups is not None:
            barcode_groups = {x: barcode_groups[x] for x in subset}
            subset_cells = list(
                set(sum([barcode_groups[x] for x in subset], [])))
        tf_expr = tf_expr.loc[:, tf_expr.columns.isin(subset_cells)]
    # Take barcode groups per variable level
    if barcode_groups is not None:
        levels = sorted(list(barcode_groups.keys()))
        barcode_groups = [barcode_groups[group] for group in levels]
    if group_variable is not None:
        levels = sorted(list(set(cell_data.tolist())))
        barcode_groups = [cell_data[cell_data.isin(
            [group])].index.tolist() for group in levels]
    # Calculate mean expression
    tf_expr_mean = pd.concat([tf_expr.loc[:, tf_expr.columns.isin(
        barcodes)].mean(axis=1) for barcodes in barcode_groups], axis=1)
    tf_expr_mean.columns = levels
    # Scale
    if standardize_expression:
        tf_expr_mean = tf_expr_mean.T
        tf_expr_mean = (tf_expr_mean-tf_expr_mean.min()+0.00000001) / \
            (tf_expr_mean.max()-tf_expr_mean.min()+0.00000001)
        tf_expr_mean = tf_expr_mean.T
    tf_expr_mean = tf_expr_mean.stack().reset_index()
    tf_expr_mean.columns = ['TF', 'Group', 'TF_expression']
    menr_df = menr_df[menr_df['Group'].isin(set(tf_expr_mean['Group']))]
    # Merge by column
    dotplot_df = pd.merge(tf_expr_mean, menr_df,
                          how='outer', on=['TF', 'Group'])
    dotplot_df = dotplot_df.replace(np.nan, 0)
    return dotplot_df

def generate_dotplot_df_AUC(scplus_obj: SCENICPLUS,
                            auc_key: str,
                            enrichment_key: str,
                            group_variable: str,
                            subset_cells: list = None,
                            subset_eRegulons: list = None,
                            use_pseudobulk: bool = False,
                            normalize_expression: bool = True,
                            standardize_expression: bool = False,
                            standardize_auc: bool = False):
    """
    DEPRECATED Function to generate dotplot dataframe from cistrome AUC enrichment

    Parameters
    ---------
    scplus_obj: `class::SCENICPLUS`
        A :class:`SCENICPLUS` object with motif enrichment results from pycistarget (`scplus_obj.menr`).
    enrichment_key: str
        Key of the motif enrichment result to use.
    group_variable: str
        Group variable to use to calculate TF expression per group. Levels of this variable should match with the entries in the
        selected motif enrichment dictionary. 
    subset_cells: list, optional
        Subset of classes to use. Default: None (use all)
    subset_eRegulons: list, optional
        List of cistromes to use. Default: None (use all)
    use_pseudobulk: bool, optional
        Whether to use pseudobulk to calculate gene expression. Only available if there is pseudobulk profiles for group_variable.
        Default: False
    normalize_expression: bool, optional
        Whether to log(CPM) normalize gene expression. Default: True
    standardize_expression: bool, optional
        Wheter to standarize gene expression between o and 1. Default: False
    standardize_auc: bool, optional
        Wheter to standarize AUC values between o and 1. Default: False

    Returns
    ---------
        A data frame with enrichment values per TF and cell group to use as input for `dotplot()`.
    """
    # Get data
    # If using pseudobulk
    if use_pseudobulk:
        dgem = scplus_obj.uns['Pseudobulk'][group_variable]['Expression'].copy(
        )
        cistromes_auc = scplus_obj.uns['Pseudobulk'][group_variable][auc_key][enrichment_key].copy(
        )
        cell_data = pd.DataFrame([x.rsplit('_', 1)[0] for x in cistromes_auc.columns],
                                 index=cistromes_auc.columns).iloc[:, 0]
    # If using all
    else:
        dgem = pd.DataFrame(scplus_obj.X_EXP, index=scplus_obj.cell_names,
                            columns=scplus_obj.gene_names).copy().T
        cistromes_auc = scplus_obj.uns[auc_key][enrichment_key].copy().T
        cell_data = scplus_obj.metadata_cell.loc[cistromes_auc.columns, group_variable]
    # Should gene expression be normalized?
    if normalize_expression:
        dgem = dgem.T / dgem.T.sum(0) * 10**6
        dgem = np.log1p(dgem).T
    # Check for cistrome subsets
    if subset_eRegulons is None:
        subset_eRegulons = cistromes_auc.index.tolist()
    # Check that cistromes are in the AUC matrix
    subset_eRegulons = set(subset_eRegulons).intersection(
        cistromes_auc.index.tolist())
    # Take TF names
    subset_tfs = [re.sub('_(.*)', '', (re.sub('_extended', '', cistrome_name)))
                  for cistrome_name in subset_eRegulons]
    check_df = pd.DataFrame([subset_eRegulons, subset_tfs]).T
    check_df.columns = ['Subset_cistromes', 'Subset_TFs']
    check_df = check_df[check_df['Subset_TFs'].isin(dgem.index.tolist())]
    subset_tfs = list(set(check_df['Subset_TFs'].tolist()))
    subset_eRegulons = list(set(check_df['Subset_cistromes'].tolist()))
    # Subset matrices
    # By cistrome
    tf_expr = dgem.loc[subset_tfs, :]
    cistromes_auc_tf = cistromes_auc.loc[subset_eRegulons, :]
    # By cells
    if subset_cells is not None:
        subset_cells = cell_data[cell_data.isin(subset_cells)].index.tolist()
        cell_data = cell_data.loc[subset_cells]
        tf_expr = tf_expr.loc[:, subset_cells]
        cistromes_auc_tf = cistromes_auc_tf.loc[:, subset_cells]
    # Take barcode groups per variable level
    levels = sorted(list(set(cell_data.tolist())))
    barcode_groups = [cell_data[cell_data.isin(
        [group])].index.tolist() for group in levels]
    # Calculate mean expression
    tf_expr_mean = pd.concat([tf_expr.loc[:, barcodes].mean(
        axis=1) for barcodes in barcode_groups], axis=1)
    tf_expr_mean.columns = levels
    # Scale
    if standardize_expression:
        tf_expr_mean = tf_expr_mean.T
        tf_expr_mean = (tf_expr_mean-tf_expr_mean.min()+0.00000001) / \
            (tf_expr_mean.max()-tf_expr_mean.min())
        tf_expr_mean = tf_expr_mean.T
    tf_expr_mean = tf_expr_mean.stack().reset_index()
    tf_expr_mean.columns = ['TF', 'Group', 'TF_expression']
    # Calculate mean AUC
    cistromes_auc_tf_mean = pd.concat([cistromes_auc_tf.loc[:, barcodes].mean(
        axis=1) for barcodes in barcode_groups], axis=1)
    cistromes_auc_tf_mean.columns = levels
    if standardize_auc:
        cistromes_auc_tf_mean = cistromes_auc_tf_mean.T
        cistromes_auc_tf_mean = (cistromes_auc_tf_mean-cistromes_auc_tf_mean.min(
        )+0.00000001)/(cistromes_auc_tf_mean.max()-cistromes_auc_tf_mean.min())
        cistromes_auc_tf_mean = cistromes_auc_tf_mean.T
    cistromes_auc_tf_mean = cistromes_auc_tf_mean.stack().reset_index()
    cistromes_auc_tf_mean.columns = ['Name', 'Group', auc_key]
    cistromes_auc_tf_mean['TF'] = cistromes_auc_tf_mean['Name'].replace(
        '_.*', '', regex=True)
    # Merge by column
    dotplot_df = pd.merge(tf_expr_mean, cistromes_auc_tf_mean,
                          how='outer', on=['TF', 'Group'])
    return dotplot_df


def _cluster_labels_to_idx(labels):
    """
    A helper function to convert cluster labels to idx
    """
    counter = 0
    order = np.zeros(labels.shape, dtype=int)
    for i in range(0, labels.max() + 1):
        for j in np.where(labels == i)[0]:
            order[j] = counter
            counter += 1
    idx = [np.where(order == i)[0][0] for i, _ in enumerate(order)]
    return idx


def dotplot(df_dotplot: pd.DataFrame,
            ax: plt.axes = None,
            region_set_key: str = 'Name',
            size_var: str = 'TF_expression',
            color_var: str = 'Cistrome_AUC',
            order_group: list = None,
            order_cistromes: list = None,
            order_cistromes_by_max: str = 'TF_expression',
            cluster: str = None,  # can be group, TF or both
            n_clust: int = 2,
            min_point_size: float = 3,
            max_point_size: float = 30,
            cmap: str = 'viridis',
            vmin: float = 0,
            vmax: float = None,
            x_tick_rotation: float = 45,
            x_tick_ha: str = 'right',
            fontsize: float = 9,
            z_score_expr: bool = True,
            z_score_enr: bool = True,
            grid_color='grey',
            grid_lw=0.5,
            highlight=None,
            highlight_lw=1,
            highlight_lc='black',
            figsize: Optional[Tuple[float, float]] = (10, 10),
            use_plotly=True,
            plotly_height=1000,
            save=None):
    """
    DEPRECATED Function to generate dotplot

    Parameters
    ---------
    df_dotplot: pd.DataFrame
        A pd.DataFrame with a column Group and varaibles to use for the dotplot
    ax: plt.axes
        An ax object, only require if use_plotly == False.
    enrichment_variable: str, optional
        Variable to use for motif/TF enrichment. Default: 'Cistrome_AUC;
    region_set_key: str, optional
        Name of the column where TF/cistromes are. Default: 'Cistrome'
    size_var: str, optional
        Variable in df_dotplot to code the dot size with. Default: 'Cistrome_AUC'
    color_var: str, optional
        Variable in df_dotplot to code the dot color with. Default: 'Cistrome_AUC'
    order_group: list, optional
        Order to plot the groups by. Default: None (given)
    order_cistromes: list, optional
        Order to plot the cistromes by. Default: None (given)
    order_cistromes_by_max: str, optional
        Variable to order the cistromes with by values in group, to form a diagonal. Default: None
    cluster: str, optional
        Whether to cluster cistromes (or TFs)/groups. It will be overwritten if order_cistromes/order_group/order_cistromes_by_max
        are specified. 
    n_clust: int, optional
        Number of clusters to form if cluster is specified. Default: 2
    min_point_size: int, optional
        Minimal point size. Only available with Cistrome AUC values.
    max_point_size: int, optional
        Maximum point size. Only available with Cistrome AUC values.
    cmap: str, optional
        Color map to use.
    vmin: int, optional
        Minimal color value.
    vmax: int, optional
        Maximum color value
    x_tick_rotation: int, optional
        Rotation of the x-axis label. Only if `use_plotly = False`
    fontsize: int, optional
        Labels fontsize
    z_score_expr: bool, optional
        Whether to z-normalize expression values
    z_score_enr: bool, optional
        Whether to z-normalize enrichment values
    grid_color: str, optional
        Color for grid. Only if `use_plotly = False`
    grid_lw: float, optional
        Line width of grid. Only if `use_plotly = False`
    highlight: list, optional
        Whether to highlight specific TFs/cistromes. Only if `use_plotly = False`
    highlight_lw: float, optional
        Line width of highlight. Only if `use_plotly = False`
    highlight_lc: str, optional
        Color of highlight. Only if `use_plotly = False`
    figsize: tuple, optional
        Figure size. Only if `use_plotly = False`
    use_plotly: bool, optional
        Whether to use plotly for plotting. Default: True
    plotly_height: int, optional
        Height of the plotly plot. Width will be adjusted accordingly
    save: str, optional
        Path to save dotplot as file
    """

    # Seperate dotsize data (motif_enrichment) and dot color data (mean expr)
    dotsizes = df_dotplot[['Group', region_set_key, size_var]].pivot_table(
        index='Group', columns=region_set_key).fillna(0)[size_var]
    dotcolors = df_dotplot[['Group', region_set_key, color_var]].pivot_table(
        index='Group', columns=region_set_key).fillna(0)[color_var]

    category_orders = {}

    # Cluster
    if cluster == 'both' or cluster == 'group':
        clustering_groups = AgglomerativeClustering(
            n_clusters=n_clust).fit(dotsizes)
        ordered_idx_grps = _cluster_labels_to_idx(clustering_groups.labels_)
        dotsizes = dotsizes.iloc[ordered_idx_grps, :]
        dotcolors = dotcolors.iloc[ordered_idx_grps, :]
        category_orders['Group'] = dotcolors.index.tolist()
    if cluster == 'both' or cluster == 'TF':
        clustering_TFs = AgglomerativeClustering(
            n_clusters=n_clust).fit(dotsizes.T)
        ordered_idx_TFs = _cluster_labels_to_idx(clustering_TFs.labels_)
        dotsizes = dotsizes.iloc[:, ordered_idx_TFs]
        dotcolors = dotcolors.iloc[:, ordered_idx_TFs]
        category_orders[region_set_key] = dotsizes.columns

    # If order TFs/cistromes by max
    if order_cistromes_by_max == color_var:
        df = dotcolors.idxmax(axis=0)
        order_cistromes = pd.concat([df[df == x] for x in dotcolors.index.tolist(
        ) if len(df[df == x]) > 0]).index.tolist()
        dotsizes = dotsizes.loc[:, order_cistromes[::-1]]
        dotcolors = dotcolors.loc[:, order_cistromes[::-1]]
        category_orders[region_set_key] = order_cistromes

    if order_cistromes_by_max == size_var:
        df = dotsizes.idxmax(axis=0)
        order_cistromes = pd.concat(
            [df[df == x] for x in dotsizes.index.tolist() if len(df[df == x]) > 0]).index.tolist()
        dotsizes = dotsizes.loc[:, order_cistromes[::-1]]
        dotcolors = dotcolors.loc[:, order_cistromes[::-1]]
        category_orders[region_set_key] = order_cistromes

    # Order by given order
    if order_group is not None:
        dotsizes = dotsizes.loc[order_group[::-1], :]
        dotcolors = dotcolors.loc[order_group[::-1], :]
        category_orders['Group'] = order_group
    if order_cistromes is not None:
        dotsizes = dotsizes.loc[:, order_cistromes[::-1]]
        dotcolors = dotcolors.loc[:, order_cistromes[::-1]]
        category_orders[region_set_key] = order_cistromes

    # Scale dotsizes
    s = dotsizes.to_numpy().flatten('F')
    s_min = (s[s != 0]).min()
    s_max = s.max()

    if not use_plotly:
        if save is not None:
            pdf = matplotlib.backends.backend_pdf.PdfPages(save)
        fig = plt.figure(figsize=figsize)
        # Calculate Z-score for expression values
        if z_score_expr:
            u = dotcolors.mean()
            sd = dotcolors.std()
            dotcolors = (dotcolors - u) / sd

        if vmin is None:
            vmin = dotcolors.min().min()

        if z_score_enr:
            u = dotsizes.mean()
            sd = dotsizes.std()
            dotsizes = (dotsizes - u) / sd

        # Generate plotting data
        n_group = len(set(df_dotplot['Group']))
        n_TF = len(set(df_dotplot[region_set_key]))

        # Generate a grid
        x = np.tile(np.arange(n_group), n_TF)
        y = [int(i / n_group) for i, _ in enumerate(x)]

        # Scale values between min_point_size and max_point_size keep zero values at 0
        s[s != 0] = min_point_size + \
            (s[s != 0] - s_min) * \
            ((max_point_size - min_point_size) / (s_max - s_min))

        # Get dot colors
        c = dotcolors.to_numpy().flatten('F')

        # Get edges for eRegulons
        if highlight is not None:
            TFs = dotsizes.columns
            groups = dotsizes.index
            linewidths = [highlight_lw if TF in highlight else 0 for
                          TF in np.repeat(TFs, n_group)]
        else:
            linewidths = None

        if vmax is None:
            vmax = c.max()

        norm = Normalize(vmin=vmin, vmax=vmax)

        ax.set_axisbelow(True)
        ax.grid(color=grid_color, linewidth=grid_lw)
        scat = ax.scatter(x, y, s=s, c=c, cmap=cmap, norm=norm,
                          edgecolors=highlight_lc, linewidths=linewidths)
        # set x ticks
        ax.set_xticks(np.arange(n_group))
        ax.set_xticklabels(dotsizes.index, rotation=x_tick_rotation,
                           ha=x_tick_ha, fontdict={'fontsize': fontsize})
        # set y ticks
        ax.set_yticks(np.arange(n_TF))
        ax.set_yticklabels(dotsizes.columns, fontdict={'fontsize': fontsize})

        # Draw colorbar
        cbar = plt.colorbar(mappable=ScalarMappable(norm=norm, cmap=cmap),
                            ax=ax, location='bottom', orientation='horizontal',
                            aspect=10, shrink=0.4, pad=0.10, anchor=(0, 0))
        cbar_label = color_var
        cbar.set_label(cbar_label)

        L = plt.legend(*scat.legend_elements("sizes", num=3),
                       loc='lower right', bbox_to_anchor=(0.8, -0.2), frameon=False, title=size_var, ncol=1, mode=None)
        # Recalculate original scale
        def to_int(x): return int(''.join(i for i in x if i.isdigit()))
        labels = np.array([to_int(t.get_text()) for t in L.get_texts()])

        def re_scale(x): return (x - min_point_size) * \
            ((s_max - s_min) / (max_point_size - min_point_size)) + s_min
        rescaled = re_scale(labels).round(2)
        for new, text in zip(rescaled, L.get_texts()):
            text.set_text(new)

        if save is not None:
            fig.save(save, bbox_inches='tight')
        plt.show()

    else:
        import plotly.express as px
        df = df_dotplot.copy()
        if min_point_size != 0 and size_var != 'NES' and size_var != 'LogFC':
            df[size_var][df[size_var] != 0] = min_point_size + \
                (df[size_var][df[size_var] != 0] - s_min) * \
                ((max_point_size - min_point_size) / (s_max - s_min))
        fig = px.scatter(df, y=region_set_key, x="Group", color=color_var, size=size_var,
                         size_max=max_point_size, category_orders=category_orders, color_continuous_scale=cmap)

        fig.update_layout(
            height=plotly_height,  # Added parameter
            legend={'itemsizing': 'trace'},
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(tickfont=dict(size=fontsize))
        )
        if save is not None:
            fig.write_image(save)
        fig.show()
