"""Plot TF expression, motif enrichment and AUC values of target genes and regions in a dotplot.

"""

import pandas as pd
import numpy as np
from plotnine import (
    ggplot, geom_point, aes, scale_fill_distiller, 
    geom_tile, theme, element_text, element_blank)
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