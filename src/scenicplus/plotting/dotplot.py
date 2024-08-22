"""Plot TF expression, motif enrichment and AUC values of target genes and regions in a dotplot.

"""

import pandas as pd
from plotnine import (
    ggplot, geom_point, aes, scale_fill_distiller, 
    geom_tile, theme, element_text, element_blank)
from plotnine.facets import facet_grid
import plotnine
from typing import List, Union, Optional, Tuple
from mudata import MuData
from scenicplus.scenicplus_mudata import ScenicPlusMuData

def _scale(X: pd.DataFrame) -> pd.DataFrame:
    return (X - X.min()) / (X.max() - X.min())

def generate_dotplot_df(
    size_matrix: pd.DataFrame,
    color_matrix: pd.DataFrame,
    group_by: List[str],
    size_features: List[str], # size_features, color_features and feature_names the order should correspond, e.g. color_features, size_features, feature_names = scplus_mudata.uns["direct_e_regulon_metadata"][["Region_signature_name", "Gene_signature_name", "eRegulon_name"]].drop_duplicates().values.T
    color_features: List[str],
    feature_names: List[str],
    scale_size_matrix: bool = True,
    scale_color_matrix: bool = True,
    group_name: str = "group",
    size_name: str = "size_variable",
    color_name: str = "color_variable",
    feature_name: str = "eRegulon") -> pd.DataFrame:
    # Validate input
    if not all(size_matrix.index == color_matrix.index):
        raise ValueError("Both size_matrix and color_matrix should have the same index")
    if len(group_by) != len(size_matrix.index):
        raise ValueError("Length of group_by does not match with the index")
    if len(size_features) != len(color_features) != len(feature_names):
        raise ValueError("The length of 'size_features', 'color_features' and 'feature_names' should be equal!")
    # Subset and order for features
    size_matrix = size_matrix[size_features]
    color_matrix = color_matrix[color_features]
    # Calculate mean by group_by variable
    color_matrix_avg = color_matrix.groupby(group_by).mean()
    size_matrix_avg = size_matrix.groupby(group_by).mean()
    # Scale matrices
    color_matrix_avg = _scale(color_matrix_avg) if scale_color_matrix else color_matrix_avg
    size_matrix_avg = _scale(size_matrix_avg) if scale_size_matrix else size_matrix_avg
    # Transform dataframe into long format
    color_matrix_avg = color_matrix_avg.stack().reset_index()
    size_matrix_avg = size_matrix_avg.stack().reset_index()
    color_matrix_avg.columns = [group_name, "color_features", color_name]
    size_matrix_avg.columns = [group_name, "size_features", size_name]
    # map between color and size feature names to feature names
    color_features_to_name = dict(zip(color_features, feature_names))
    size_features_to_name = dict(zip(size_features, feature_names))
    # Add feature names to dataframe
    color_matrix_avg[feature_name] = [
        color_features_to_name[f] for f in color_matrix_avg["color_features"]]
    size_matrix_avg[feature_name] = [
        size_features_to_name[f] for f in size_matrix_avg["size_features"]]
    color_matrix_avg = color_matrix_avg.drop("color_features", axis = 1)
    size_matrix_avg = size_matrix_avg.drop("size_features", axis = 1)
    dotplot_df = color_matrix_avg.merge(
        size_matrix_avg,
        on = [group_name, feature_name])
    return dotplot_df

def heatmap_dotplot(
    scplus_mudata: Union[MuData, ScenicPlusMuData],
    size_modality: str,
    color_modality: str,
    group_variable: str,
    eRegulon_metadata_key: str,
    size_feature_key: str,
    color_feature_key: str,
    feature_name_key: str,
    sort_data_by: str,
    subset_feature_names: Optional[List[str]] = None,
    scale_size_matrix: bool = True,
    scale_color_matrix: bool = True,
    group_variable_order: Optional[List[str]] = None,
    save: Optional[str] = None,
    figsize: Tuple[float, float] = (5, 8),
    split_repressor_activator: bool = True,
    orientation: str = 'vertical'):
    # Generate dataframe for plotting
    size_matrix = scplus_mudata[size_modality].to_df()
    color_matrix = scplus_mudata[color_modality].to_df()
    group_by = scplus_mudata.obs[group_variable].tolist()
    if subset_feature_names is None:
        size_features, color_features, feature_names = scplus_mudata.uns[eRegulon_metadata_key][
            [size_feature_key, color_feature_key, feature_name_key]] \
            .drop_duplicates().values.T
    else:
        size_features, color_features, feature_names = scplus_mudata.uns[eRegulon_metadata_key][
            [size_feature_key, color_feature_key, feature_name_key]] \
            .drop_duplicates().query(f"{feature_name_key} in @subset_feature_names").values.T
    plotting_df = generate_dotplot_df(
        size_matrix=size_matrix,
        color_matrix=color_matrix,
        group_by=group_by,
        size_features=size_features,
        color_features=color_features,
        feature_names=feature_names,
        scale_size_matrix=scale_size_matrix,    
        scale_color_matrix=scale_color_matrix,
        group_name=group_variable,
        size_name=size_modality,
        color_name=color_modality,
        feature_name=feature_name_key)
    # Order data
    if group_variable_order is not None:
        if len(set(group_variable_order) & set(plotting_df[group_variable])) != len(set(plotting_df[group_variable])):
            Warning('not all indices are provided in index_order, order will not be changed!')
        else:
            plotting_df[group_variable] = pd.Categorical(plotting_df[group_variable], categories=group_variable_order)
    tmp = plotting_df[[group_variable, feature_name_key, sort_data_by]] \
            .pivot_table(index=group_variable, columns=feature_name_key) \
            .fillna(0)[sort_data_by]
    if group_variable_order is not None:
        tmp = tmp.loc[group_variable_order]
    idx_max = tmp.idxmax(axis = 0)
    order = pd.concat([idx_max[idx_max == x] for x in tmp.index.tolist() if len(plotting_df[plotting_df == x]) > 0]).index.tolist()
    plotting_df[feature_name_key] = pd.Categorical(plotting_df[feature_name_key], categories=order)
    # Plotting
    plotnine.options.figure_size = figsize
    plotting_df["repressor_activator"] = [
            "activator" if n.split("_")[2].split("/")[0] == "+" else "repressor" for n in plotting_df[feature_name_key]]
    if split_repressor_activator and len(set(plotting_df["repressor_activator"])) == 2:
        if orientation == 'vertical':
            plot = (
                ggplot(plotting_df, aes(group_variable, feature_name_key))
                + facet_grid(
                    'repressor_activator ~ .', 
                    scales = "free", 
                    space = {'x': [1], 'y': [sum(plotting_df['repressor_activator'] == 'activator'), sum(plotting_df['repressor_activator'] == 'repressor')]})
                + geom_tile(mapping = aes(fill = color_modality))
                + scale_fill_distiller(type = 'div', palette = 'RdYlBu')
                + geom_point(
                        mapping = aes(size = size_modality),
                        colour = "black")
                + theme(axis_text_x=element_text(rotation=90, hjust=1))
                + theme(axis_title_x = element_blank(), axis_title_y = element_blank()))
        elif orientation == 'horizontal':
            plot = (
                ggplot(plotting_df, aes(feature_name_key, group_variable))
                + facet_grid(
                    '. ~ repressor_activator', 
                    scales = "free", 
                    space = {'y': [1], 'x': [sum(plotting_df['repressor_activator'] == 'activator'), sum(plotting_df['repressor_activator'] == 'repressor')]})
                + geom_tile(mapping = aes(fill = color_modality))
                + scale_fill_distiller(type = 'div', palette = 'RdYlBu')
                + geom_point(
                        mapping = aes(size = size_modality),
                        colour = "black")
                + theme(axis_text_x=element_text(rotation=90, hjust=1))
                + theme(axis_title_x = element_blank(), axis_title_y = element_blank()))
    else:
        if orientation == 'vertical':
            plot = (
                ggplot(plotting_df, aes(group_variable, feature_name_key))
                + geom_tile(mapping = aes(fill = color_modality))
                + scale_fill_distiller(type = 'div', palette = 'RdYlBu')
                + geom_point(
                        mapping = aes(size = size_modality),
                        colour = "black")
                + theme(axis_title_x = element_blank(), axis_title_y = element_blank()))
        elif orientation == 'horizontal':
            plot = (
                ggplot(plotting_df, aes(feature_name_key, group_variable))
                + geom_tile(mapping = aes(fill = color_modality))
                + scale_fill_distiller(type = 'div', palette = 'RdYlBu')
                + geom_point(
                        mapping = aes(size = size_modality),
                        colour = "black")
                + theme(axis_title_x = element_blank(), axis_title_y = element_blank()))
    if save is not None:
        plot.save(save)
    else:
        return plot