"""Dimmensionality reduction and clustering based on target genes and target regions AUC.

"""

import pandas as pd
import umap
import sklearn
import matplotlib.backends.backend_pdf
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import random
import numpy as np
from adjustText import adjust_text
import igraph as ig
import leidenalg as la
from sklearn.neighbors import kneighbors_graph
from typing import Dict, List, Tuple
from typing import Optional, Union
from sklearn.decomposition import PCA
import harmonypy as hm

from .scenicplus_class import SCENICPLUS

def harmony(scplus_obj: SCENICPLUS,
            variable: str,
            auc_key: str = 'eRegulon_AUC',
            out_key: str = 'eRegulon_AUC_harmony',
            signature_keys : List[str] = ['Gene_based', 'Region_based'],
            scale: Optional[bool] = True,
            random_state: Optional[int] = 555,
            **kwargs):
    """
    Apply harmony batch effect correction (Korsunsky et al, 2019) over eRegulon AUC values
    Parameters
    ---------
    scplus_obj: `class::SCENICPLUS`
        A SCENICPLUS object with eRegulon AUC values calculated.
    variable: str
        Variable in scplus.metadata_cell to correct by.
    auc_key: str, optional
        Key where AUC values are stored.
    out_key: str, optional
        Key where corrected eRegulon values will be output
    signature_keys: list, optional
        Whether to scale probability matrix prior to correction. Default: True
    scale: bool, optional
        Whether to scale the AUC values
    random_state: int, optional
        Random seed used to use with harmony. Default: 555
    References
    ---------
    Korsunsky, I., Millard, N., Fan, J., Slowikowski, K., Zhang, F., Wei, K., ... & Raychaudhuri, S. (2019). Fast, sensitive and accurate integration of
    single-cell data with Harmony. Nature methods, 16(12), 1289-1296.
    """

    for key in signature_keys:
        cell_data = scplus_obj.metadata_cell.copy()
        data = scplus_obj.uns[auc_key][key].copy().T
        if scale:
            data = pd.DataFrame(sklearn.preprocessing.StandardScaler().fit_transform(
                data), index=data.index.to_list(), columns=data.columns)
        data_np = data.transpose().to_numpy()
        ho = hm.run_harmony(
            data_np,
            cell_data,
            variable,
            random_state=random_state,
            **kwargs)
        data_harmony = pd.DataFrame(
            ho.Z_corr,
            index=data.index.to_list(),
            columns=data.columns).T
        
        if out_key not in scplus_obj.uns.keys():
            scplus_obj.uns[out_key]={}
        scplus_obj.uns[out_key][key] = data_harmony


def find_clusters(scplus_obj: SCENICPLUS,
                  auc_key: Optional[str] = 'eRegulon_AUC',
                  signature_keys: Optional[List[str]] = ['Gene_based', 'Region_based'],
                  k: Optional[int] = 10,
                  res: Optional[List[float]] = [0.6],
                  seed: Optional[int] = 555,
                  scale: Optional[bool] = True,
                  prefix: Optional[str] = '',
                  selected_regulons: Optional[List[int]] = None,
                  selected_cells: Optional[List[str]] = None,
                  **kwargs):
    """
    Performing leiden cell or region clustering and add results to SCENICPLUS object's metadata.

    Parameters
    ---------
    scplus_obj: `class::SCENICPLUS`
            A SCENICPLUS object with eRegulons AUC computed.
    auc_key: str, optional
            Key to extract AUC values from. Default: 'eRegulon_AUC'
    signature_keys: List, optional
            Keys to extract AUC values from. Default: ['Gene_based', 'Region_based']
    k: int, optional
            Number of neighbours in the k-neighbours graph. Default: 10
    res: float, optional
            Resolution parameter for the leiden algorithm step. Default: 0.6
    seed: int, optional
            Seed parameter for the leiden algorithm step. Default: 555
    scale: bool, optional
            Whether to scale the enrichment prior to the clustering. Default: False
    prefix: str, optional
            Prefix to add to the clustering name when adding it to the correspondent metadata attribute. Default: ''
    selected_regulons: list, optional
            A list with selected regulons to be used for clustering. Default: None (use all regulons)
    selected_cells: list, optional
            A list with selected features cells to cluster. Default: None (use all cells)
    """

    if scale:
        data_mat = pd.concat([pd.DataFrame(sklearn.preprocessing.StandardScaler().fit_transform(
            scplus_obj.uns[auc_key][x].T), index=scplus_obj.uns[auc_key][x].T.index.to_list(), columns=scplus_obj.uns[auc_key][x].T.columns) for x in signature_keys])
    else:
        data_mat = pd.concat([scplus_obj.uns[auc_key][x]
                             for x in signature_keys]).T
    data_names = data_mat.columns.tolist()

    if selected_regulons is not None:
        selected_regulons = [
            x for x in selected_regulons if x in data_mat.index]
        data_mat = data_mat.loc[selected_regulons]
    if selected_cells is not None:
        data_mat = data_mat[selected_cells]
        data_names = selected_cells

    data_mat = data_mat.T.fillna(0)

    A = kneighbors_graph(data_mat, k)
    sources, targets = A.nonzero()
    G = ig.Graph(directed=True)
    G.add_vertices(A.shape[0])
    edges = list(zip(sources, targets))
    G.add_edges(edges)
    for C in res:
        partition = la.find_partition(
            G,
            la.RBConfigurationVertexPartition,
            resolution_parameter=C,
            seed=seed)
        cluster = pd.DataFrame(
            partition.membership,
            index=data_names,
            columns=[
                prefix +
                'leiden_' +
                str(k) +
                '_' +
                str(C)]).astype(str)

        scplus_obj.add_cell_data(cluster)


def run_eRegulons_tsne(scplus_obj: SCENICPLUS,
                       scale: Optional[bool] = True,
                       auc_key: Optional[str] = 'eRegulon_AUC',
                       signature_keys: Optional[List[str]] = ['Gene_based', 'Region_based'],
                       reduction_name: Optional[str] = 'eRegulons_tSNE',
                       random_state: Optional[int] = 555,
                       selected_regulons: Optional[List[int]] = None,
                       selected_cells: Optional[List[str]] = None,
                       **kwargs):
    """
    Run TSNE and add it to the dimensionality reduction dictionary.

    Parameters
    ---------
    scplus_obj: `class::SCENICPLUS`
            A SCENICPLUS object with eRegulons AUC computed.
    scale: bool, optional
            Whether to scale the enrichments prior to the dimensionality reduction. Default: False
    auc_key: str, optional
            Key to extract AUC values from. Default: 'eRegulon_AUC'
    signature_keys: List, optional
            Keys to extract AUC values from. Default: ['Gene_based', 'Region_based']
    reduction_name: str, optional
            Key used to store dimensionality reduction in scplud_obj.dr_cell. Default: eRegulon_AUC.
    random_state: int, optional
            Seed parameter for running UMAP. Default: 555
    selected_regulons: list, optional
            A list with selected regulons to be used for clustering. Default: None (use all regulons)
    selected_cells: list, optional
            A list with selected features cells to cluster. Default: None (use all cells)
    **kwargs
            Parameters to pass to the tSNE functions.
    """

    if scale:
        data_mat = pd.concat([pd.DataFrame(sklearn.preprocessing.StandardScaler().fit_transform(
            scplus_obj.uns[auc_key][x].T), index=scplus_obj.uns[auc_key][x].T.index.to_list(), columns=scplus_obj.uns[auc_key][x].T.columns) for x in signature_keys])
    else:
        data_mat = pd.concat([scplus_obj.uns[auc_key][x]
                             for x in signature_keys]).T
    data_names = data_mat.columns.tolist()

    if selected_regulons is not None:
        selected_regulons = [
            x for x in selected_regulons if x in data_mat.index]
        data_mat = data_mat.loc[selected_regulons]
    if selected_cells is not None:
        data_mat = data_mat[selected_cells]
        data_names = selected_cells

    data_mat = data_mat.T.fillna(0)

    try:
        import fitsne
        embedding = fitsne.FItSNE(
            np.ascontiguousarray(
                data_mat.to_numpy()),
            rand_seed=random_state,
            perplexity=perplexity, **kwargs)
    except BaseException:
        embedding = sklearn.manifold.TSNE(
            n_components=2, random_state=random_state).fit_transform(
            data_mat.to_numpy(), **kwargs)
    dr = pd.DataFrame(
        embedding,
        index=data_names,
        columns=[
            'tSNE_1',
            'tSNE_2'])
    if not hasattr(scplus_obj, 'dr_cell'):
        scplus_obj.dr_cell = {}
    scplus_obj.dr_cell[reduction_name] = dr


def run_eRegulons_umap(scplus_obj: SCENICPLUS,
                       scale: Optional[bool] = True,
                       auc_key: Optional[str] = 'eRegulon_AUC',
                       signature_keys: Optional[List[str]] = ['Gene_based', 'Region_based'],
                       reduction_name: Optional[str] = 'eRegulons_UMAP',
                       random_state: Optional[int] = 555,
                       selected_regulons: Optional[List[int]] = None,
                       selected_cells: Optional[List[str]] = None,
                       **kwargs):
    """
    Run UMAP and add it to the dimensionality reduction dictionary.

    Parameters
    ---------
    scplus_obj: `class::SCENICPLUS`
            A SCENICPLUS object with eRegulons AUC computed.
    scale: bool, optional
            Whether to scale the cell-topic or topic-regions contributions prior to the dimensionality reduction. Default: False
    auc_key: str, optional
            Key to extract AUC values from. Default: 'eRegulon_AUC'
    signature_keys: List, optional
            Keys to extract AUC values from. Default: ['Gene_based', 'Region_based']
    reduction_name: str, optional
            Key used to store dimensionality reduction in scplud_obj.dr_cell. Default: eRegulon_AUC.
    random_state: int, optional
            Seed parameter for running UMAP. Default: 555
    selected_regulons: list, optional
            A list with selected regulons to be used for clustering. Default: None (use all regulons)
    selected_cells: list, optional
            A list with selected features cells to cluster. Default: None (use all cells)
    **kwargs
            Parameters to pass to umap.UMAP.
    """

    if scale:
        data_mat = pd.concat([pd.DataFrame(sklearn.preprocessing.StandardScaler().fit_transform(
            scplus_obj.uns[auc_key][x].T), index=scplus_obj.uns[auc_key][x].T.index.to_list(), columns=scplus_obj.uns[auc_key][x].T.columns) for x in signature_keys])
    else:
        data_mat = pd.concat([scplus_obj.uns[auc_key][x]
                             for x in signature_keys]).T
    data_names = data_mat.columns.tolist()

    if selected_regulons is not None:
        selected_regulons = [
            x for x in selected_regulons if x in data_mat.index]
        data_mat = data_mat.loc[selected_regulons]
    if selected_cells is not None:
        data_mat = data_mat[selected_cells]
        data_names = selected_cells

    if scale:
        data_mat = pd.DataFrame(sklearn.preprocessing.StandardScaler().fit_transform(
            data_mat), index=data_mat.index.to_list(), columns=data_mat.columns)

    data_mat = data_mat.T.fillna(0)

    reducer = umap.UMAP(random_state=random_state, **kwargs)
    embedding = reducer.fit_transform(data_mat)
    dr = pd.DataFrame(
        embedding,
        index=data_names,
        columns=[
            'UMAP_1',
            'UMAP_2'])
    if not hasattr(scplus_obj, 'dr_cell'):
        scplus_obj.dr_cell = {}
    scplus_obj.dr_cell[reduction_name] = dr


def run_eRegulons_pca(scplus_obj: SCENICPLUS,
                      scale: Optional[bool] = True,
                      auc_key: Optional[str] = 'eRegulon_AUC',
                      signature_keys: Optional[List[str]] = ['Gene_based', 'Region_based'],
                      reduction_name: Optional[str] = 'eRegulons_PCA',
                      random_state: Optional[int] = 555,
                      selected_regulons: Optional[List[int]] = None,
                      selected_cells: Optional[List[str]] = None,
                      n_pcs: Optional[int] = 50,
                      **kwargs):
    """
    Run UMAP and add it to the dimensionality reduction dictionary.

    Parameters
    ---------
    scplus_obj: `class::SCENICPLUS`
            A SCENICPLUS object with eRegulons AUC computed.
    scale: bool, optional
            Whether to scale the cell-topic or topic-regions contributions prior to the dimensionality reduction. Default: False
    auc_key: str, optional
            Key to extract AUC values from. Default: 'eRegulon_AUC'
    signature_keys: List, optional
            Keys to extract AUC values from. Default: ['Gene_based', 'Region_based']
    reduction_name: str, optional
            Key used to store dimensionality reduction in scplud_obj.dr_cell. Default: eRegulon_AUC.
    random_state: int, optional
            Seed parameter for running UMAP. Default: 555
    selected_regulons: list, optional
            A list with selected regulons to be used for clustering. Default: None (use all regulons)
    selected_cells: list, optional
            A list with selected features cells to cluster. Default: None (use all cells)
    n_pcs: int, optional
            Number of principle components to calculate. Default: 50
    **kwargs
            Parameters to pass to umap.UMAP.
    """

    if scale:
        data_mat = pd.concat([pd.DataFrame(sklearn.preprocessing.StandardScaler().fit_transform(
            scplus_obj.uns[auc_key][x].T), index=scplus_obj.uns[auc_key][x].T.index.to_list(), columns=scplus_obj.uns[auc_key][x].T.columns) for x in signature_keys])
    else:
        data_mat = pd.concat([scplus_obj.uns[auc_key][x]
                             for x in signature_keys]).T
    data_names = data_mat.columns.tolist()

    if selected_regulons is not None:
        selected_regulons = [
            x for x in selected_regulons if x in data_mat.index]
        data_mat = data_mat.loc[selected_regulons]
    if selected_cells is not None:
        data_mat = data_mat[selected_cells]
        data_names = selected_cells

    if scale:
        data_mat = pd.DataFrame(sklearn.preprocessing.StandardScaler().fit_transform(
            data_mat), index=data_mat.index.to_list(), columns=data_mat.columns)

    data_mat = data_mat.T.fillna(0)

    reducer = PCA(n_components=n_pcs, random_state=random_state)
    embedding = reducer.fit_transform(data_mat)

    dr = pd.DataFrame(
        embedding,
        index=data_names,
        columns=[f'PC_{i}' for i in range(n_pcs)])[['PC_0', 'PC_1']]
    if not hasattr(scplus_obj, 'dr_cell'):
        scplus_obj.dr_cell = {}
    scplus_obj.dr_cell[reduction_name] = dr

def plot_metadata_given_ax(scplus_obj,
                  ax: matplotlib.axes,
                  reduction_name: str,
                  variable: str,
                  remove_nan: Optional[bool] = True,
                  show_label: Optional[bool] = True,
                  show_legend: Optional[bool] = False,
                  cmap: Optional[Union[str, 'matplotlib.cm']] = cm.viridis,
                  dot_size: Optional[int] = 10,
                  text_size: Optional[int] = 10,
                  alpha: Optional[Union[float, int]] = 1,
                  seed: Optional[int] = 555,
                  color_dictionary: Optional[Dict[str, str]] = {},
                  selected_cells: Optional[List[str]] = None):
    """
    Plot categorical and continuous metadata into dimensionality reduction.

    Parameters
    ---------
    scplus_obj: `class::SCENICPLUS`
            A SCENICPLUS object with dimensionality reductions.
    ax: matplotlib.axes
            Axes to which to plot metadata.
    reduction_name: str
            Name of the dimensionality reduction to use
    variable: Str
            Variable to plot. It should be included in `class::SCENICPLUS.metadata_cell`.
    remove_nan: bool, optional
            Whether to remove data points for which the variable value is 'nan'. Default: True
    show_label: bool, optional
            For categorical variables, whether to show the label in the plot. Default: True
    show_legend: bool, optional
            For categorical variables, whether to show the legend next to the plot. Default: False
    cmap: str or 'matplotlib.cm', optional
            For continuous variables, color map to use for the legend color bar. Default: cm.viridis
    dot_size: int, optional
            Dot size in the plot. Default: 10
    text_size: int, optional
            For categorical variables and if show_label is True, size of the labels in the plot. Default: 10
    alpha: float, optional
            Transparency value for the dots in the plot. Default: 1
    seed: int, optional
            Random seed used to select random colors. Default: 555
    color_dictionary: dict, optional
            A dictionary containing an entry per variable, whose values are dictionaries with variable levels as keys and corresponding colors as values.
            Default: None
    figsize: tuple, optional
            Size of the figure. If num_columns is 1, this is the size for each figure; if num_columns is above 1, this is the overall size of the figure (if keeping
            default, it will be the size of each subplot in the figure). Default: (6.4, 4.8)
    num_columns: int, optional
            For multiplot figures, indicates the number of columns (the number of rows will be automatically determined based on the number of plots). Default: 1
    selected_cells: list, optional
            A list with selected cells to plot.
    save: str, optional
            Path to save plot. Default: None.
    """
    embedding = scplus_obj.dr_cell[reduction_name]
    data_mat = scplus_obj.metadata_cell.copy()

    if selected_cells is not None:
        data_mat = data_mat.loc[selected_cells]
        embedding = embedding.loc[selected_cells]

    data_mat = data_mat.loc[embedding.index.to_list()]

    var_data = data_mat.copy().loc[:, variable].dropna().to_list()
    if isinstance(var_data[0], str):
        if (remove_nan) & (data_mat[variable].isnull().sum() > 0):
            var_data = data_mat.copy().loc[:, variable].dropna().to_list()
            emb_nan = embedding.loc[data_mat.copy(
            ).loc[:, var].dropna().index.tolist()]
            label_pd = pd.concat(
                [emb_nan, data_mat.loc[:, [variable]].dropna()], axis=1, sort=False)
        else:
            var_data = data_mat.copy().astype(
                str).fillna('NA').loc[:, variable].to_list()
            label_pd = pd.concat([embedding, data_mat.astype(
                str).fillna('NA').loc[:, [variable]]], axis=1, sort=False)

        categories = set(var_data)
        try:
            color_dict = color_dictionary[variable]
        except BaseException:
            random.seed(seed)
            color = list(map(
                lambda i: "#" +
                "%06x" % random.randint(
                    0, 0xFFFFFF), range(len(categories))
            ))
            color_dict = dict(zip(categories, color))

        if (remove_nan) & (data_mat[variable].isnull().sum() > 0):
            ax.scatter(emb_nan.iloc[:, 0], emb_nan.iloc[:, 1], c=data_mat.loc[:, variable].dropna(
            ).apply(lambda x: color_dict[x]), s=dot_size, alpha=alpha)
            ax.set_xlabel(emb_nan.columns[0])
            ax.set_ylabel(emb_nan.columns[1])
        else:
            ax.scatter(embedding.iloc[:, 0], embedding.iloc[:, 1], c=data_mat.astype(str).fillna(
                'NA').loc[:, variable].apply(lambda x: color_dict[x]), s=dot_size, alpha=alpha)
            ax.set_xlabel(embedding.columns[0])
            ax.set_ylabel(embedding.columns[1])

        if show_label:
            label_pos = label_pd.groupby(variable).agg(
                {label_pd.columns[0]: np.mean, label_pd.columns[1]: np.mean})
            texts = []
            for label in label_pos.index.tolist():
                texts.append(
                    ax.text(
                        label_pos.loc[label][0],
                        label_pos.loc[label][1],
                        label,
                        horizontalalignment='center',
                        verticalalignment='center',
                        size=text_size,
                        weight='bold',
                        color=color_dict[label],
                        path_effects=[
                            PathEffects.withStroke(
                                linewidth=3,
                                foreground='w')]))
            adjust_text(texts)

        ax.set_title(variable)
        patchList = []
        for key in color_dict:
            data_key = mpatches.Patch(color=color_dict[key], label=key)
            patchList.append(data_key)
        if show_legend:
            ax.legend(
                handles=patchList, bbox_to_anchor=(
                    1.04, 1), loc="upper left")
        return ax
    else:
        var_data = data_mat.copy().loc[:, variable].to_list()
        o = np.argsort(var_data)
        ax.scatter(embedding.iloc[o, 0], embedding.iloc[o, 1], c=subset_list(
            var_data, o), cmap=cmap, s=dot_size, alpha=alpha)
        ax.set_xlabel(embedding.columns[0])
        ax.set_ylabel(embedding.columns[1])
        ax.set_title(variable)
        # setup the colorbar
        normalize = mcolors.Normalize(
            vmin=np.array(var_data).min(),
            vmax=np.array(var_data).max())
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
        scalarmappaple.set_array(var_data)
        plt.colorbar(scalarmappaple, ax = ax)
        return ax

def plot_metadata(scplus_obj: SCENICPLUS,
                  reduction_name: str,
                  variables: List[str],
                  remove_nan: Optional[bool] = True,
                  show_label: Optional[bool] = True,
                  show_legend: Optional[bool] = False,
                  cmap: Optional[Union[str, 'matplotlib.cm']] = cm.viridis,
                  dot_size: Optional[int] = 10,
                  text_size: Optional[int] = 10,
                  alpha: Optional[Union[float, int]] = 1,
                  seed: Optional[int] = 555,
                  color_dictionary: Optional[Dict[str, str]] = {},
                  figsize: Optional[Tuple[float, float]] = (6.4, 4.8),
                  num_columns: Optional[int] = 1,
                  selected_cells: Optional[List[str]] = None,
                  save: Optional[str] = None):
    """
    Plot categorical and continuous metadata into dimensionality reduction.

    Parameters
    ---------
    scplus_obj: `class::SCENICPLUS`
            A SCENICPLUS object with dimensionality reductions.
    reduction_name: str
            Name of the dimensionality reduction to use
    variables: list
            List of variables to plot. They should be included in `class::SCENICPLUS.metadata_cell`.
    remove_nan: bool, optional
            Whether to remove data points for which the variable value is 'nan'. Default: True
    show_label: bool, optional
            For categorical variables, whether to show the label in the plot. Default: True
    show_legend: bool, optional
            For categorical variables, whether to show the legend next to the plot. Default: False
    cmap: str or 'matplotlib.cm', optional
            For continuous variables, color map to use for the legend color bar. Default: cm.viridis
    dot_size: int, optional
            Dot size in the plot. Default: 10
    text_size: int, optional
            For categorical variables and if show_label is True, size of the labels in the plot. Default: 10
    alpha: float, optional
            Transparency value for the dots in the plot. Default: 1
    seed: int, optional
            Random seed used to select random colors. Default: 555
    color_dictionary: dict, optional
            A dictionary containing an entry per variable, whose values are dictionaries with variable levels as keys and corresponding colors as values.
            Default: None
    figsize: tuple, optional
            Size of the figure. If num_columns is 1, this is the size for each figure; if num_columns is above 1, this is the overall size of the figure (if keeping
            default, it will be the size of each subplot in the figure). Default: (6.4, 4.8)
    num_columns: int, optional
            For multiplot figures, indicates the number of columns (the number of rows will be automatically determined based on the number of plots). Default: 1
    selected_cells: list, optional
            A list with selected cells to plot.
    save: str, optional
            Path to save plot. Default: None.
    """

    embedding = scplus_obj.dr_cell[reduction_name]
    data_mat = scplus_obj.metadata_cell.copy()

    if selected_cells is not None:
        data_mat = data_mat.loc[selected_cells]
        embedding = embedding.loc[selected_cells]

    data_mat = data_mat.loc[embedding.index.to_list()]

    pdf = None
    if (save is not None) & (num_columns == 1):
        pdf = matplotlib.backends.backend_pdf.PdfPages(save)

    if num_columns > 1:
        num_rows = int(np.ceil(len(variables) / num_columns))
        if figsize == (6.4, 4.8):
            figsize = (6.4 * num_columns, 4.8 * num_rows)
        i = 1

    fig = plt.figure(figsize=figsize)

    for var in variables:
        var_data = data_mat.copy().loc[:, var].dropna().to_list()
        if isinstance(var_data[0], str):
            if (remove_nan) & (data_mat[var].isnull().sum() > 0):
                var_data = data_mat.copy().loc[:, var].dropna().to_list()
                emb_nan = embedding.loc[data_mat.copy(
                ).loc[:, var].dropna().index.tolist()]
                label_pd = pd.concat(
                    [emb_nan, data_mat.loc[:, [var]].dropna()], axis=1, sort=False)
            else:
                var_data = data_mat.copy().astype(
                    str).fillna('NA').loc[:, var].to_list()
                label_pd = pd.concat([embedding, data_mat.astype(
                    str).fillna('NA').loc[:, [var]]], axis=1, sort=False)

            categories = set(var_data)
            try:
                color_dict = color_dictionary[var]
            except BaseException:
                random.seed(seed)
                color = list(map(
                    lambda i: "#" +
                    "%06x" % random.randint(
                        0, 0xFFFFFF), range(len(categories))
                ))
                color_dict = dict(zip(categories, color))

            if num_columns > 1:
                plt.subplot(num_rows, num_columns, i)
                i = i + 1

            if (remove_nan) & (data_mat[var].isnull().sum() > 0):
                plt.scatter(emb_nan.iloc[:, 0], emb_nan.iloc[:, 1], c=data_mat.loc[:, var].dropna(
                ).apply(lambda x: color_dict[x]), s=dot_size, alpha=alpha)
                plt.xlabel(emb_nan.columns[0])
                plt.ylabel(emb_nan.columns[1])
            else:
                plt.scatter(embedding.iloc[:, 0], embedding.iloc[:, 1], c=data_mat.astype(str).fillna(
                    'NA').loc[:, var].apply(lambda x: color_dict[x]), s=dot_size, alpha=alpha)
                plt.xlabel(embedding.columns[0])
                plt.ylabel(embedding.columns[1])

            if show_label:
                label_pos = label_pd.groupby(var).agg(
                    {label_pd.columns[0]: np.mean, label_pd.columns[1]: np.mean})
                texts = []
                for label in label_pos.index.tolist():
                    texts.append(
                        plt.text(
                            label_pos.loc[label][0],
                            label_pos.loc[label][1],
                            label,
                            horizontalalignment='center',
                            verticalalignment='center',
                            size=text_size,
                            weight='bold',
                            color=color_dict[label],
                            path_effects=[
                                PathEffects.withStroke(
                                    linewidth=3,
                                    foreground='w')]))
                adjust_text(texts)

            plt.title(var)
            patchList = []
            for key in color_dict:
                data_key = mpatches.Patch(color=color_dict[key], label=key)
                patchList.append(data_key)
            if show_legend:
                plt.legend(
                    handles=patchList, bbox_to_anchor=(
                        1.04, 1), loc="upper left")

            if num_columns == 1:
                if save is not None:
                    pdf.savefig(fig, bbox_inches='tight')
                plt.show()
        else:
            var_data = data_mat.copy().loc[:, var].to_list()
            o = np.argsort(var_data)
            if num_columns > 1:
                plt.subplot(num_rows, num_columns, i)
                i = i + 1
            plt.scatter(embedding.iloc[o, 0], embedding.iloc[o, 1], c=subset_list(
                var_data, o), cmap=cmap, s=dot_size, alpha=alpha)
            plt.xlabel(embedding.columns[0])
            plt.ylabel(embedding.columns[1])
            plt.title(var)
            # setup the colorbar
            normalize = mcolors.Normalize(
                vmin=np.array(var_data).min(),
                vmax=np.array(var_data).max())
            scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
            scalarmappaple.set_array(var_data)
            plt.colorbar(scalarmappaple)
            if num_columns == 1:
                if save is not None:
                    pdf.savefig(fig, bbox_inches='tight')
                plt.show()

    if num_columns > 1:
        plt.tight_layout()
        if save is not None:
            fig.savefig(save, bbox_inches='tight')
        plt.show()
    if (save is not None) & (num_columns == 1):
        pdf = pdf.close()


def subset_list(target_list, index_list):
    X = list(map(target_list.__getitem__, index_list))
    return X

def plot_AUC_given_ax(
    scplus_obj: SCENICPLUS,
    reduction_name: str,
    feature: str,
    ax: matplotlib.axes,
    auc_key: str = 'eRegulon_AUC',
    signature_key: str = 'Gene_based',
    cmap: matplotlib.cm = cm.viridis,
    dot_size: int = 10,
    alpha: float = 1,
    scale: bool = False,
    selected_cells: List = None):
    """
    Plot eRegulon AUC values on dimmensionality reduction

    Parameters
    ----------
    scplus_obj: `class::SCENICPLUS`
        A SCENICPLUS object with dimensionality reductions.
    reduction_name: str
        Name of the dimensionality reduction to use.
    feature: str
        eRegulon to plot, should be included in scplus_obj.uns[auc_key][signature_key] matrix.
    ax: matplotlib.axes
        matplotlib axes to which to plot.
    auc_key: str, optional
        key in scplus_obj.uns under which the AUC values are stored
    signature_key: str, optional
        key in scplus_obj.uns[auc_key] to plot (usually Gene_based or Region_based)
    cmap: matplotlib.cm, optional
        color map to use for plotting.
    dot_size: int, optional
        Dot size in the plot. Default: 10
    alpha: float, optional
        Transparency value for the dots in the plot. Default: 1
    scale: bool, optional
        Wether to scale AUC values before plotting
    selected_cells: List, optional
        A list with selected cells to plot.
    """

    embedding = scplus_obj.dr_cell[reduction_name].copy()

    if scale:
        data_mat = pd.DataFrame(
                    sklearn.preprocessing.StandardScaler().fit_transform(
                        scplus_obj.uns[auc_key][signature_key].T), 
                        index=scplus_obj.uns[auc_key][signature_key].T.index.to_list(), 
                        columns=scplus_obj.uns[auc_key][signature_key].T.columns).T
    else:
        data_mat = scplus_obj.uns[auc_key][signature_key]

    if selected_cells is not None:
        data_mat = data_mat[selected_cells]

    data_mat = data_mat.fillna(0)
    feature_data = data_mat[feature].squeeze()
    feature_data = feature_data.sort_values()
    embedding_plot = embedding.loc[feature_data.index.tolist()]
    o = np.argsort(feature_data)
    if not scale:
        ax.scatter(
            embedding_plot.iloc[o, 0], 
            embedding_plot.iloc[o, 1], 
            c=subset_list(feature_data, o), 
            cmap=cmap, 
            s=dot_size, 
            alpha=alpha,
            vmin=0, 
            vmax=max(feature_data))
        normalize = mcolors.Normalize(
            vmin=0, vmax=np.array(feature_data).max())
    else:
        ax.scatter(
            embedding_plot.iloc[o, 0], 
            embedding_plot.iloc[o, 1], 
            c=subset_list(feature_data, o), cmap=cmap, s=dot_size, alpha=alpha)
        normalize = mcolors.Normalize(
            vmin=np.array(feature_data).min(),
            vmax=np.array(feature_data).max())
    ax.set_xlabel(embedding_plot.columns[0])
    ax.set_ylabel(embedding_plot.columns[1])
    ax.set_title(feature)
    # setup the colorbar
    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
    scalarmappaple.set_array(feature_data)
    #plt.colorbar(scalarmappaple)
    return ax


def plot_eRegulon(scplus_obj: SCENICPLUS,
                  reduction_name: str,
                  auc_key: Optional[str] = 'eRegulon_AUC',
                  signature_keys: Optional[List[str]] = ['Gene_based', 'Region_based'],
                  normalize_tf_expression: Optional[bool] = True,
                  cmap: Optional[Union[str, 'matplotlib.cm']] = cm.viridis,
                  dot_size: Optional[int] = 10,
                  alpha: Optional[Union[float, int]] = 1,
                  scale: Optional[bool] = False,
                  selected_regulons: Optional[List[int]] = None,
                  selected_cells: Optional[List[str]] = None,
                  figsize: Optional[Tuple[float, float]] = (6.4, 4.8),
                  num_columns: Optional[int] = 3,
                  save: Optional[str] = None):
    """
    Plot TF expression and eRegulon AUC (gene and region based) into dimensionality reduction.

    Parameters
    ---------
    scplus_obj: `class::SCENICPLUS`
            A cisTopic object with dimensionality reductions in `class::CistopicObject.projections`.
    reduction_name: str
            Name of the dimensionality reduction to use
    auc_key: str, optional
            Key to extract AUC values from. Default: 'eRegulon_AUC'
    signature_keys: List, optional
            Keys to extract AUC values from. Default: ['Gene_based', 'Region_based'] 
    normalize_tf_expression: bool, optional
            Whether logCPM normalize TF expression. Default: True    
    cmap: str or 'matplotlib.cm', optional
            For continuous variables, color map to use for the legend color bar. Default: cm.viridis
    dot_size: int, optional
            Dot size in the plot. Default: 10
    alpha: float, optional
            Transparency value for the dots in the plot. Default: 1
    scale: bool, optional
            Whether to scale the cell-topic or topic-regions contributions prior to plotting. Default: False
    selected_regulons: list, optional
            A list with selected regulons to be used for clustering. Default: None (use all regulons)
    selected_cells: list, optional
            A list with selected features cells to cluster. Default: None (use all cells)
    figsize: tuple, optional
            Size of the figure. If num_columns is 1, this is the size for each figure; if num_columns is above 1, this is the overall size of the figure (if keeping
            default, it will be the size of each subplot in the figure). Default: (6.4, 4.8)
    num_columns: int, optional
            For multiplot figures, indicates the number of columns (the number of rows will be automatically determined based on the number of plots). Default: 1
    save: str, optional
            Path to save plot. Default: None.
    """

    embedding = scplus_obj.dr_cell[reduction_name].copy()

    if scale:
        data_mat = pd.concat([pd.DataFrame(sklearn.preprocessing.StandardScaler().fit_transform(
            scplus_obj.uns[auc_key][x].T), index=scplus_obj.uns[auc_key][x].T.index.to_list(), columns=scplus_obj.uns[auc_key][x].T.columns) for x in signature_keys])
    else:
        data_mat = pd.concat([scplus_obj.uns[auc_key][x]
                             for x in signature_keys]).T
    data_names = data_mat.columns.tolist()

    if selected_cells is not None:
        data_mat = data_mat[selected_cells]
        data_names = selected_cells

    if selected_regulons is None:
        selected_regulons = [x.rsplit('_', 1)[0] for x in data_mat.index]

    if (save is not None) & (num_columns == 1):
        pdf = matplotlib.backends.backend_pdf.PdfPages(save)

    if num_columns > 1:
        num_rows = int(np.ceil(len(selected_regulons)))
        if figsize == (6.4, 4.8):
            figsize = (6.4 * num_columns, 4.8 * num_rows)
        i = 1

    fig = plt.figure(figsize=figsize)

    dgem = pd.DataFrame(scplus_obj.X_EXP, index=scplus_obj.cell_names,
                        columns=scplus_obj.gene_names).copy()

    if normalize_tf_expression:
        dgem = dgem.T / dgem.T.sum(0) * 10**6
        dgem = np.log1p(dgem).T
    dgem = dgem.T
    data_mat = data_mat.T.fillna(0)
    for regulon in selected_regulons:
        tf_name = regulon.split('_')[0]
        regulon_names = data_mat.columns[data_mat.columns.str.contains(
            regulon + '_(', regex=False)]
        gene_based_name = regulon_names[regulon_names.str.contains(
            'g)', regex=False)]
        region_based_name = regulon_names[regulon_names.str.contains(
            'r)', regex=False)]
        if tf_name in dgem.index:
            # TF expression
            tf_expr = dgem.loc[tf_name]
            tf_expr = tf_expr.sort_values()
            embedding_plot = embedding.loc[tf_expr.index.tolist()]
            o = np.argsort(tf_expr)
            if num_columns > 1:
                plt.subplot(num_rows, num_columns, i)
                i = i + 1
            if not scale:
                plt.scatter(embedding_plot.iloc[o, 0], embedding_plot.iloc[o, 1], c=subset_list(
                    tf_expr, o), cmap=cmap, s=dot_size, alpha=alpha, vmin=0, vmax=max(tf_expr))
                normalize = mcolors.Normalize(
                    vmin=0, vmax=np.array(tf_expr).max())
            else:
                plt.scatter(embedding_plot.iloc[o, 0], embedding_plot.iloc[o, 1], c=subset_list(
                    tf_expr, o), cmap=cmap, s=dot_size, alpha=alpha)
                normalize = mcolors.Normalize(
                    vmin=np.array(tf_expr).min(),
                    vmax=np.array(tf_expr).max())
            plt.xlabel(embedding_plot.columns[0])
            plt.ylabel(embedding_plot.columns[1])
            plt.title(tf_name+'_expression')
            # setup the colorbar
            scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
            scalarmappaple.set_array(tf_expr)
            plt.colorbar(scalarmappaple)
            if num_columns == 1:
                if save is not None:
                    pdf.savefig(fig, bbox_inches='tight')
                plt.show()
            # Gene data
            if num_columns > 1:
                plt.subplot(num_rows, num_columns, i)
                i = i + 1
            if 'Gene_based' in signature_keys:
                gene_data = data_mat[gene_based_name].squeeze()
                gene_data = gene_data.sort_values()
                embedding_plot = embedding.loc[gene_data.index.tolist()]
                o = np.argsort(gene_data)
                if not scale:
                    plt.scatter(embedding_plot.iloc[o, 0], embedding_plot.iloc[o, 1], c=subset_list(
                        gene_data, o), cmap=cmap, s=dot_size, alpha=alpha, vmin=0, vmax=max(gene_data))
                    normalize = mcolors.Normalize(
                        vmin=0, vmax=np.array(gene_data).max())
                else:
                    plt.scatter(embedding_plot.iloc[o, 0], embedding_plot.iloc[o, 1], c=subset_list(
                        gene_data, o), cmap=cmap, s=dot_size, alpha=alpha)
                    normalize = mcolors.Normalize(
                        vmin=np.array(gene_data).min(),
                        vmax=np.array(gene_data).max())
                plt.xlabel(embedding_plot.columns[0])
                plt.ylabel(embedding_plot.columns[1])
                plt.title(gene_based_name[0])
                # setup the colorbar
                scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
                scalarmappaple.set_array(gene_data)
                plt.colorbar(scalarmappaple)
                if num_columns == 1:
                    if save is not None:
                        pdf.savefig(fig, bbox_inches='tight')
                    plt.show()
            else:
                plt.figure()
            # Region data
            if num_columns > 1:
                plt.subplot(num_rows, num_columns, i)
                i = i + 1
            if 'Region_based' in signature_keys:
                region_data = data_mat[region_based_name].squeeze()
                region_data = region_data.sort_values()
                embedding_plot = embedding.loc[region_data.index.tolist()]
                o = np.argsort(region_data)
                if not scale:
                    plt.scatter(embedding_plot.iloc[o, 0], embedding_plot.iloc[o, 1], c=subset_list(
                        region_data, o), cmap=cmap, s=dot_size, alpha=alpha, vmin=0, vmax=max(region_data))
                    normalize = mcolors.Normalize(
                        vmin=0, vmax=np.array(region_data).max())
                else:
                    plt.scatter(embedding_plot.iloc[o, 0], embedding_plot.iloc[o, 1], c=subset_list(
                        region_data, o), cmap=cmap, s=dot_size, alpha=alpha)
                    normalize = mcolors.Normalize(
                        vmin=np.array(region_data).min(),
                        vmax=np.array(region_data).max())
                plt.xlabel(embedding_plot.columns[0])
                plt.ylabel(embedding_plot.columns[1])
                plt.title(region_based_name[0])
                # setup the colorbar
                scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
                scalarmappaple.set_array(region_data)
                plt.colorbar(scalarmappaple)
                if num_columns == 1:
                    if save is not None:
                        pdf.savefig(fig, bbox_inches='tight')
                    plt.show()
            else:
                plt.figure()

    if num_columns > 1:
        plt.tight_layout()
        if save is not None:
            fig.savefig(save, bbox_inches='tight')
        plt.show()

    if (save is not None) & (num_columns == 1):
        pdf.close()
