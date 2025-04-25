"""Plot correlation and overlap of eRegulons

"""

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
from typing import List, Tuple
from typing import Optional, Union
import sklearn
import matplotlib.pyplot as plt
from itertools import combinations

from ..utils import p_adjust_bh, flatten_list


def correlation_heatmap(scplus_obj,
                        auc_key: Optional[str] = 'eRegulon_AUC',
                        signature_keys: Optional[List[str]] = ['Gene_based', 'Region_based'],
                        scale: Optional[bool] = False,
                        linkage_method: Optional[str] = 'average',
                        fcluster_threshold: Optional[float] = 0.1,
                        selected_regulons: Optional[List[int]] = None,
                        cmap: Optional[str] = 'viridis',
                        plotly_height: Optional[int] = 1000,
                        fontsize: Optional[int] = 3,
                        save: Optional[str] = None,
                        use_plotly: Optional[int] = True,
                        figsize: Optional[Tuple[int, int]] = (20, 20)
                        ):
    """
    Plot correlation between eRegulons enrichment,

    Parameters
    ---------
    scplus_obj: `class::SCENICPLUS`
        A SCENICPLUS object with eRegulons AUC computed.
    auc_key: str, optional
        Key to extract AUC values from. Default: 'eRegulon_AUC'
    signature_keys: List, optional
        Keys to extract AUC values from. Default: ['Gene_based', 'Region_based']
    scale: bool, optional
        Whether to scale the enrichments prior to the dimensionality reduction. Default: False
    linkage_method: str, optional
        Linkage method to use for clustering. See `scipy.cluster.hierarchy.linkage`.
    fcluster_threshold: float, optional
        Threshold to use to divide hierarchical clustering into clusters. See `scipy.cluster.hierarchy.fcluster`.
    selected_regulons: list, optional
        A list with selected regulons to be used for clustering. Default: None (use all regulons)
    cmap: str or 'matplotlib.cm', optional
        For continuous variables, color map to use for the legend color bar. Default: cm.viridis
    plotly_height: int, optional
        Height of the plotly plot. Width will be adjusted accordingly
    fontsize: int, optional
        Labels fontsize
    save: str, optional
        Path to save heatmap as file
    use_plotly: bool, optional
        Use plotly or seaborn to generate the image
    figsize: tupe, optional
        Matplotlib figsize, used only if use_plotly == False
    """
    if scale:
        data_mat = pd.concat([pd.DataFrame(sklearn.preprocessing.StandardScaler().fit_transform(
            scplus_obj.uns[auc_key][x].T), index=scplus_obj.uns[auc_key][x].T.index.to_list(), columns=scplus_obj.uns[auc_key][x].T.columns) for x in signature_keys]).T
    else:
        data_mat = pd.concat([scplus_obj.uns[auc_key][x]
                             for x in signature_keys], axis=1)
    if selected_regulons is not None:
        subset = [x for x in selected_regulons if x in data_mat.columns]
        data_mat = data_mat[subset]
    # check if some eRegulon AUC are all 0 and remove them
    all_zero_eregs = data_mat.columns[np.where(data_mat.sum(0) == 0)]
    if len(all_zero_eregs) > 0:
        print(
            f"Following eregulons have an AUC value of all zero and will be removed: {', '.join(all_zero_eregs)}")
        data_mat.drop(all_zero_eregs, axis=1, inplace=True)
    correlations = data_mat.corr()
    similarity = 1 - correlations
    # use np.clip: due to floating point impercisions some very small values become negative, clip them to 0
    Z = linkage(np.clip(squareform(similarity), 0,
                similarity.to_numpy().max()), linkage_method)
    # Clusterize the data
    labels = fcluster(Z, fcluster_threshold)
    # Keep the indices to sort labels
    labels_order = np.argsort(labels)
    # Build a new dataframe with the sorted columns
    for idx, i in enumerate(data_mat.columns[labels_order]):
        if idx == 0:
            clustered = pd.DataFrame(data_mat[i])
        else:
            df_to_append = pd.DataFrame(data_mat[i])
            clustered = pd.concat([clustered, df_to_append], axis=1)
    correlations = clustered.corr()
    if use_plotly:
        fig = px.imshow(clustered.corr(), color_continuous_scale=cmap)
        fig.update_layout(
            height=plotly_height,  # Added parameter
            legend={'itemsizing': 'trace'},
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(tickfont=dict(size=fontsize)),
            xaxis=dict(tickfont=dict(size=fontsize)),
        )
        if save is not None:
            fig.write_image(save)
        fig.show()
    else:
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            data=correlations,
            cmap=cmap,
            square=True,
            ax=ax,
            robust=True,
            cbar_kws={"shrink": .50, 'label': 'Correlation'},
            xticklabels=False)
        if save is not None:
            fig.tight_layout()
            fig.savefig(save)
        plt.show()
        plt.close(fig)


def _jaccard(signature1, signature2):
    s_signature1 = set(signature1)
    s_signature2 = set(signature2)
    intersect = len(s_signature1 & s_signature2)
    union = len(s_signature1) + len(s_signature2) - intersect
    return intersect / union

def _intersect_norm_by_one(signature1, signature2):
    s_signature1 = set(signature1)
    s_signature2 = set(signature2)
    intersect = len(s_signature1 & s_signature2)
    return intersect / len(s_signature1)

from scipy.stats import fisher_exact
def _fisher_exact_sign(signature1, signature2, total):
    overlap = len(signature1 & signature2)
    contingency_table = np.array(
        [
            [total - len(signature1) - len(signature2), len(signature1) - overlap],
            [len(signature2) - overlap,                 overlap]
        ])
    return fisher_exact(contingency_table, alternative = 'greater')


def jaccard_heatmap(scplus_obj,
                    method: str = 'jaccard',
                    gene_or_region_based: str = 'Gene_based',
                    signature_key: Optional[str] = 'eRegulon_signatures',
                    selected_regulons: Optional[List[int]] = None,
                    linkage_method: Optional[str] = 'average',
                    fcluster_threshold: Optional[float] = 0.1,
                    cmap: Optional[str] = 'viridis',
                    plotly_height: Optional[int] = 1000,
                    fontsize: Optional[int] = 3,
                    save: Optional[str] = None,
                    use_plotly: Optional[int] = True,
                    figsize: Optional[Tuple[int, int]] = (20, 20),
                    vmin=None,
                    vmax=None,
                    return_data = False
                    ):
    """
    Plot jaccard index of regions/genes

    Parameters
    ---------
    scplus_obj: `class::SCENICPLUS`
        A SCENICPLUS object with eRegulon signatures.
    method: str
        Wether to use Jaccard (jaccard) or normalized intersection (intersect) as metric
    gene_or_region_based: str
        Gene_based or Region_based eRegulon signatures to use.
    signature_key: List, optional
        Key to extract eRegulon signatures from
    selected_regulons: list, optional
        A list with selected regulons to be used for clustering. Default: None (use all regulons)
    linkage_method: str, optional
        Linkage method to use for clustering. See `scipy.cluster.hierarchy.linkage`.
    fcluster_threshold: float, optional
        Threshold to use to divide hierarchical clustering into clusters. See `scipy.cluster.hierarchy.fcluster`.
    cmap: str or 'matplotlib.cm', optional
        For continuous variables, color map to use for the legend color bar. Default: cm.viridis
    plotly_height: int, optional
        Height of the plotly plot. Width will be adjusted accordingly
    fontsize: int, optional
        Labels fontsize
    save: str, optional
        Path to save heatmap as file
    use_plotly: bool, optional
        Use plotly or seaborn to generate the image
    figsize: tupe, optional
        Matplotlib figsize, used only if use_plotly == False
    return_data: boolean, optional
        Return data
    plot_dendrogram: boolean, optional
    """
    signatures = scplus_obj.uns[signature_key][gene_or_region_based]
    if selected_regulons is not None:
        signatures = {k: signatures[k]
                      for k in signatures.keys() if k in selected_regulons}
    signatures_names = list(signatures.keys())
    sign_combinations = list(combinations(signatures_names, 2))
    n_signatures = len(signatures_names)
    jaccards = np.zeros((n_signatures, n_signatures))

    for signature_1, signature_2 in sign_combinations:
        idx_1 = signatures_names.index(signature_1)
        idx_2 = signatures_names.index(signature_2)
        if method == 'jaccard':
            jaccards[idx_1, idx_2] = _jaccard(
                signatures[signature_1], signatures[signature_2])
            jaccards[idx_2, idx_1] = _jaccard(
                signatures[signature_2], signatures[signature_1])
        elif method == 'intersect':
            jaccards[idx_1, idx_2] = _intersect_norm_by_one(
                signatures[signature_1], signatures[signature_2])
            jaccards[idx_2, idx_1] = _intersect_norm_by_one(
                signatures[signature_2], signatures[signature_1])
    np.fill_diagonal(jaccards, 1)
    similarity = 1 - jaccards
    Z = linkage(similarity, linkage_method)
    # Clusterize the data
    labels = fcluster(Z, fcluster_threshold)
    # Keep the indices to sort labels
    labels_order = np.argsort(labels)
    clustered_jaccard_df = pd.DataFrame(
        jaccards, index=signatures_names, columns=signatures_names).iloc[labels_order, labels_order]
    if use_plotly:
        fig = px.imshow(clustered_jaccard_df, color_continuous_scale=cmap)
        fig.update_layout(
            height=plotly_height,  # Added parameter
            legend={'itemsizing': 'trace'},
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(tickfont=dict(size=fontsize)),
            xaxis=dict(tickfont=dict(size=fontsize)),
        )
        if save is not None:
            fig.write_image(save)
        fig.show()
    else:
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            data=clustered_jaccard_df,
            cmap=cmap,
            square=False,
            ax=ax,
            robust=True,
            cbar_kws={"shrink": .50, 'label': 'Jaccard'},
            xticklabels=False,
            vmin=vmin,
            vmax=vmax)
        if save is not None:
            fig.tight_layout()
            fig.savefig(save)
        plt.show()
        plt.close(fig)
    if return_data:
        return clustered_jaccard_df, Z


from scipy.stats import fisher_exact
def _fisher_exact_sign(signature1, signature2, total):
    overlap = len(signature1 & signature2)
    contingency_table = np.array(
        [
            [total - len(signature1) - len(signature2), len(signature1) - overlap],
            [len(signature2) - overlap,                 overlap]
        ])
    return fisher_exact(contingency_table, alternative = 'greater')


def fisher_exact_test_heatmap(
    scplus_obj,
    gene_or_region_based: str = 'Gene_based',
    signature_key: Optional[str] = 'eRegulon_signatures',
    selected_regulons: Optional[List[int]] = None,
    linkage_method: Optional[str] = 'average',
    fcluster_threshold: Optional[float] = 0.1,
    cmap: Optional[str] = 'viridis',
    plotly_height: Optional[int] = 1000,
    fontsize: Optional[int] = 3,
    save: Optional[str] = None,
    use_plotly: Optional[int] = True,
    figsize: Optional[Tuple[int, int]] = (20, 20),
    vmin=None,
    vmax=None,
    return_data = False):
    """
    Plot jaccard index of regions/genes

    Parameters
    ---------
    scplus_obj: `class::SCENICPLUS`
        A SCENICPLUS object with eRegulon signatures.
    method: str
        Wether to use Jaccard (jaccard) or normalized intersection (intersect) as metric
    gene_or_region_based: str
        Gene_based or Region_based eRegulon signatures to use.
    signature_key: List, optional
        Key to extract eRegulon signatures from
    selected_regulons: list, optional
        A list with selected regulons to be used for clustering. Default: None (use all regulons)
    linkage_method: str, optional
        Linkage method to use for clustering. See `scipy.cluster.hierarchy.linkage`.
    fcluster_threshold: float, optional
        Threshold to use to divide hierarchical clustering into clusters. See `scipy.cluster.hierarchy.fcluster`.
    cmap: str or 'matplotlib.cm', optional
        For continuous variables, color map to use for the legend color bar. Default: cm.viridis
    plotly_height: int, optional
        Height of the plotly plot. Width will be adjusted accordingly
    fontsize: int, optional
        Labels fontsize
    save: str, optional
        Path to save heatmap as file
    use_plotly: bool, optional
        Use plotly or seaborn to generate the image
    figsize: tupe, optional
        Matplotlib figsize, used only if use_plotly == False
    return_data: boolean, optional
        Return data
    plot_dendrogram: boolean, optional
    """
    signatures = scplus_obj.uns[signature_key][gene_or_region_based]
    if selected_regulons is not None:
        signatures = {k: signatures[k]
                      for k in signatures.keys() if k in selected_regulons}
    signatures_names = list(signatures.keys())
    sign_combinations = list(combinations(signatures_names, 2))
    n_signatures = len(signatures_names)
    fisher_exact_values = np.zeros((n_signatures, n_signatures))
    fisher_exact_pvalues =  np.ones((n_signatures, n_signatures))


    total_elements = len(set(flatten_list(scplus_obj.uns[signature_key][gene_or_region_based].values())))
    
    for signature_1, signature_2 in sign_combinations:
        idx_1 = signatures_names.index(signature_1)
        idx_2 = signatures_names.index(signature_2)
        test_result = _fisher_exact_sign(set(signatures[signature_1]), set(signatures[signature_2]), total_elements)
        fisher_exact_values[idx_1, idx_2] = test_result[0]
        fisher_exact_values[idx_2, idx_1] = test_result[0]
        fisher_exact_pvalues[idx_1, idx_2] = test_result[1]
        fisher_exact_pvalues[idx_2, idx_1] = test_result[1]
    
    pvals = fisher_exact_pvalues[np.triu_indices(fisher_exact_pvalues.shape[0], k = 1)]
    p_adj = p_adjust_bh(pvals)
    p_adj_mat = np.zeros((fisher_exact_pvalues.shape[0],fisher_exact_pvalues.shape[0]))
    p_adj_mat[np.triu_indices(p_adj_mat.shape[0], k = 1)] = p_adj
    p_adj_mat = p_adj_mat + p_adj_mat.T
    
    similarity = 1 - (fisher_exact_values - fisher_exact_values.min()) / (fisher_exact_values.max() - fisher_exact_values.min())
    Z = linkage(similarity, linkage_method)
    # Clusterize the data
    labels = fcluster(Z, fcluster_threshold)
    # Keep the indices to sort labels
    labels_order = np.argsort(labels)
    clustered_jaccard_df = pd.DataFrame(
        fisher_exact_values, index=signatures_names, columns=signatures_names).iloc[labels_order, labels_order]
    p_adj_df = pd.DataFrame(
        p_adj_mat, index = signatures_names, columns=signatures_names).iloc[labels_order, labels_order]
    if use_plotly:
        fig = px.imshow(clustered_jaccard_df, color_continuous_scale=cmap)
        fig.update_layout(
            height=plotly_height,  # Added parameter
            legend={'itemsizing': 'trace'},
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(tickfont=dict(size=fontsize)),
            xaxis=dict(tickfont=dict(size=fontsize)),
        )
        if save is not None:
            fig.write_image(save)
        fig.show()
    else:
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            data=clustered_jaccard_df,
            cmap=cmap,
            square=False,
            ax=ax,
            robust=True,
            cbar_kws={"shrink": .50, 'label': 'Jaccard'},
            xticklabels=False,
            vmin=vmin,
            vmax=vmax)
        if save is not None:
            fig.tight_layout()
            fig.savefig(save)
        plt.show()
        plt.close(fig)
    if return_data:
        return clustered_jaccard_df,p_adj_df, Z


    


