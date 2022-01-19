"""Calculate the specificty of eRegulons in clusters of cells.

Calculates the distance between the real distribution of eRegulon AUC values and a fictional distribution where the eRegulon is only expressed/accessible in cells of a certain cluster.

"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from math import ceil, floor
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Tuple
from typing import Optional
from adjustText import adjust_text
import sklearn

from .scenicplus_class import SCENICPLUS


def regulon_specificity_scores(scplus_obj: SCENICPLUS,
                               variable: str,
                               auc_key: Optional[str] = 'eRegulon_AUC',
                               signature_keys: Optional[List[str]] = ['Gene_based', 'Region_based'],
                               selected_regulons: Optional[List[int]] = None,
                               scale: Optional[bool] = False,
                               out_key_suffix: Optional[str] = ''):
    """
    Calculate the Regulon Specificty Scores (RSS). [doi: 10.1016/j.celrep.2018.10.045]

    Parameters
    ---------
    scplus_obj: `class::SCENICPLUS`
        A SCENICPLUS object with eRegulons AUC computed.
    variable: str
        Variable to calculate the RSS values for.
    auc_key: str, optional
        Key to extract AUC values from. Default: 'eRegulon_AUC'
    signature_keys: List, optional
        Keys to extract AUC values from. Default: ['Gene_based', 'Region_based']
    scale: bool, optional
        Whether to scale the enrichment prior to the clustering. Default: False
    out_key_suffix: str, optional
        Suffix to add to the variable name to store the values (at scplus_obj.uns['RSS'])
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

    cell_data_series = scplus_obj.metadata_cell.loc[data_mat.index, variable]
    cell_data = list(cell_data_series.unique())
    n_types = len(cell_data)
    regulons = list(data_mat.columns)
    n_regulons = len(regulons)
    rss_values = np.empty(shape=(n_types, n_regulons), dtype=np.float)

    def rss(aucs, labels):
        # jensenshannon function provides distance which is the sqrt of the JS divergence.
        return 1.0 - jensenshannon(aucs / aucs.sum(), labels / labels.sum())

    for cidx, regulon_name in enumerate(regulons):
        for ridx, type in enumerate(cell_data):
            rss_values[ridx, cidx] = rss(
                data_mat[regulon_name], (cell_data_series == type).astype(int))

    rss_values = pd.DataFrame(
        data=rss_values, index=cell_data, columns=regulons)

    if not 'RSS' in scplus_obj.uns.keys():
        scplus_obj.uns['RSS'] = {}
    out_key = variable + out_key_suffix
    if not out_key in scplus_obj.uns['RSS'].keys():
        scplus_obj.uns['RSS'][out_key] = {}
    scplus_obj.uns['RSS'][out_key] = rss_values


def plot_rss(scplus_obj: SCENICPLUS,
             rss_key: str,
             top_n: Optional[int] = 5,
             selected_groups: Optional[List[str]] = None,
             num_columns: Optional[int] = 1,
             figsize: Optional[Tuple[float, float]] = (6.4, 4.8),
             fontsize: Optional[int] = 12,
             save: str = None):
    """
    Plot RSS values per group

    Parameters
    ---------
    scplus_obj: `class::SCENICPLUS`
        A SCENICPLUS object with eRegulons AUC computed.
    rss_key: str, optional
        Key to extract RSS values from.
    top_n: int, optional
        Number of top eRegulons to highlight.
    selected_groups: List, optional
        Groups to plot. Default: None (all)
    num_columns: int, optional
        Number of columns for multiplotting
    figsize: tuple, optional
        Size of the figure. If num_columns is 1, this is the size for each figure; if num_columns is above 1, this is the overall size of the figure (if keeping
        default, it will be the size of each subplot in the figure). Default: (6.4, 4.8)
    fontsize: int, optional
        Size of the eRegulons names in plot.
    save: str, optional
        Path to save plot. Default: None.
    """
    data_mat = scplus_obj.uns['RSS'][rss_key]
    if selected_groups is None:
        cats = sorted(data_mat.index.tolist())
    else:
        cats = selected_groups

    if num_columns > 1:
        num_rows = int(np.ceil(len(cats) / num_columns))
        if figsize == (6.4, 4.8):
            figsize = (6.4 * num_columns, 4.8 * num_rows)
        i = 1
        fig = plt.figure(figsize=figsize)

    pdf = None
    if (save is not None) & (num_columns == 1):
        pdf = matplotlib.backends.backend_pdf.PdfPages(save)

    for c in cats:
        x = data_mat.T[c]
        if num_columns > 1:
            ax = fig.add_subplot(num_rows, num_columns, i)
            i = i + 1
        else:
            fig = plt.figure(figsize=figsize)
            ax = plt.axes()
        _plot_rss_internal(data_mat, c, top_n=top_n, max_n=None, ax=ax)
        ax.set_ylim(x.min()-(x.max()-x.min())*0.05,
                    x.max()+(x.max()-x.min())*0.05)
        for t in ax.texts:
            t.set_fontsize(fontsize)
        ax.set_ylabel('')
        ax.set_xlabel('')
        adjust_text(ax.texts, autoalign='xy', ha='right', va='bottom', arrowprops=dict(
            arrowstyle='-', color='lightgrey'), precision=0.001)
        if num_columns == 1:
            fig.text(0.5, 0.0, 'eRegulon rank', ha='center',
                     va='center', size='x-large')
            fig.text(0.00, 0.5, 'eRegulon specificity score (eRSS)',
                     ha='center', va='center', rotation='vertical', size='x-large')
            plt.tight_layout()
            plt.rcParams.update({
                'figure.autolayout': True,
                'figure.titlesize': 'large',
                'axes.labelsize': 'medium',
                'axes.titlesize': 'large',
                'xtick.labelsize': 'medium',
                'ytick.labelsize': 'medium'
            })
            if save is not None:
                pdf.savefig(fig, bbox_inches='tight')
            plt.show()

    if num_columns > 1:
        fig.text(0.5, 0.0, 'eRegulon rank', ha='center',
                 va='center', size='x-large')
        fig.text(0.00, 0.5, 'eRegulon specificity score (eRSS)',
                 ha='center', va='center', rotation='vertical', size='x-large')
        plt.tight_layout()
        plt.rcParams.update({
            'figure.autolayout': True,
            'figure.titlesize': 'large',
            'axes.labelsize': 'medium',
            'axes.titlesize': 'large',
            'xtick.labelsize': 'medium',
            'ytick.labelsize': 'medium'
        })
        if save is not None:
            fig.savefig(save, bbox_inches='tight')
        plt.show()
    if (save is not None) & (num_columns == 1):
        pdf = pdf.close()


def _plot_rss_internal(rss, cell_type, top_n=5, max_n=None, ax=None):
    """
    Helper function to plot RSS
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(4, 4))
    if max_n is None:
        max_n = rss.shape[1]
    data = rss.T[cell_type].sort_values(ascending=False)[0:max_n]
    ax.plot(np.arange(len(data)), data, '.')
    ax.set_ylim([floor(data.min() * 100.0) / 100.0,
                ceil(data.max() * 100.0) / 100.0])
    ax.set_title(cell_type)
    ax.set_xticklabels([])

    font = {
        'color': 'red',
        'weight': 'normal'
    }

    for idx, (regulon_name, rss_val) in enumerate(zip(data[0:top_n].index, data[0:top_n].values)):
        ax.plot([idx, idx], [rss_val, rss_val], 'r.')
        ax.text(
            idx + (max_n / 25),
            rss_val,
            regulon_name,
            fontdict=font,
            horizontalalignment='left',
            verticalalignment='center',
        )
