"""Calculate the specificty of eRegulons in clusters of cells.

Calculates the distance between the real distribution of eRegulon AUC values and
a fictional distribution where the eRegulon is only expressed/accessible in cells
of a certain cluster.

"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from math import ceil, floor
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Tuple, Union
from adjustText import adjust_text
from mudata import MuData
from scenicplus.scenicplus_mudata import ScenicPlusMuData

def rss(aucs, labels):
        # jensenshannon function provides distance 
        # which is the sqrt of the JS divergence.
        return 1.0 - jensenshannon(aucs / aucs.sum(), labels / labels.sum())

def regulon_specificity_scores_df(
                               data_matrix: pd.DataFrame,
                               variable_matrix: pd.Series):

    """
    Calculate the Regulon Specificty Scores (RSS). [doi: 10.1016/j.celrep.2018.10.045]

    Parameters
    ---------
    data_matrix: 'class::pd.DataFrame`
        A pandas dataframe containing regulon scores per cell.
    variable_matrix: 'class::pd.Series'
        A pandas series with an annotation per cell.
    """

    cell_types = list(variable_matrix.unique())
    n_types = len(cell_types)
    regulons = list(data_matrix.columns)
    n_regulons = len(regulons)
    rss_values = np.empty(shape=(n_types, n_regulons), dtype=float)

    for cidx, regulon_name in enumerate(regulons):
        for ridx, type in enumerate(cell_types):
            rss_values[ridx, cidx] = rss(
                data_matrix[regulon_name], (variable_matrix == type).astype(int))

    rss_values = pd.DataFrame(
        data=rss_values, index=cell_types, columns=regulons)

    return rss_values

def regulon_specificity_scores(scplus_mudata: Union[MuData, ScenicPlusMuData],
                               variable: str,
                               modalities: list,
                               selected_regulons: List[int] = None):
    """
    Calculate the Regulon Specificty Scores (RSS). [doi: 10.1016/j.celrep.2018.10.045]

    Parameters
    ---------
    scplus_mudata: `class::MuData` or 'class::ScenicPlusMuData'
        A MuData object with eRegulons AUC computed.
    variable: str
        Variable to calculate the RSS values for.
    modalities: List,
        A list of modalities to calculate RSS values for.
    selected_regulons: List, optional
        Regulons to calculate RSS values for.
    """

    #TODO: add checks

    rss_values_per_modality = []

    for modality in modalities:
        if selected_regulons is not None:
            modality_regulons = [regulon for regulon in selected_regulons 
            if regulon in scplus_mudata.mod[modality].var_names]
            
        else:
            modality_regulons = list(scplus_mudata.mod[modality].var_names)
        data_matrix = scplus_mudata.mod[modality][:, modality_regulons].to_df()

        variable_matrix = scplus_mudata.obs.loc[data_matrix.index, variable]

        rss_values_per_modality.append(regulon_specificity_scores_df(data_matrix=data_matrix,
                                         variable_matrix=variable_matrix))
    


    return pd.concat(rss_values_per_modality, axis=1)


def plot_rss(data_matrix: pd.DataFrame,
             top_n: int = 5,
             selected_groups: List[str] = None,
             num_columns: int = 1,
             figsize: Tuple[float, float] = (6.4, 4.8),
             fontsize: int = 12,
             save: str = None):
    """
    Plot RSS values per group

    Parameters
    ---------
    data_matrix: `class::pd.DataFrame`
        A pandas dataframe with RSS scores per variable.
    top_n: int, optional
        Number of top eRegulons to highlight.
    selected_groups: List, optional
        Groups to plot. Default: None (all)
    num_columns: int, optional
        Number of columns for multiplotting
    figsize: tuple, optional
        Size of the figure. If num_columns is 1, this is the size for each figure;
        if num_columns is above 1, this is the overall size of the figure (if keeping
        default, it will be the size of each subplot in the figure). Default: (6.4, 4.8)
    fontsize: int, optional
        Size of the eRegulons names in plot.
    save: str, optional
        Path to save plot. Default: None.
    """

    if selected_groups is None:
        cats = sorted(data_matrix.index.tolist())
    else:
        cats = selected_groups

    if num_columns > 1:
        num_rows = int(np.ceil(len(cats) / num_columns))
        figsize = (figsize[0] * num_columns, figsize[1] * num_rows)
        i = 1
        fig = plt.figure(figsize=figsize)

    pdf = None
    if (save is not None) & (num_columns == 1):
        pdf = matplotlib.backends.backend_pdf.PdfPages(save)

    for c in cats:
        x = data_matrix.T[c]
        if num_columns > 1:
            ax = fig.add_subplot(num_rows, num_columns, i)
            i = i + 1
        else:
            fig = plt.figure(figsize=figsize)
            ax = plt.axes()
        _plot_rss_internal(data_matrix, c, top_n=top_n, max_n=None, ax=ax)
        ax.set_ylim(x.min()-(x.max()-x.min())*0.05,
                    x.max()+(x.max()-x.min())*0.05)
        for t in ax.texts:
            t.set_fontsize(fontsize)
        ax.set_ylabel('')
        ax.set_xlabel('')
        adjust_text(ax.texts)
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

    for idx, (regulon_name, rss_val) in enumerate(
        zip(data[0:top_n].index, data[0:top_n].values)):

        ax.plot([idx, idx], [rss_val, rss_val], 'r.')
        ax.text(
            idx + (max_n / 25),
            rss_val,
            regulon_name,
            fontdict=font,
            horizontalalignment='left',
            verticalalignment='center',
        )
