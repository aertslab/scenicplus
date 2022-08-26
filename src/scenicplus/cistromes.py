# -*- coding: utf-8 -*-
"""Merging, scoring and assessing TF correlations of cistromes

"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pycistarget
from pycistarget.motif_enrichment_dem import *
from pycisTopic.diff_features import *
from pycisTopic.signature_enrichment import *
import pyranges as pr
from random import sample
import seaborn as sns
from scipy.stats import pearsonr
from typing import List
from .utils import region_names_to_coordinates, target_to_overlapping_query, p_adjust_bh, Groupby, flatten_list

from .scenicplus_class import SCENICPLUS

def _signatures_to_iter(menr):
    for x in menr.keys():
        if isinstance(menr[x], pycistarget.motif_enrichment_dem.DEM):
            for y in menr[x].cistromes['Region_set'].keys():
                for z in menr[x].cistromes['Region_set'][y]:
                    yield x, y, z, menr[x].cistromes['Region_set'][y][z]
        elif isinstance(menr[x], dict):
            for y in menr[x].keys():
                if not isinstance(menr[x][y], pycistarget.motif_enrichment_cistarget.cisTarget):
                    raise ValueError(f'Only motif enrichment results from pycistarget or DEM are allowed, not {type(menr[x][y])}')
                for z in menr[x][y].cistromes['Region_set']:
                    yield x, y, z, menr[x][y].cistromes['Region_set'][z]
        else:
            raise ValueError(f'Only motif enrichment results from pycistarget or DEM are allowed, not {type(menr[x])}')

def _get_signatures_as_dict(i):
    return {z+'__'+x+'__'+y: regions for x, y, z, regions in i}

def _merge_dict_of_signatures(d, suffix = ''):
    arr_keys_signatures = np.array(list(d.keys()))
    grouper = Groupby([x.split('_')[0] for x in arr_keys_signatures])
    merged_signatures = {}
    for TF, idx in zip(grouper.keys, grouper.indices):
        merged_signatures[TF + suffix] = pr.PyRanges(
            region_names_to_coordinates(set(flatten_list([d[x] for x in arr_keys_signatures[idx]]))))
    return merged_signatures

def _overlap_if_necessary(d, test_regions, regions_to_overlap):
    d_overlap = {}
    for k in d.keys():
        s_query_regions = set(coord_to_region_names(d[k]))
        #if the signature regions are already in the scplus_obj coordinate system, do nothing, otherwise overlap
        if len(s_query_regions & test_regions) != len(s_query_regions):
            signature_regions = target_to_overlapping_query(regions_to_overlap, d[k])
        else:
            signature_regions = d[k]
        if len(signature_regions) != 0:
            d_overlap[k] = signature_regions
    return d_overlap


def merge_cistromes(scplus_obj: SCENICPLUS,
                    cistromes_key: str = 'Unfiltered',
                    subset: pr.PyRanges = None):
    """Generate cistromes from motif enrichment tables

    Parameters
    ---------
    scplus_obj: :class:`SCENICPLUS`
        A :class:`SCENICPLUS` object with motif enrichment results from pycistarget (`scplus_obj.menr`).
        Several analyses can be included in the slot (topics/DARs/other; and different methods [Homer/DEM/cistarget]).
    cistromes_key: str, optional
        Key to store cistromes. Cistromes will stored at `scplus_obj.uns['Cistromes'][siganture_key]`
    subset: list
        A PyRanges containing a set of regions that regions in cistromes must overlap. This is useful when
        aiming for cell type specific cistromes for example (e.g. providing the cell type's MACS peaks)
    """
    menr = scplus_obj.menr
    # Get signatures from Homer/Cistarget outputs
    signatures = _get_signatures_as_dict(_signatures_to_iter(menr))

    #split direct and indirect signatures
    signatures_direct = {x: signatures[x] for x in signatures.keys() if not 'extended' in x}
    signatures_extend = {x: signatures[x] for x in signatures.keys() if     'extended' in x}

    if len(signatures_direct.keys()) == 0 and len(signatures_extend.keys()) == 0:
        raise ValueError("No cistromes found! Make sure that the motif enrichment results look good!")
    
    #merge regions by TF name
    if len(signatures_direct.keys()) > 0:
        merged_signatures_direct = _merge_dict_of_signatures(signatures_direct, suffix = '')
    if len(signatures_extend.keys()) > 0:
        merged_signatures_extend = _merge_dict_of_signatures(signatures_extend, suffix = '_extended')
    
    #overlap regions with scplus_regions
    regions = set(scplus_obj.region_names)
    pr_regions = pr.PyRanges(region_names_to_coordinates(regions))
    if subset is not None:
        #make sure subset is in scplus_regions coordinate system
        regions_to_overlap = target_to_overlapping_query(pr_regions, subset)
    else:
        regions_to_overlap = pr_regions
    
    if len(signatures_direct.keys()) > 0:
        merged_signatures_direct = _overlap_if_necessary(merged_signatures_direct, regions, regions_to_overlap)
    if len(signatures_extend.keys()) > 0:
        merged_signatures_extend = _overlap_if_necessary(merged_signatures_extend, regions, regions_to_overlap)
    
    # Sort alphabetically
    if len(signatures_direct.keys()) > 0:
        merged_signatures_direct = dict(
            sorted(merged_signatures_direct.items(), key=lambda x: x[0].lower()))
    if len(signatures_extend.keys()) > 0:
        merged_signatures_extend = dict(
            sorted(merged_signatures_extend.items(), key=lambda x: x[0].lower()))
    # Combine
    if len(signatures_direct.keys()) > 0 and len(signatures_extend.keys()) > 0:
        merged_signatures = {**merged_signatures_direct,
                             **merged_signatures_extend}
    elif len(signatures_direct.keys()) > 0 and len(signatures_extend.keys()) == 0:
        merged_signatures = merged_signatures_direct
    elif len(signatures_extend.keys()) > 0 and len(signatures_direct.keys()) == 0:
        merged_signatures = merged_signatures_extend
    # Add number of regions
    merged_signatures = {
        x + '_(' + str(len(merged_signatures[x])) + 'r)': merged_signatures[x] for x in merged_signatures.keys()}
    # Store in object
    if not 'Cistromes' in scplus_obj.uns.keys():
        scplus_obj.uns['Cistromes'] = {}
    scplus_obj.uns['Cistromes'][cistromes_key] = merged_signatures

# Score cistromes in cells


def score_cistromes(scplus_obj: SCENICPLUS,
                    ranking: CistopicImputedFeatures,
                    cistromes_key: str = 'Unfiltered',
                    enrichment_type: str = 'region',
                    auc_threshold: float = 0.05,
                    normalize: bool = False,
                    n_cpu: int = 1):
    """
    Get enrichment of a region signature in cells  using AUCell (Van de Sande et al., 2020)

    Parameters
    ---------
    scplus_obj: :class:`SCENICPLUS`
        A :class:`SCENICPLUS` object with motif enrichment results from pycistarget (`scplus_obj.menr`).
    rankings: CistopicImputedFeatures
        A CistopicImputedFeatures object with ranking values
    cistromes_key: str, optional
        Key to store where cistromes are stored. Cistromes will retrieved from `scplus_obj.uns['Cistromes'][siganture_key]`
        and AUC values will be stored in `scplus_obj.uns['Cistromes_AUC'][cistromes_key]`.
    enrichment_type: str
        Whether features are genes or regions
    auc_threshold: float
        The fraction of the ranked genome to take into account for the calculation of the Area Under the recovery Curve. Default: 0.05
    normalize: bool
        Normalize the AUC values to a maximum of 1.0 per regulon. Default: False
    num_workers: int
        The number of cores to use. Default: 1

    References
    ---------
    Van de Sande, B., Flerin, C., Davie, K., De Waegeneer, M., Hulselmans, G., Aibar, S., ... & Aerts, S. (2020). A scalable SCENIC workflow for single-cell gene 
    regulatory network analysis. Nature Protocols, 15(7), 2247-2276.
    """
    if not 'Cistromes_AUC' in scplus_obj.uns.keys():
        scplus_obj.uns['Cistromes_AUC'] = {}
    scplus_obj.uns['Cistromes_AUC'][cistromes_key] = signature_enrichment(ranking,
                                                                          scplus_obj.uns['Cistromes'][cistromes_key],
                                                                          enrichment_type,
                                                                          auc_threshold,
                                                                          normalize,
                                                                          n_cpu)
# Create pseudobulks


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
            sample_cells = sample(cells, nr_cells)
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


def prune_plot(scplus_obj: SCENICPLUS,
               name: str,
               pseudobulk_variable: str = None,
               auc_key: str = 'Cistromes_AUC',
               signature_key: str = 'Unfiltered',
               use_pseudobulk: bool = True,
               show_dot_plot: bool = True,
               show_line_plot: bool = False,
               color_dict=None,
               subset=None,
               seed=555,
               ax: plt.axes = None,
               **kwargs):
    """
    Plot cistrome accessibility versus TF expression

    Parameters
    ---------
    scplus_obj: :class:`SCENICPLUS`
        A :class:`SCENICPLUS` object.
    cistrome_name: str, optional
        Cistrome to plot. The TF name (or followed by extended) is enough to quert.
    pseudobulk_variable: str, optional
        Key to retrieve the pseudobulk matrices.  Cistrome accessibility will be retrieved from
        `scplus_obj.uns['Pseudobulk'][pseudobulk_variable]['Cistromes_AUC'][cistromes_key]` and 
        gene expression from `scplus_obj.uns['Pseudobulk'][pseudobulk_variable]['Expression']`.
    cistromes_key: str, optional
        Key to retrieve the pseudobulk matrices.  Cistrome accessibility will be retrieved from
        `scplus_obj.uns['Pseudobulk'][pseudobulk_variable]['Cistromes_AUC'][cistromes_key]` and 
        gene expression from `scplus_obj.uns['Pseudobulk'][pseudobulk_variable]['Expression']`.
    use_pseudobulk: bool, optional
        Whether to use pseudobulk matrix or actual values.
    show_dot_plot: bool, optional
        Whether to show dots in plot
    show_line_plot: bool, optional
        Whether to show line fitting to plot
    color_dict: Dict, optional
        Color dictionary to specify colors
    subset: List, optional
        Subset of pseudobulks/cells to use
    seed: int
        Seed to ensure that pseudobulk are reproducible.
    ax: plt.axes, optional
        matplotlib axes to plot to.
    **kwargs:
        Parameters for seaborn plotting.

    """
    if use_pseudobulk:
        dgem = scplus_obj.uns['Pseudobulk'][pseudobulk_variable]['Expression'].copy(
        )
        cistromes_auc = scplus_obj.uns['Pseudobulk'][pseudobulk_variable][auc_key][signature_key].copy(
        )
        cell_data = pd.DataFrame([x.rsplit('_', 1)[
                                 0] for x in cistromes_auc.columns], index=cistromes_auc.columns).iloc[:, 0]
    else:
        dgem = pd.DataFrame(scplus_obj.X_EXP, index=scplus_obj.cell_names,
                            columns=scplus_obj.gene_names).copy().T
        cistromes_auc = scplus_obj.uns[auc_key][signature_key].copy().T
        cell_data = scplus_obj.metadata_cell.loc[cistromes_auc.columns,
                                                 pseudobulk_variable]
    if subset is None:
        tf_expr = dgem.loc[name.split('_')[0], :]
        tf_acc = cistromes_auc.index[cistromes_auc.index.str.contains(
            name + '_(', regex=False)][0]
        cistromes_auc_tf = cistromes_auc.loc[tf_acc, :]
    else:
        subset_cells = cell_data[cell_data.isin(subset)].index.tolist()
        cell_data = cell_data.loc[subset_cells]
        tf_expr = dgem.loc[name.split('_')[0], subset_cells]
        tf_acc = cistromes_auc.index[cistromes_auc.index.str.contains(
            name + '_(', regex=False)][0]
        cistromes_auc_tf = cistromes_auc.loc[tf_acc, subset_cells]
    random.seed(seed)
    if cell_data is not None:
        categories = list(set(cell_data))
        if color_dict is None:
            color = list(map(
                lambda i: "#" +
                "%06x" % random.randint(0, 0xFFFFFF), range(len(categories))
            ))
            color_dict = dict(zip(categories, color))
        color = [color_dict[x] for x in cell_data]
        patchList = []
        for key in color_dict:
            data_key = mpatches.Patch(color=color_dict[key], label=key)
            patchList.append(data_key)
        if show_dot_plot:
            data = pd.DataFrame(list(zip(tf_expr, cistromes_auc_tf, cell_data)),
                                columns=['TF_Expression', auc_key, 'Variable'])
            sns.scatterplot(x="TF_Expression", y=auc_key, data=data,
                            hue='Variable', palette=color_dict, ax=ax,  **kwargs)
            if ax is None:
                plt.legend(handles=patchList, bbox_to_anchor=(
                    1.04, 1), loc="upper left")
            else:
                ax.legend(handles=patchList, bbox_to_anchor=(
                    1.04, 1), loc="upper left")
        if show_line_plot:
            data = pd.DataFrame(list(zip(tf_expr, cistromes_auc_tf, cell_data)),
                                columns=['TF_Expression', auc_key, 'Variable'])
            sns.regplot(x="TF_Expression", y=auc_key, data=data,
                        scatter_kws={'color': color}, ax=ax, **kwargs)
            if ax is None:
                plt.legend(handles=patchList, bbox_to_anchor=(
                    1.04, 1), loc="upper left")
            else:
                ax.legend(handles=patchList, bbox_to_anchor=(
                    1.04, 1), loc="upper left")
    else:
        if show_dot_plot:
            data = pd.DataFrame(list(zip(tf_expr, cistromes_auc_tf, cell_data)),
                                columns=['TF_Expression', auc_key, 'Variable'])
            sns.scatterplot(x="TF_Expression", y=auc_key,
                            data=data, ax=ax,  **kwargs)
        if show_line_plot:
            data = pd.DataFrame(list(zip(tf_expr, cistromes_auc_tf)),
                                columns=['TF_Expression', auc_key])
            sns.regplot(x="TF_Expression", y=auc_key,
                        data=data, ax=ax, **kwargs)

    corr, _ = pearsonr(tf_expr, cistromes_auc_tf)
    if ax is None:
        plt.xlabel('TF_Expression\nCorrelation ' + str(corr) +
                   '\n' + 'P-value:' + str(_), fontsize=10)
        plt.title(name)
    else:
        ax.set_xlabel('TF_Expression\nCorrelation ' + str(corr) +
                      '\n' + 'P-value:' + str(_), fontsize=10)
        ax.set_title(name)
    if ax is None:
        plt.show()
