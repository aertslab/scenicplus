# -*- coding: utf-8 -*-
"""Merging, scoring and assessing TF correlations of cistromes

"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pycistarget
from pycistarget.utils import get_TF_list, get_motifs_per_TF
from random import sample
import seaborn as sns
from scipy.stats import pearsonr
from typing import List, Dict, Set, Iterable
from scenicplus.utils import p_adjust_bh
from scenicplus.scenicplus_class import SCENICPLUS
import numpy as np
import pandas as pd
from dataclasses import dataclass
import anndata
from scipy import sparse

@dataclass
class Cistrome:
    """
    Dataclass for intermediate use
    """
    tf_name: str
    target_regions: Set[str]
    extended: bool

def _signatures_to_iter(menr):
    for x in menr.keys():
        if isinstance(menr[x], pycistarget.motif_enrichment_dem.DEM):
            for y in menr[x].motif_enrichment.keys():
                yield menr[x].motif_enrichment[y], menr[x].motif_hits["Region_set"][y]
        elif isinstance(menr[x], dict):
            for y in menr[x].keys():
                if not isinstance(menr[x][y], pycistarget.motif_enrichment_cistarget.cisTarget):
                    raise ValueError(f'Only motif enrichment results from pycistarget or DEM are allowed, not {type(menr[x][y])}')
                yield menr[x][y].motif_enrichment, menr[x][y].motif_hits["Region_set"]
        else:
            raise ValueError(f'Only motif enrichment results from pycistarget or DEM are allowed, not {type(menr[x])}')

def _get_cistromes(
        motif_enrichment_table: pd.DataFrame,
        motif_hits: Dict[str, str],
        scplus_regions: Set[str],
        direct_annotation: List[str],
        extended_annotation: List[str]) -> List[Cistrome]:
    """
    Helper function to get region TF target regions based on motif hits

    Parameters
    ----------
        motif_enrichment_table: 
            Pandas DataFrame containing motif enrichment data
        motif_hits: 
            dict of motif hits (mapping motifs to regions)
        scplus_regions:
            set of regions in the scplus_obj
        direct_annotation: 
            list of annotations to use as 'direct'
        extended_annotation: 
            list of annotations to use as 'extended'
            
    Returns
    -------
        List of cistromes
    """
    tfs_direct = get_TF_list(
        motif_enrichment_table = motif_enrichment_table,
        annotation = direct_annotation)
    tfs_extended = get_TF_list(
        motif_enrichment_table = motif_enrichment_table,
        annotation = extended_annotation)
    cistromes = []
    for tf_name in tfs_direct:
        motifs_annotated_to_tf = get_motifs_per_TF(
            motif_enrichment_table = motif_enrichment_table,
            tf = tf_name,
            motif_column = "Index",
            annotation = direct_annotation)
        target_regions_motif: Set[str] = set()
        for motif in motifs_annotated_to_tf:
            if motif in motif_hits.keys():
                target_regions_motif.update(motif_hits[motif])
            else:
                raise ValueError(f"Motif enrichment table and motif hits don't match for the TF: {tf_name}")
        cistromes.append(
            Cistrome(
                tf_name = tf_name,
                target_regions = target_regions_motif & scplus_regions,
                extended = False))
    for tf_name in tfs_extended:
        motifs_annotated_to_tf = get_motifs_per_TF(
            motif_enrichment_table = motif_enrichment_table,
            tf = tf_name,
            motif_column = "Index",
            annotation = extended_annotation)
        target_regions_motif: Set[str] = set()
        for motif in motifs_annotated_to_tf:
            if motif in motif_hits.keys():
                target_regions_motif.update(motif_hits[motif])
            else:
                raise ValueError(f"Motif enrichment table and motif hits don't match for the TF: {tf_name}")
        cistromes.append(
            Cistrome(
                tf_name = tf_name,
                target_regions = target_regions_motif & scplus_regions,
                extended = True))
    return cistromes

def _merge_cistromes(cistromes: List[Cistrome]) -> Iterable[Cistrome]:
    a_cistromes = np.array(cistromes, dtype = 'object')
    tf_names = np.array([cistrome.tf_name for cistrome in a_cistromes])
    tf_names_sorted_idx = np.argsort(tf_names)
    a_cistromes = a_cistromes[tf_names_sorted_idx]
    tf_names = tf_names[tf_names_sorted_idx]
    u_tf_names, idx_tf_names = np.unique(tf_names, return_index = True)
    for i, tf_name in enumerate(u_tf_names):
        if i < len(u_tf_names) - 1:
            cistromes_tf = a_cistromes[idx_tf_names[i]:idx_tf_names[i + 1]]
        else:
            cistromes_tf = a_cistromes[idx_tf_names[i]:]
        assert all([x.tf_name == tf_name for x in cistromes_tf])
        assert all([x.extended == cistromes_tf[0].extended for x in cistromes_tf])
        yield Cistrome(
            tf_name = tf_name,
            target_regions = set.union(
                *[cistrome.target_regions for cistrome in cistromes_tf]),
            extended = cistromes_tf[0].extended)

def _cistromes_to_adata(cistromes: List[Cistrome]) -> anndata.AnnData:
    tf_names = [cistrome.tf_name for cistrome in cistromes]
    union_target_regions= list(set.union(
            *[cistrome.target_regions for cistrome in cistromes]))
    cistrome_hit_mtx = np.zeros(
        (len(union_target_regions), len(tf_names)),
        dtype = bool)
    for i in range(len(tf_names)):
        cistrome_hit_mtx[:, i] = [
            region in cistromes[i].target_regions 
            for region in union_target_regions]
    return anndata.AnnData(
        X = sparse.csc_matrix(cistrome_hit_mtx), dtype = bool,
        obs = pd.DataFrame(index = list(union_target_regions)),
        var = pd.DataFrame(index = tf_names))

def get_and_merge_cistromes(
        scplus_obj: SCENICPLUS,
        cistromes_key: str = 'Unfiltered',
        direct_annotation: List[str] = ['Direct_annot'],
        extended_annotation: List[str] = ['Orthology_annot']):
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
    direct_annotation: list
        A list of strings with motif-to-TF annotation to use as direct annotation
    extended_annotation: list
        A list of strings with motif-to-TF annotation to use as extended annotation
    """
    menr = scplus_obj.menr
    # get cistromes
    cistromes = []
    for motif_enrichment_table, motif_hits in _signatures_to_iter(menr):
        cistromes.extend(
            _get_cistromes(
                motif_enrichment_table = motif_enrichment_table,
                motif_hits = motif_hits,
                scplus_regions = set(scplus_obj.region_names),
                direct_annotation = direct_annotation,
                extended_annotation = extended_annotation))
    # merge cistromes. Seperatly for direct and extended
    direct_cistromes = [cistrome for cistrome in cistromes if not cistrome.extended]
    extended_cistromes = [cistrome for cistrome in cistromes if cistrome.extended]
    merged_direct_cistromes = list(_merge_cistromes(direct_cistromes))
    merged_extended_cistromes = list(_merge_cistromes(extended_cistromes))
    adata_direct_cistromes = _cistromes_to_adata(merged_direct_cistromes)
    adata_extended_cistromes = _cistromes_to_adata(merged_extended_cistromes)
    if 'Cistromes' not in scplus_obj.uns.keys():
        scplus_obj.uns['Cistromes'] = {}
    scplus_obj.uns['Cistromes'][cistromes_key]['direct'] = adata_direct_cistromes
    scplus_obj.uns['Cistromes'][cistromes_key]['extended'] = adata_extended_cistromes

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
