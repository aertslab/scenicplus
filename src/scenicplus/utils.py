import pandas as pd
import pyranges as pr
import numpy as np
import networkx as nx
from typing import List, Union
from random import sample
import random
from matplotlib import cm
from matplotlib.colors import Normalize, rgb2hex
from numba import njit, float64, int64, prange
from pycisTopic.utils import region_names_to_coordinates
import subprocess
import os
import re
from typing import Callable
import joblib

ASM_SYNONYMS = {
    'hg38': 'GRCh38',
    'hg19': 'GRCh37',
    'mm9': 'MGSCv37',
    'mm10': 'GRCm38',
    'mm39': 'GRCm39',
    'dm6': 'BDGP6',
    'galGal6': 'GRCg6a'}


def flatten_list(t): return [item for sublist in t for item in sublist]


def extend_pyranges(pr_obj: pr.PyRanges,
                    upstream: int,
                    downstream: int):
    """
    A helper function to extend coordinates downstream/upstream in a pyRanges given upstream and downstream
    distances.
    """
    # Split per strand
    positive_pr = pr_obj[pr_obj.Strand == '+']
    negative_pr = pr_obj[pr_obj.Strand == '-']
    # Extend space
    if len(positive_pr) > 0:
        positive_pr.Start = (positive_pr.Start - upstream).astype(np.int32)
        positive_pr.End = (positive_pr.End + downstream).astype(np.int32)
    if len(negative_pr) > 0:
        negative_pr.Start = (negative_pr.Start - downstream).astype(np.int32)
        negative_pr.End = (negative_pr.End + upstream).astype(np.int32)
    extended_pr = pr.PyRanges(
        pd.concat([positive_pr.df, negative_pr.df], axis=0, sort=False))
    return extended_pr


def extend_pyranges_with_limits(pr_obj: pr.PyRanges):
    """
    A helper function to extend coordinates downstream/upstream in a pyRanges with Distance_upstream and
    Distance_downstream columns.
    """
    # Split per strand
    positive_pr = pr_obj[pr_obj.Strand == '+']
    negative_pr = pr_obj[pr_obj.Strand == '-']
    # Extend space
    if len(positive_pr) > 0:
        positive_pr.Start = (positive_pr.Start -
                             positive_pr.Distance_upstream).astype(np.int32)
        positive_pr.End = (positive_pr.End +
                           positive_pr.Distance_downstream).astype(np.int32)
    if len(negative_pr) > 0:
        negative_pr.Start = (negative_pr.Start -
                             negative_pr.Distance_downstream).astype(np.int32)
        negative_pr.End = (negative_pr.End +
                           negative_pr.Distance_upstream).astype(np.int32)
    extended_pr = pr.PyRanges(
        pd.concat([positive_pr.df, negative_pr.df], axis=0, sort=False))
    return extended_pr


def reduce_pyranges_with_limits_b(pr_obj: pr.PyRanges):
    """
    A helper function to reduce coordinates downstream/upstream in a pyRanges with Distance_upstream and
    Distance_downstream columns.
    """
    # Split per strand
    positive_pr = pr_obj[pr_obj.Strand == '+']
    negative_pr = pr_obj[pr_obj.Strand == '-']
    # Extend space
    if len(positive_pr) > 0:
        positive_pr.Start_b = (positive_pr.Start_b +
                               positive_pr.Distance_upstream).astype(np.int32)
        positive_pr.End_b = (positive_pr.End_b -
                             positive_pr.Distance_downstream).astype(np.int32)
    if len(negative_pr) > 0:
        negative_pr.Start_b = (negative_pr.Start_b +
                               negative_pr.Distance_downstream).astype(np.int32)
        negative_pr.End_b = (negative_pr.End_b -
                             negative_pr.Distance_upstream).astype(np.int32)
    extended_pr = pr.PyRanges(
        pd.concat([positive_pr.df, negative_pr.df], axis=0, sort=False))
    return extended_pr


def calculate_distance_with_limits_join(pr_obj: pr.PyRanges):
    """
    A helper function to calculate distances between regions and genes, returning information on what is the relative
    distance to the TSS and end of the gene.
    """
    # Split per strand
    pr_obj_df = pr_obj.df
    distance_df = pd.DataFrame(
        [
            pr_obj_df.Start_b -
            pr_obj_df.Start,
            pr_obj_df.End_b -
            pr_obj_df.Start,
            pr_obj_df.Strand
        ],
        index=['start_dist', 'end_dist', 'strand'])
    distance_df = distance_df.transpose()
    distance_df.loc[:, 'min_distance'] = abs(
        distance_df.loc[:, ['start_dist', 'end_dist']].transpose()).min()
    distance_df.strand[distance_df.strand == '+'] = 1
    distance_df.strand[distance_df.strand == '-'] = -1
    distance_df.loc[:, 'location'] = 0
    distance_df.loc[(distance_df.start_dist > 0) & (
        distance_df.end_dist > 0), 'location'] = 1
    distance_df.loc[(distance_df.start_dist < 0) & (
        distance_df.end_dist < 0), 'location'] = -1
    distance_df.loc[:, 'location'] = distance_df.loc[:,
                                                     'location'] * distance_df.loc[:, 'strand']
    pr_obj.Distance = distance_df.loc[:, 'location'] * \
        distance_df.loc[:, 'min_distance'].astype(np.int32)
    pr_obj = pr_obj[['Chromosome',
                     'Start',
                     'End',
                     'Strand',
                     'Name',
                     'Gene',
                     'Gene_width',
                     'Distance',
                     'Distance_upstream',
                     'Distance_downstream']]
    return pr_obj


def reduce_pyranges_b(pr_obj: pr.PyRanges,
                      upstream: int,
                      downstream: int):
    """
    A helper function to reduce coordinates downstream/upstream in a pyRanges given upstream and downstream
    distances.
    """
    # Split per strand
    positive_pr = pr_obj[pr_obj.Strand == '+']
    negative_pr = pr_obj[pr_obj.Strand == '-']
    # Extend space
    if len(positive_pr) > 0:
        positive_pr.Start_b = (positive_pr.Start_b + upstream).astype(np.int32)
        positive_pr.End_b = (positive_pr.End_b - downstream).astype(np.int32)
    if len(negative_pr) > 0:
        negative_pr.Start_b = (negative_pr.Start_b +
                               downstream).astype(np.int32)
        negative_pr.End_b = (negative_pr.End_b - upstream).astype(np.int32)
    extended_pr = pr.PyRanges(
        pd.concat([positive_pr.df, negative_pr.df], axis=0, sort=False))
    return extended_pr


def calculate_distance_join(pr_obj: pr.PyRanges):
    """
    A helper function to calculate distances between regions and genes.
    """
    # Split per strand
    pr_obj_df = pr_obj.df
    distance_df = pd.DataFrame(
        [
            pr_obj_df.Start_b - pr_obj_df.Start,
            pr_obj_df.End_b - pr_obj_df.Start,
            pr_obj_df.Strand
        ],
        index=['start_dist', 'end_dist', 'strand']
    )
    distance_df = distance_df.transpose()
    distance_df.loc[:, 'min_distance'] = abs(
        distance_df.loc[:, ['start_dist', 'end_dist']].transpose()).min()
    distance_df.strand[distance_df.strand == '+'] = 1
    distance_df.strand[distance_df.strand == '-'] = -1
    distance_df.loc[:, 'location'] = 0
    distance_df.loc[(distance_df.start_dist > 0) & (
        distance_df.end_dist > 0), 'location'] = 1
    distance_df.loc[(distance_df.start_dist < 0) & (
        distance_df.end_dist < 0), 'location'] = -1
    distance_df.loc[:, 'location'] = distance_df.loc[:,
                                                     'location'] * distance_df.loc[:, 'strand']
    pr_obj.Distance = distance_df.loc[:, 'location'] * \
        distance_df.loc[:, 'min_distance'].astype(np.int32)
    pr_obj = pr_obj[['Chromosome',
                     'Start',
                     'End',
                     'Strand',
                     'Name',
                     'Gene',
                     'Gene_width',
                     'Distance']]
    return pr_obj


def coord_to_region_names(coord):
    """
    PyRanges to region names
    """
    if isinstance(coord, pr.PyRanges):
        coord = coord.as_df()
        return list(coord['Chromosome'].astype(str) + ':' + coord['Start'].astype(str) + '-' + coord['End'].astype(str))


def region_names_to_coordinates(region_names):
    chrom = pd.DataFrame([i.split(':', 1)[0]
                         for i in region_names if ':' in i])
    coor = [i.split(':', 1)[1] for i in region_names if ':' in i]
    start = pd.DataFrame([int(i.split('-', 1)[0]) for i in coor])
    end = pd.DataFrame([int(i.split('-', 1)[1]) for i in coor])
    regiondf = pd.concat([chrom, start, end], axis=1, sort=False)
    regiondf.index = [i for i in region_names if ':' in i]
    regiondf.columns = ['Chromosome', 'Start', 'End']
    return (regiondf)


def target_to_overlapping_query(target: Union[pr.PyRanges, List[str]],
                                query: Union[pr.PyRanges, List[str]],
                                fraction_overlap: float = 0.4):
    """
    Return mapping between two sets of regions
    """
    # Read input
    if isinstance(target, str):
        target_pr = pr.read_bed(target)
    if isinstance(target, list):
        target_pr = pr.PyRanges(region_names_to_coordinates(target))
    if isinstance(target, pr.PyRanges):
        target_pr = target
    # Read input
    if isinstance(query, str):
        query_pr = pr.read_bed(query)
    if isinstance(query, list):
        query_pr = pr.PyRanges(region_names_to_coordinates(query))
    if isinstance(query, pr.PyRanges):
        query_pr = query

    join_pr = target_pr.join(query_pr, report_overlap=True)
    if len(join_pr) > 0:
        join_pr.Overlap_query = join_pr.Overlap / \
            (join_pr.End_b - join_pr.Start_b)
        join_pr.Overlap_target = join_pr.Overlap/(join_pr.End - join_pr.Start)
        join_pr = join_pr[(join_pr.Overlap_query > fraction_overlap) | (
            join_pr.Overlap_target > fraction_overlap)]
        join_pr = join_pr[['Chromosome', 'Start', 'End']]
        return join_pr.drop_duplicate_positions()
    else:
        return pr.PyRanges()


def p_adjust_bh(p: float):
    """
    Benjamini-Hochberg p-value correction for multiple hypothesis testing.
    from: pyCistopic: https://github.com/aertslab/pycisTopic/blob/d06246d9860157e028fcfee933fb3e784220b2c3/pycisTopic/diff_features.py#L747
    """
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]


class Groupby:
    # from: http://esantorella.com/2016/06/16/groupby/
    def __init__(self, keys):
        self.keys, self.keys_as_int = np.unique(keys, return_inverse=True)
        self.n_keys = max(self.keys_as_int) + 1
        self.set_indices()

    def set_indices(self):
        self.indices = [[] for i in range(self.n_keys)]
        for i, k in enumerate(self.keys_as_int):
            self.indices[k].append(i)
        self.indices = [np.array(elt) for elt in self.indices]

    def apply(
            self,
            function: Callable,
            vector:np.ndarray,
            broadcast:bool,
            temp_dir: str,
            n_cpu:int=1) -> np.ndarray:
        if n_cpu > 1:
            if broadcast:
                result = np.zeros(len(vector))
                results_unsorted = joblib.Parallel(
                    n_jobs=n_cpu,
                    temp_folder=temp_dir)(
                        joblib.delayed(function)(
                            vector[idx])
                        for idx in self.indices)
                for idx, res in zip(self.indices, results_unsorted):
                    result[idx] = res
            else:
                result = np.zeros(self.n_keys)
                results_unsorted = joblib.Parallel(
                    n_jobs=n_cpu,
                    temp_folder=temp_dir)(
                        joblib.delayed(function)(
                            vector[idx])
                        for _, idx in enumerate(self.indices))
                for (k, _), res in zip(enumerate(self.indices), results_unsorted):
                    result[self.keys_as_int[k]] = res
        else:
            if broadcast:
                result = np.zeros(len(vector))
                for idx in self.indices:
                    result[idx] = function(vector[idx])
            else:
                result = np.zeros(self.n_keys)
                for k, idx in enumerate(self.indices):
                    result[self.keys_as_int[k]] = function(vector[idx])

        return result


def split_eregulons_by_influence(l_eRegulons):
    pos_tf2g__pos_r2g = [eReg for eReg in l_eRegulons if (
        'positive tf2g' in eReg.context and 'positive r2g' in eReg.context)]
    neg_tf2g__pos_r2g = [eReg for eReg in l_eRegulons if (
        'negative tf2g' in eReg.context and 'positive r2g' in eReg.context)]
    pos_tf2g__neg_r2g = [eReg for eReg in l_eRegulons if (
        'positive tf2g' in eReg.context and 'negative r2g' in eReg.context)]
    neg_tf2g__neg_r2g = [eReg for eReg in l_eRegulons if (
        'negative tf2g' in eReg.context and 'negative r2g' in eReg.context)]

    for c, r in zip(['pos_tf2g;pos_r2g', 'neg_tf2g;pos_r2g', 'pos_tf2g;neg_r2g', 'neg_tf2g;neg_r2g'],
                    [pos_tf2g__pos_r2g, neg_tf2g__pos_r2g, pos_tf2g__neg_r2g, neg_tf2g__neg_r2g]):
        yield c, r


def only_keep_extended_eregulons_if_not_direct(l_eRegulons):
    direct_eRegulons = [eReg for eReg in l_eRegulons if not eReg.is_extended]
    extended_eRegulons = [eReg for eReg in l_eRegulons if eReg.is_extended]

    eRegulons_to_return = []
    for (_, direct_eRegulons_split_by_influence), (_, extended_eRegulons_split_by_influence) in zip(
            split_eregulons_by_influence(direct_eRegulons), split_eregulons_by_influence(extended_eRegulons)):
        direct_TFs = [
            eReg.transcription_factor for eReg in direct_eRegulons_split_by_influence]
        extended_TFs = [
            eReg.transcription_factor for eReg in extended_eRegulons_split_by_influence]
        extended_TFs_not_in_direct_TFs = np.isin(
            extended_TFs, direct_TFs, invert=True)

        extended_eRegulons_to_keep = np.array(extended_eRegulons_split_by_influence)[
            extended_TFs_not_in_direct_TFs]
        eRegulons_to_return.extend(
            [*direct_eRegulons_split_by_influence, *extended_eRegulons_to_keep])

    return eRegulons_to_return


def eRegulons_to_networkx(SCENICPLUS_obj,
                          eRegulons_key_to_use: str = 'eRegulons',
                          only_keep_extended_if_not_direct=True,
                          only_TF_TF_interactions: bool = False,
                          selected_TFs: List[str] = None,
                          r2g_importance_key: str = 'importance',
                          only_keep_pos: bool = False):
    if eRegulons_key_to_use not in SCENICPLUS_obj.uns.keys():
        raise ValueError(
            f'key SCENICPLUS_obj.uns["{eRegulons_key_to_use}"] not found!')

    l_eRegulons = only_keep_extended_eregulons_if_not_direct(
        SCENICPLUS_obj.uns[eRegulons_key_to_use]) if only_keep_extended_if_not_direct else SCENICPLUS_obj.uns[eRegulons_key_to_use]

    if not only_keep_pos:
        G = nx.MultiDiGraph()
    else:
        G = nx.DiGraph()
    for c, eRegulons_split_by_influence in split_eregulons_by_influence(l_eRegulons):
        if only_keep_pos and 'neg' in c:
            continue
        TF_to_region = np.array(flatten_list([[(eReg.transcription_factor, region, 0) for region in eReg.target_regions]
                                              for eReg in eRegulons_split_by_influence]))
        if selected_TFs is not None:
            TF_to_region = TF_to_region[np.isin(
                TF_to_region[:, 0], selected_TFs)]

        _tmp_regions = set(TF_to_region[:, 1])

        TFs = list(set(TF_to_region[:, 0]))
        if only_TF_TF_interactions:
            region_to_gene = np.array(flatten_list([flatten_list([[(getattr(r2g, 'region'), getattr(r2g, 'target'), getattr(r2g, r2g_importance_key))] for r2g in eReg.regions2genes
                                                                  if (getattr(r2g, 'target') in TFs and getattr(r2g, 'region') in _tmp_regions)])
                                                    for eReg in eRegulons_split_by_influence]))
        else:
            region_to_gene = np.array(flatten_list([flatten_list([[(getattr(r2g, 'region'), getattr(r2g, 'target'), getattr(r2g, r2g_importance_key))] for r2g in eReg.regions2genes
                                                                  if getattr(r2g, 'region') in _tmp_regions])
                                                    for eReg in eRegulons_split_by_influence]))
        if len(region_to_gene) > 0:
            regions = list(set(region_to_gene[:, 0]))
            genes = list(set(region_to_gene[:, 1]))

            # only keep TF_to_region if region has a target gene
            TF_to_region = TF_to_region[np.isin(
                np.array(TF_to_region)[:, 1], regions)]

            # make sure weight is float (by converting back and forward to numpy arrays this will be converted in str)
            TF_to_region = [(TF, region, float(weight))
                            for TF, region, weight in TF_to_region]
            region_to_gene = [(region, gene, float(weight))
                              for region, gene, weight in region_to_gene]

            G.add_nodes_from(TFs, type='TF')
            G.add_nodes_from(regions, type='region')
            G.add_nodes_from(list(set(genes) - set(TFs)), type='gene')
            G.add_weighted_edges_from(
                TF_to_region, interaction_type=c.split(';')[0])
            G.add_weighted_edges_from(
                region_to_gene, interaction_type=c.split(';')[1])

    return G


def generate_pseudocells_for_numpy(X: np.array,
                                   grouper: Groupby,
                                   nr_cells: list,
                                   nr_pseudobulks: list,
                                   axis=0):
    if len(nr_cells) != len(grouper.indices):
        raise ValueError(
            f'Length of nr_cells ({len(nr_cells)}) should be the same as length of grouper.indices ({len(grouper.indices)})')
    if len(nr_pseudobulks) != len(grouper.indices):
        raise ValueError(
            f'Length of nr_cells ({len(nr_pseudobulks)}) should be the same as length of grouper.indices ({len(grouper.indices)})')
    if axis == 0:
        shape_pseudo = (sum(nr_pseudobulks), X.shape[1])
    elif axis == 1:
        shape_pseudo = (X.shape[0], sum(nr_pseudobulks))
    else:
        raise ValueError(f'axis should be either 0 or 1 not {axis}')
    X_pseudo = np.zeros(shape=shape_pseudo)
    current_index = 0
    for idx, n_pseudobulk, n_cell in zip(grouper.indices, nr_pseudobulks, nr_cells):
        for x in range(n_pseudobulk):
            random.seed(x)
            sample_idx = sample(list(idx), n_cell)
            if axis == 0:
                sample_X = X[sample_idx, :]
            elif axis == 1:
                sample_X = X[:, sample_idx]
            mean_sample_X = sample_X.mean(axis=axis)
            if axis == 0:
                X_pseudo[current_index, :] = mean_sample_X
            elif axis == 1:
                X_pseudo[:, current_index] = mean_sample_X
            current_index += 1  # store index in X_pseudo where mean should be placed
    return X_pseudo


def generate_pseudocell_names(grouper: Groupby,
                              nr_pseudobulks: list,
                              sep='_'):
    if len(nr_pseudobulks) != len(grouper.indices):
        raise ValueError(
            f'Length of nr_cells ({len(nr_pseudobulks)}) should be the same as length of grouper.indices ({len(grouper.indices)})')
    names = []
    for idx, n_pseudobulk, name in zip(grouper.indices, nr_pseudobulks, grouper.keys):
        names.extend([name + sep + str(x) for x in range(n_pseudobulk)])

    return names


def _create_idx_pairs(adjacencies: pd.DataFrame, exp_mtx: pd.DataFrame) -> np.ndarray:
    """
    :precondition: The column index of the exp_mtx should be sorted in ascending order.
            `exp_mtx = exp_mtx.sort_index(axis=1)`
    from pyscenic.utils
    """

    # Create sorted list of genes that take part in a TF-target link.
    genes = set(adjacencies.TF).union(set(adjacencies.target))
    sorted_genes = sorted(genes)

    # Find column idx in the expression matrix of each gene that takes part in a link. Having the column index of genes
    # sorted as well as the list of link genes makes sure that we can map indexes back to genes! This only works if
    # all genes we are looking for are part of the expression matrix.
    assert len(set(exp_mtx.columns).intersection(genes)) == len(genes)
    symbol2idx = dict(zip(sorted_genes, np.nonzero(
        exp_mtx.columns.isin(sorted_genes))[0]))

    # Create numpy array of idx pairs.
    return np.array([[symbol2idx[s1], symbol2idx[s2]] for s1, s2 in zip(adjacencies.TF, adjacencies.target)])


@njit(float64(float64[:], float64[:], float64))
def masked_rho(x: np.ndarray, y: np.ndarray, mask: float = 0.0) -> float:
    """
    Calculates the masked correlation coefficient of two vectors.

    :param x: A vector with the observations of a single variable.
    :param y: Another vector with the same number of observations for a variable.
    :param mask: The value to be masked.
    :return: Pearson correlation coefficient for x and y.
    """
    idx = (x != mask) & (y != mask)
    x_masked = x[idx]
    y_masked = y[idx]
    if (len(x_masked) == 0) or (len(y_masked) == 0):
        return np.nan
    x_demeaned = x_masked - x_masked.mean()
    y_demeaned = y_masked - y_masked.mean()
    cov_xy = np.dot(x_demeaned, y_demeaned)
    std_x = np.sqrt(np.dot(x_demeaned, x_demeaned))
    std_y = np.sqrt(np.dot(y_demeaned, y_demeaned))
    if (std_x * std_y) == 0:
        return np.nan
    return cov_xy / (std_x * std_y)


@njit(float64[:](float64[:, :], int64[:, :], float64), parallel=True)
def masked_rho4pairs(mtx: np.ndarray, col_idx_pairs: np.ndarray, mask: float = 0.0) -> np.ndarray:
    """
    Calculates the masked correlation of columns pairs in a matrix.

    :param mtx: the matrix from which columns will be used.
    :param col_idx_pairs: the pairs of column indexes (nx2).
    :return: array with correlation coefficients (n).
    """
    # Numba can parallelize loops automatically but this is still an experimental feature.
    n = col_idx_pairs.shape[0]
    rhos = np.empty(shape=n, dtype=np.float64)
    for n_idx in prange(n):
        x = mtx[:, col_idx_pairs[n_idx, 0]]
        y = mtx[:, col_idx_pairs[n_idx, 1]]
        rhos[n_idx] = masked_rho(x, y, mask)
    return rhos


def get_interaction_pr(SCENICPLUS_obj,
                       species,
                       assembly,
                       region_to_gene_key='region_to_gene',
                       eRegulons_key='eRegulons',
                       subset_for_eRegulons_regions=True,
                       key_to_add='interaction_pr',
                       inplace=True,
                       pbm_host='http://www.ensembl.org',
                       key_for_color='importance',
                       cmap_pos='Blues',
                       cmap_neg='Reds',
                       scale_by_gene=True,
                       vmin=0, vmax=1):
    if region_to_gene_key not in SCENICPLUS_obj.uns.keys():
        raise ValueError(
            f'key {region_to_gene_key} not found in SCENICPLUS_obj.uns.keys()')

    region_to_gene_df = SCENICPLUS_obj.uns[region_to_gene_key].copy()
    region_to_gene_df.rename(columns={'target': 'Gene'}, inplace=True)

    if subset_for_eRegulons_regions:
        if eRegulons_key not in SCENICPLUS_obj.uns.keys():
            raise ValueError(
                f'key {eRegulons_key} not found in SCENICPLUS_obj.uns.keys()')
        eRegulon_regions = list(set(flatten_list(
            [ereg.target_regions for ereg in SCENICPLUS_obj.uns[eRegulons_key]])))
        region_to_gene_df.index = region_to_gene_df['region']
        region_to_gene_df = region_to_gene_df.loc[eRegulon_regions].reset_index(
            drop=True)

    import pybiomart as pbm
    dataset_name = '{}_gene_ensembl'.format(species)
    server = pbm.Server(host=pbm_host, use_cache=False)
    mart = server['ENSEMBL_MART_ENSEMBL']
    dataset_display_name = getattr(mart.datasets[dataset_name], 'display_name')
    if not (ASM_SYNONYMS[assembly] in dataset_display_name or assembly in dataset_display_name):
        print(
            f'\u001b[31m!! The provided assembly {assembly} does not match the biomart host ({dataset_display_name}).\n Please check biomart host parameter\u001b[0m\nFor more info see: https://m.ensembl.org/info/website/archives/assembly.html')
    dataset = mart[dataset_name]
    if 'external_gene_name' not in dataset.attributes.keys():
        external_gene_name_query = 'hgnc_symbol'
    else:
        external_gene_name_query = 'external_gene_name'
    if 'transcription_start_site' not in dataset.attributes.keys():
        transcription_start_site_query = 'transcript_start'
    else:
        transcription_start_site_query = 'transcription_start_site'
    annot = dataset.query(attributes=['chromosome_name',
                                      'start_position',
                                      'end_position',
                                      'strand',
                                      external_gene_name_query,
                                      transcription_start_site_query,
                                      'transcript_biotype'])
    annot.columns = ['Chromosome', 'Start', 'End', 'Strand',
                     'Gene', 'Transcription_Start_Site', 'Transcript_type']
    annot['Chromosome'] = 'chr' + \
        annot['Chromosome'].astype(str)
    annot = annot[annot.Transcript_type == 'protein_coding']
    annot.Strand[annot.Strand == 1] = '+'
    annot.Strand[annot.Strand == -1] = '-'
    if not any(['chr' in c for c in SCENICPLUS_obj.region_names]):
        annot.Chromosome = annot.Chromosome.str.replace('chr', '')
        
    annot['TSSeqStartEnd'] = np.logical_or(
        annot['Transcription_Start_Site'] == annot['Start'], annot['Transcription_Start_Site'] == annot['End'])
    gene_to_tss = annot[['Gene', 'Transcription_Start_Site']].groupby(
        'Gene').agg(lambda x: list(map(str, x)))
    startEndEq = annot[['Gene', 'TSSeqStartEnd']
                       ].groupby('Gene').agg(lambda x: list(x))
    gene_to_tss['Transcription_Start_Site'] = [np.array(tss[0])[eq[0]][0] if sum(
        eq[0]) >= 1 else tss[0][0] for eq, tss in zip(startEndEq.values, gene_to_tss.values)]
    gene_to_tss.columns = ['TSS_Gene']

    # get gene to strand mapping
    gene_to_strand = annot[['Gene', 'Strand']].groupby(
        'Gene').agg(lambda x: list(map(str, x))[0])

    # get gene to chromosome mapping (should be the same as the regions mapped to the gene)
    gene_to_chrom = annot[['Gene', 'Chromosome']].groupby(
        'Gene').agg(lambda x: list(map(str, x))[0])

    # add TSS for each gene to region_to_gene_df
    region_to_gene_df = region_to_gene_df.join(gene_to_tss, on='Gene')

    # add strand for each gene to region_to_gene_df
    region_to_gene_df = region_to_gene_df.join(gene_to_strand, on='Gene')

    # add chromosome for each gene to region_to_gene_df
    region_to_gene_df = region_to_gene_df.join(gene_to_chrom, on='Gene')

    region_to_gene_df.dropna(axis=0, how='any', inplace=True)
    arr = region_names_to_coordinates(region_to_gene_df['region']).to_numpy()
    chrom, chromStart, chromEnd = np.split(arr, 3, 1)
    chrom = chrom[:, 0]
    chromStart = chromStart[:, 0]
    chromEnd = chromEnd[:, 0]

    sourceChrom = chrom
    sourceStart = np.array(
        list(map(int, chromStart + (chromEnd - chromStart)/2 - 1)))
    sourceEnd = np.array(
        list(map(int, chromStart + (chromEnd - chromStart)/2)))

    # get target chrom, chromStart, chromEnd (i.e. TSS)
    targetChrom = region_to_gene_df['Chromosome']
    targetStart = region_to_gene_df['TSS_Gene'].values
    targetEnd = list(map(str, np.array(list(map(int, targetStart))) + np.array(
        [1 if strand == '+' else -1 for strand in region_to_gene_df['Strand'].values])))

    norm = Normalize(vmin=vmin, vmax=vmax)

    if scale_by_gene:
        grouper = Groupby(
            region_to_gene_df.loc[region_to_gene_df['rho'] >= 0, 'Gene'].to_numpy())
        scores = region_to_gene_df.loc[region_to_gene_df['rho']
                                       >= 0, key_for_color].to_numpy()
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap_pos)

        def _value_to_color(scores):
            S = (scores - scores.min()) / (scores.max() - scores.min())
            return [rgb2hex(mapper.to_rgba(s)) for s in S]

        colors_pos = np.zeros(len(scores), dtype='object')
        for idx in grouper.indices:
            colors_pos[idx] = _value_to_color(scores[idx])

        grouper = Groupby(
            region_to_gene_df.loc[region_to_gene_df['rho'] < 0, 'Gene'].to_numpy())
        scores = region_to_gene_df.loc[region_to_gene_df['rho']
                                       < 0, key_for_color].to_numpy()
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap_neg)

        def _value_to_color(scores):
            S = (scores - scores.min()) / (scores.max() - scores.min())
            return [rgb2hex(mapper.to_rgba(s)) for s in S]

        colors_neg = np.zeros(len(scores), dtype='object')
        for idx in grouper.indices:
            colors_neg[idx] = _value_to_color(scores[idx])

    else:
        scores = region_to_gene_df.loc[region_to_gene_df['rho']
                                       >= 0, key_for_color].to_numpy()
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap_pos)
        colors_pos = [rgb2hex(mapper.to_rgba(s)) for s in scores]

        scores = region_to_gene_df.loc[region_to_gene_df['rho']
                                       < 0, key_for_color].to_numpy()
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap_neg)
        colors_neg = [rgb2hex(mapper.to_rgba(s)) for s in scores]

    region_to_gene_df.loc[region_to_gene_df['rho'] >= 0, 'color'] = colors_pos
    region_to_gene_df.loc[region_to_gene_df['rho'] < 0,  'color'] = colors_neg
    region_to_gene_df['color'] = region_to_gene_df['color'].fillna('#525252')

    # get name for regions (add incremental number to gene in range of regions linked to gene)
    counter = 1
    previous_gene = region_to_gene_df['Gene'].values[0]
    names = []
    for gene in region_to_gene_df['Gene'].values:
        if gene != previous_gene:
            counter = 1
        else:
            counter += 1
        names.append(gene + '_' + str(counter))
        previous_gene = gene
    df_interact = pd.DataFrame(
        data={
            'Chromosome':        chrom,
            'Start':   chromStart,
            'End':     chromEnd,
            'name':         names,
            'score':        np.repeat(0, len(region_to_gene_df)),
            'value':        region_to_gene_df['importance'].values,
            'exp':          np.repeat('.', len(region_to_gene_df)),
            'color':        region_to_gene_df['color'].values,
            'sourceChrom':  sourceChrom,
            'sourceStart':  sourceStart,
            'sourceEnd':    sourceEnd,
            'sourceName':   names,
            'sourceStrand': np.repeat('.', len(region_to_gene_df)),
            'targetChrom':  targetChrom,
            'targetStart':  targetStart,
            'targetEnd':    targetEnd,
            'targetName':   region_to_gene_df['Gene'].values,
            'targetStrand': region_to_gene_df['Strand'].values
        }
    )
    pr_interact = pr.PyRanges(df_interact)

    if inplace:
        SCENICPLUS_obj.uns[key_to_add] = pr_interact
    else:
        return pr_interact


def format_egrns(scplus_obj,
                 eregulons_key: str = 'eRegulons',
                 TF2G_key: str = 'TF2G_adj',
                 key_added: str = 'eRegulon_metadata'):
    """
    A function to format eRegulons to a pandas dataframe
    """

    egrn_list = scplus_obj.uns[eregulons_key]
    TF = [egrn_list[x].transcription_factor for x in range(len(egrn_list))]
    is_extended = [str(egrn_list[x].is_extended)
                   for x in range(len(egrn_list))]
    r2g_data = [pd.DataFrame.from_records(egrn_list[x].regions2genes, columns=[
                                          'Region', 'Gene', 'R2G_importance', 'R2G_rho', 'R2G_importance_x_rho', 'R2G_importance_x_abs_rho']) for x in range(len(egrn_list))]
    egrn_name = [TF[x] + '_extended' if is_extended[x] ==
                 'True' else TF[x] for x in range(len(egrn_list))]
    egrn_name = [egrn_name[x] + '_+' if 'positive tf2g' in egrn_list[x]
                 .context else egrn_name[x] + '_-' for x in range(len(egrn_list))]
    egrn_name = [egrn_name[x] + '_+' if 'positive r2g' in egrn_list[x]
                 .context else egrn_name[x] + '_-' for x in range(len(egrn_list))]
    region_signature_name = [
        egrn_name[x] + '_(' + str(len(set(r2g_data[x].Region))) + 'r)' for x in range(len(egrn_list))]
    gene_signature_name = [
        egrn_name[x] + '_(' + str(len(set(r2g_data[x].Gene))) + 'g)' for x in range(len(egrn_list))]

    for x in range(len(egrn_list)):
        r2g_data[x].insert(0, "TF", TF[x])
        r2g_data[x].insert(1, "is_extended", is_extended[x])
        r2g_data[x].insert(0, "Gene_signature_name", gene_signature_name[x])
        r2g_data[x].insert(0, "Region_signature_name",
                           region_signature_name[x])

    tf2g_data = scplus_obj.uns[TF2G_key].copy()
    tf2g_data.columns = ['TF', 'Gene', 'TF2G_importance', 'TF2G_regulation',
                         'TF2G_rho', 'TF2G_importance_x_abs_rho', 'TF2G_importance_x_rho']
    egrn_metadata = pd.concat([pd.merge(r2g_data[x], tf2g_data[tf2g_data.TF == r2g_data[x].TF[0]], on=[
                              'TF', 'Gene']) for x in range(len(egrn_list))])
    scplus_obj.uns[key_added] = egrn_metadata



def export_eRegulons(scplus_obj: 'SCENICPLUS',
                    out_file: str,
                    assembly: str,
                    bigbed_outfile: str = None,
                    eRegulon_metadata_key: str = 'eRegulon_metadata',
                    eRegulon_signature_key: str = 'eRegulon_signatures',
                    path_bedToBigBed: str = None):
    """
    Export region based eRegulons to bed
    
    Parameters
    ----------
    scplus_obj: SCENICPLUS
        A SCENICPLUS object
    out_file: str
        Path to save file
    assembly: str
        Genomic assembly
    bigbed_outfile: str
        Path to bb file
    eRegulon_metadata_key: str
        Key where the eRegulon metadata is stored
    eRegulon_signature_key
        Key where the eRegulon signatures are stored
    path_bedToBigBed: str
        Path to bedToBigBed bin 
    """
    signatures = list(set(scplus_obj.uns[eRegulon_metadata_key][['Region_signature_name', 'Gene_signature_name']].stack().to_numpy()))
    l_eRegulons = signatures
    direct_eRegulons = [e for e in l_eRegulons if not 'extended' in e]
    extended_eRegulons = [e for e in l_eRegulons if  'extended' in e]
    direct_eRegulons_ng = [x.split('_(')[0] for x in direct_eRegulons]
    extended_eRegulons_ng = [x.split('_(')[0] for x in extended_eRegulons]
    extended_eRegulons_simplified = [re.sub('_extended', '', x) for x in extended_eRegulons_ng]
    extended_TFs_not_in_direct_TFs = np.isin(extended_eRegulons_simplified, direct_eRegulons_ng, invert = True)
    extended_eRegulons_to_keep = np.array(extended_eRegulons)[extended_TFs_not_in_direct_TFs]
    eRegulons_to_keep = [*direct_eRegulons, *extended_eRegulons_to_keep]

    regions = []
    for sign_name in scplus_obj.uns[eRegulon_signature_key]['Region_based'].keys():
            if sign_name in eRegulons_to_keep:
                    tmp = region_names_to_coordinates(scplus_obj.uns[eRegulon_signature_key]['Region_based'][sign_name])
                    tmp.reset_index(drop = True, inplace = True)
                    tmp['Name'] = sign_name.rsplit('_', 1)[0]
                    regions.append(tmp)

    regions = pd.concat(regions)
    regions.groupby(['Chromosome', 'Start', 'End'], as_index = False).agg({'Name': lambda x: ','.join(x)})

    regions = regions.sort_values(['Chromosome', 'Start', 'End'])

    regions.to_csv(out_file, header = False, index = False, sep = '\t')
    
    if bigbed_outfile != None:
        outfolder = bigbed_outfile.rsplit('/', 1)[0]

        cmds = [
            os.path.join(path_bedToBigBed, 'bedToBigBed'),
            out_file,
            'https://hgdownload.cse.ucsc.edu/goldenpath/' + assembly + '/bigZips/' + assembly + '.chrom.sizes',
            bigbed_outfile
        ]
        p = subprocess.Popen(cmds, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        if p.returncode:
            raise ValueError(
                "cmds: %s\nstderr:%s\nstdout:%s" % (
                    " ".join(cmds), stderr, stdout)
            )
    return regions

def annotate_eregulon_by_influence(ereg):
        TF = ereg.transcription_factor
        r2g = '+' if 'positive r2g' in ereg.context else '-'
        tf2g = '+' if 'positive tf2g' in ereg.context else '-'
        is_extended = ereg.is_extended
        if not is_extended:
                return f"{TF}_{tf2g}_{r2g}"
        else:
                return f"{TF}_{tf2g}_{r2g}_extended"

def timestamp(dt):
    return f"{dt.year}{dt.month}{dt.day}_{dt.hour}{dt.minute}{dt.second}"
