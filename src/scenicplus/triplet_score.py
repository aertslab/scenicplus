"""Calculate the TF-region-gene triplet ranking

The triplet ranking is the aggregated ranking of TF-to-region scores, region-to-gene scores and TF-to-region scores. 
The TF-to-gene and TF-to-region scores are defined as the feature importance scores for predicting gene expression from resp. TF expression and region accessibility.
The TF-to-region score is defined as the maximum motif-score-rank for a certain region across all motifs annotated to the TF of interest.

"""

from scenicplus.scenicplus_class import SCENICPLUS
from typing import List
from pycistarget._io.object_converter import dict_motif_enrichment_results_to_mudata
from pycistarget.data_transformations import merge, get_max_rank_for_TF_to_region
import logging
import sys
import numba
import numpy as np

@numba.jit(nopython=True)
def _calculate_cross_species_rank_ratio_with_order_statistics(motif_id_rank_ratios_for_one_region_or_gene: np.ndarray) -> np.ndarray:
    """
    Calculate cross-species combined rank ratio for a region/gene from rank ratios of a certain region/gene scored for
    a certain motif in multiple species with order statistics.
    Code based on applyOrderStatistics function:
      https://github.com/aertslab/orderstatistics/blob/master/OrderStatistics.java
    Paper:
      https://www.nature.com/articles/nbt1203
    :param motif_id_rank_ratios_for_one_region_or_gene:
        Numpy array of rank ratios of a certain region/gene scored for a certain motif in multiple species.
        This array is sorted inplace, so if the original array is required afterwards, provide a copy to this function.
    :return: Cross species combined rank ratio.
    FROM: https://github.com/aertslab/create_cisTarget_databases/blob/master/orderstatistics.py
    """

    # Number of species for which to calculate a cross-species combined rank ratio score.
    rank_ratios_size = motif_id_rank_ratios_for_one_region_or_gene.shape[0]

    if rank_ratios_size == 0:
        return np.float64(1.0)
    else:
        # Sort rank ratios inplace.
        motif_id_rank_ratios_for_one_region_or_gene.sort()

        w = np.zeros((rank_ratios_size + 1,), dtype=np.float64)
        w[0] = np.float64(1.0)
        w[1] = motif_id_rank_ratios_for_one_region_or_gene[rank_ratios_size - 1]

        for k in range(2, rank_ratios_size + 1):
            f = np.float64(-1.0)
            for j in range(0, k):
                f = -(f * (k - j) * motif_id_rank_ratios_for_one_region_or_gene[rank_ratios_size - k]) / (j + 1.0)
                w[k] = w[k] + (w[k - j - 1] * f)

        # Cross species combined rank ratio.
        return w[rank_ratios_size]

rng = np.random.default_rng(seed=123)
def _rank_scores_and_assign_random_ranking_in_range_for_ties(
    scores_with_ties_for_motif_or_track_numpy: np.ndarray
) -> np.ndarray:
        #
        # Create random permutation so tied scores will have a different ranking each time.
        random_permutations_to_break_ties_numpy = rng.permutation(
            scores_with_ties_for_motif_or_track_numpy.shape[0]
        )
        ranking_with_broken_ties_for_motif_or_track_numpy = random_permutations_to_break_ties_numpy[
            (-scores_with_ties_for_motif_or_track_numpy)[
                random_permutations_to_break_ties_numpy].argsort()
        ].argsort().astype(np.uint32)

        return ranking_with_broken_ties_for_motif_or_track_numpy

def calculate_TF_to_region_score(
    scplus_obj: SCENICPLUS,
    ctx_db_fname: str,
    annotations_to_use: List[str] = ['Orthology_annot', 'Direct_annot'],
    eRegulon_metadata_key: str = 'eRegulon_metadata',
    key_added: str = 'TF_to_region_max_rank') -> None:
    """
    Calculated TF-to-regions scores based on the maximum motif-rank-position for each TF and each region.

    Parameters
    ----------
    scplus_obj: SCENICPLUS
        A SCENIC+ object.
    ctx_db_fname: str
        Path to the cistarget ranking database used for motif enrichment analysis.
    annotations_to_use: List[str] = ['Orthology_annot', 'Direct_annot']
        List specifying which motif-to-TF annotations to use.
    eRegulon_metadata_key: str = 'eRegulon_metadata'
        Key in .uns under which to find the eRegulon metadata, scores will be stored in this dataframe
     key_added: str = 'TF_to_region_max_rank'
        Column name under which to store the TF-to-region scores.
    """
    level = logging.INFO
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger('Triplet score')
    log.info("Converting motif enrichment dictionary (scplus_obj.menr) to AnnData ...")
    mdata_motifs = dict_motif_enrichment_results_to_mudata(scplus_obj.menr)
    adata_motifs = merge(mdata_motifs)
    log.info("Getting maximum motif-TF ranking for each TF and region.")
    adata_max_rank = get_max_rank_for_TF_to_region(
        adata_motifs, ctx_db_fname, annotations_to_use = annotations_to_use)
    df_max_rank = adata_max_rank.to_df()
    TF_region_iter = scplus_obj.uns[eRegulon_metadata_key][['TF', 'Region']].to_numpy()
    TF_to_region_score = [df_max_rank.loc[region, TF] for TF, region in TF_region_iter]
    scplus_obj.uns[eRegulon_metadata_key][key_added] = TF_to_region_score

def calculate_triplet_score(
    scplus_obj: SCENICPLUS,
    eRegulon_metadata_key: str = 'eRegulon_metadata',
    key_added: str = 'triplet_score',
    TF2G_score_key: str = 'TF2G_importance',
    R2G_score_key: str = 'R2G_importance',
    TF2R_score_key: str = 'TF_to_region_max_rank') -> None:
    """
    Calculate aggregated ranking score based on TF-to-region, region-to-gene and TF-to-gene score.

    Parameters
    ----------
    scplus_obj: SCENICPLUS
        A SCENIC+ object.
    eRegulon_metadata_key: str = 'eRegulon_metadata'
        Key in .uns under which to find the eRegulon metadata, scores will be stored in this dataframe
    key_added: str = 'triplet_score'
        Column name under which to store the triplet scores.
    TF2G_score_key: str = 'TF2G_importance'
        Columns name containing TF-to-gene scores.
    R2G_score_key: str = 'R2G_importance'
        Columns name containing Region-to-gene scores.
    TF2R_score_key: str = 'TF_to_region_max_rank'
        Columns name containing TF-to-region scores.
    
    Examples
    --------
    >>> calculate_TF_to_region_score(scplus_obj, ctx_db_fname = 'cluster_SCREEN.regions_vs_motifs.rankings.v2.feather')
    >>> calculate_triplet_score(scplus_obj)
    >>> scplus_obj.uns['eRegulon_metadata'][['TF2G_importance', 'R2G_importance', 'TF_to_region_max_rank', 'triplet_score']].sort_values('triplet_score').head()
              TF2G_importance  R2G_importance  TF_to_region_max_rank  triplet_score
        1003         8.689184        0.084093                      0              0
        1514        62.050138        0.089701                     20              1
        1514        62.050138        0.089701                     20              2
        2313        27.908796        0.213620                     17              3
        2496        27.908796        0.213620                     17              4
    """
    TF2G_score = scplus_obj.uns[eRegulon_metadata_key][TF2G_score_key].to_numpy()
    R2G_score = scplus_obj.uns[eRegulon_metadata_key][R2G_score_key].to_numpy()
    TF2R_score = scplus_obj.uns[eRegulon_metadata_key][TF2R_score_key].to_numpy()
    #rank the scores
    TF2G_rank = _rank_scores_and_assign_random_ranking_in_range_for_ties(TF2G_score)
    R2G_rank = _rank_scores_and_assign_random_ranking_in_range_for_ties(R2G_score)
    TF2R_rank = _rank_scores_and_assign_random_ranking_in_range_for_ties(-TF2R_score) #negate because lower score is better
    #create rank ratios
    TF2G_rank_ratio = (TF2G_rank.astype(np.float64) + 1) / TF2G_rank.shape[0]
    R2G_rank_ratio = (R2G_rank.astype(np.float64) + 1) / R2G_rank.shape[0]
    TF2R_rank_ratio = (TF2R_rank.astype(np.float64) + 1) / TF2R_rank.shape[0]
    #create aggregated rank
    rank_ratios = np.array([TF2G_rank_ratio, R2G_rank_ratio, TF2R_rank_ratio])
    aggregated_rank = np.zeros((rank_ratios.shape[1],), dtype = np.float64)
    for i in range(rank_ratios.shape[1]):
            aggregated_rank[i] = _calculate_cross_species_rank_ratio_with_order_statistics(rank_ratios[:, i])
    scplus_obj.uns[eRegulon_metadata_key][key_added] = aggregated_rank.argsort().argsort()




