"""Score eRegulon target genes and regions in cells using AUC algorithm.

"""

from pycisTopic.signature_enrichment import signature_enrichment
from pyscenic.binarization import binarize
from typing import Literal, Dict, List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass


# Temporary class so old code from pycisTopic works.
# This class replaces the "CistopicImputedFeatures" from pycisTopic,
# Better rewrite the code from pycisTopic and put in seperate repo (ctxcore?).
# mtx is a numpy array storing the ranking across cells, columns should be cells and
# rows should be regions or genes.
# TODO!
@dataclass
class ranked_data:
    mtx: np.ndarray
    feature_names: List[str]
    cell_names: List[str]

def rank_data(
        df: pd.DataFrame,
        axis: Literal[0, 1] = 1,
        seed: int = 123) -> ranked_data:
    """
    """
    # Initialize random number generator, for handling ties
    rng = np.random.default_rng(seed=seed)

    # Function to make rankings per array
    def rank_scores_and_assign_random_ranking_in_range_for_ties(
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
        ].argsort().astype(imputed_acc_obj_ranking_db_dtype)

        return ranking_with_broken_ties_for_motif_or_track_numpy

    # Get dtype of the scores
    imputed_acc_obj_ranking_db_dtype = 'uint32'
    mtx = df.to_numpy()
    ranking = np.zeros_like(mtx)

    if axis == 0:
        for i in range(mtx.shape[1]):
            ranking[:, i] = rank_scores_and_assign_random_ranking_in_range_for_ties(
                mtx[:, i])
    elif axis == 1:
        for i in range(mtx.shape[0]):
            ranking[i, :] = rank_scores_and_assign_random_ranking_in_range_for_ties(
                mtx[i, :])
    else:
        raise ValueError(f"Axis can only be 0 or 1 not {axis}")

    return ranked_data(
        mtx=ranking.T,
        feature_names=df.columns if axis == 1 else df.index,
        cell_names=df.index if axis == 1 else df.columns)

def get_eRegulons_as_signatures(
        eRegulons: pd.DataFrame
) -> Dict[str, Dict[str, List[str]]]:
    """
    """
    region_signatures: Dict[str, List[str]] = eRegulons.groupby("Region_signature_name")["Region"].apply(
        lambda x: list(set(x))).to_dict()
    gene_signatures: Dict[str, List[str]] = eRegulons.groupby("Gene_signature_name")["Gene"].apply(
        lambda x: list(set(x))).to_dict()
    return {"Gene_based": gene_signatures, "Region_based": region_signatures}

def score_eRegulons(
        eRegulons: pd.DataFrame,
        gex_mtx: pd.DataFrame,
        acc_mtx: pd.DataFrame,
        auc_threshold: float = 0.05,
        normalize: bool = False,
        n_cpu: int = 1
) -> Dict[str, pd.DataFrame]:
    """
    """
    eRegulon_signatures = get_eRegulons_as_signatures(eRegulons=eRegulons)
    gex_ranking = rank_data(gex_mtx)
    acc_ranking = rank_data(acc_mtx)
    gex_AUC = signature_enrichment(
        gex_ranking,
        eRegulon_signatures["Gene_based"],
        enrichment_type='gene',
        auc_threshold=auc_threshold,
        normalize=normalize,
        n_cpu=n_cpu)
    acc_AUC = signature_enrichment(
        acc_ranking,
        eRegulon_signatures["Region_based"],
        enrichment_type='gene',
        auc_threshold=auc_threshold,
        normalize=normalize,
        n_cpu=n_cpu)
    return {"Gene_based": gex_AUC, "Region_based": acc_AUC}

def binarize_AUC(scplus_obj,
                 auc_key: Optional[str] = 'eRegulon_AUC',
                 out_key: Optional[str] = 'eRegulon_AUC_thresholds',
                 signature_keys: Optional[List[str]] = ['Gene_based', 'Region_based'],
                 n_cpu: Optional[int] = 1):
    """
    Binarize eRegulons using AUCell

    Parameters
    ----------
    scplus_obj: `class::SCENICPLUS`
        A SCENICPLUS object with eRegulons AUC.
    auc_key: str, optional
        Key where the AUC values are stored
    out_key: str, optional
        Key where the AUCell thresholds will be stored (in `scplus_obj.uns`)
    signature_keys: List, optional
        Keys to extract AUC values from. Default: ['Gene_based', 'Region_based']
    n_cpu: int
        The number of cores to use. Default: 1
    """
    if not out_key in scplus_obj.uns.keys():
        scplus_obj.uns[out_key] = {}
    for signature in signature_keys:
        auc_mtx = scplus_obj.uns[auc_key][signature]
        _, auc_thresholds = binarize(auc_mtx, num_workers=n_cpu)
        scplus_obj.uns[out_key][signature] = auc_thresholds
