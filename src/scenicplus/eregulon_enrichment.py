"""Score eRegulon target genes and regions in cells using AUC algorithm.

"""

from pycisTopic.diff_features import *
from pycisTopic.signature_enrichment import *
from pyscenic.binarization import binarize

from .scenicplus_class import SCENICPLUS


def get_eRegulons_as_signatures(scplus_obj: SCENICPLUS,
                                eRegulon_metadata_key: str = 'eRegulon_metadata',
                                key_added: str = 'eRegulon_signatures'):
    """
    Format eRegulons for scoring

    Parameters
    ----------
    scplus_obj: `class::SCENICPLUS`
        A SCENICPLUS object with eRegulons metadata computed.
    eRegulon_metadata_key: str, optional
        Key where the eRegulon metadata is stored (in `scplus_obj.uns`)
    key_added: str, optional
        Key where formated signatures will be stored (in `scplus_obj.uns`)
    """
    region_signatures = {x: list(set(scplus_obj.uns[eRegulon_metadata_key][scplus_obj.uns[eRegulon_metadata_key].Region_signature_name == x]['Region'])) for x in list(
        set(scplus_obj.uns[eRegulon_metadata_key].Region_signature_name))}
    gene_signatures = {x: list(set(scplus_obj.uns[eRegulon_metadata_key][scplus_obj.uns[eRegulon_metadata_key].Gene_signature_name == x]['Gene'])) for x in list(
        set(scplus_obj.uns[eRegulon_metadata_key].Gene_signature_name))}

    if not key_added in scplus_obj.uns.keys():
        scplus_obj.uns[key_added] = {}

    scplus_obj.uns[key_added]['Gene_based'] = gene_signatures
    scplus_obj.uns[key_added]['Region_based'] = region_signatures


def make_rankings(scplus_obj: SCENICPLUS,
                  target: str = 'region',
                  seed: int = 123):
    """
    A function to generate rankings per cell based on the imputed accessibility scores per region
    or the gene expression per cell.

    Parameters
    ---------
    scplus_obj: :class:`SCENICPLUS`
        A :class:`SCENICPLUS` object with motif enrichment results from pycistarget (`scplus_obj.menr`).
    target: str, optional
        Whether rankings should be done based on gene expression or region accessibilty. Default: 'region'
    seed: int, optional
        Random seed to ensure reproducibility of the rankings when there are ties

    Return
    ------
       CistopicImputedFeatures
        A :class:`CistopicImputedFeatures` containing with ranking values rather than scores.
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

    # Create zeroed imputed object rankings database.
    if target == 'region':
        imputed_acc_ranking = CistopicImputedFeatures(
            np.zeros((len(scplus_obj.region_names),
                     len(scplus_obj.cell_names)), dtype=np.int32),
            scplus_obj.region_names,
            scplus_obj.cell_names,
            'Ranking')
    if target == 'gene':
        imputed_acc_ranking = CistopicImputedFeatures(
            np.zeros((len(scplus_obj.gene_names), len(scplus_obj.cell_names)), dtype=np.int32),
            scplus_obj.gene_names,
            scplus_obj.cell_names,
            'Ranking')

    # Get dtype of the scores
    imputed_acc_obj_ranking_db_dtype = 'uint32'

    # Convert to csc
    if target == 'region':
        if sparse.issparse(scplus_obj.X_ACC):
            mtx = scplus_obj.X_ACC.tocsc()
        else:
            mtx = scplus_obj.X_ACC
    elif target == 'gene':
        if sparse.issparse(scplus_obj.X_EXP):
            mtx = scplus_obj.X_EXP.T.tocsc()
        else:
            mtx = scplus_obj.X_EXP.T

    # Rank all scores per motif/track and assign a random ranking in range for regions/genes with the same score.
    for col_idx in range(len(imputed_acc_ranking.cell_names)):
        imputed_acc_ranking.mtx[:, col_idx] = rank_scores_and_assign_random_ranking_in_range_for_ties(
            mtx[:, col_idx].toarray().flatten() if sparse.issparse(
                mtx) else mtx[:, col_idx].flatten()
        )

    return imputed_acc_ranking


def score_eRegulons(scplus_obj: SCENICPLUS,
                    ranking: CistopicImputedFeatures,
                    inplace: bool = True,
                    eRegulon_signatures_key: str = 'eRegulon_signatures',
                    key_added: str = 'eRegulon_AUC',
                    enrichment_type: str = 'region',
                    auc_threshold: float = 0.05,
                    normalize: bool = False,
                    n_cpu: int = 1):
    """
    Score eRegulons using AUCell

    Parameters
    ----------
    scplus_obj: `class::SCENICPLUS`
        A SCENICPLUS object with formatted eRegulons.
    ranking: `class::CistopicImputedFeatures`
        A CistopicImputedFeatures object containing rankings, generated using the function make_rankings.
    inplace: bool, optional
        If set to True store result in scplus_obj, otherwise it is returned.
    eRegulon_signatures_key: str, optional
        Key where formated signatures are stored (in `scplus_obj.uns`)
    key_added: str, optional
        Key where formated AUC values will be stored (in `scplus_obj.uns`)
    enrichment_type: str, optional
        Whether region or gene signatures are being used
    auc_threshold: float
        The fraction of the ranked genome to take into account for the calculation of the Area Under the recovery Curve. Default: 0.05
    normalize: bool
        Normalize the AUC values to a maximum of 1.0 per regulon. Default: False
    n_cpu: int
        The number of cores to use. Default: 1
    """
    if not key_added in scplus_obj.uns.keys():
        scplus_obj.uns[key_added] = {}

    if enrichment_type == 'region':
        key = 'Region_based'
    if enrichment_type == 'gene':
        key = 'Gene_based'
    if inplace:
        scplus_obj.uns[key_added][key] = signature_enrichment(ranking,
                                                            scplus_obj.uns[eRegulon_signatures_key][key],
                                                            enrichment_type='gene',
                                                            auc_threshold=auc_threshold,
                                                            normalize=normalize,
                                                            n_cpu=n_cpu)
    else:
        return signature_enrichment(ranking,
                                    scplus_obj.uns[eRegulon_signatures_key][key],
                                    enrichment_type='gene',
                                    auc_threshold=auc_threshold,
                                    normalize=normalize,
                                    n_cpu=n_cpu)

def binarize_AUC(scplus_obj: SCENICPLUS,
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
