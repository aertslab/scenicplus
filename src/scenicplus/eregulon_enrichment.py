from pycisTopic.diff_features import *
from pycisTopic.signature_enrichment import *
from pyscenic.binarization import binarize

def get_eRegulons_as_signatures(scplus_obj: 'SCENICPLUS',
                              eRegulon_metadata_key: str ='eRegulon_metadata', 
                              key_added: str = 'eRegulon_signatures'):
    """
    Format eRegulons for scoring
    
    Parameters
    ----------
    scplus_obj: `class::SCENICPLUS`
        A SCENICPLUS object with eRegulons metadata computed.
    eRegulons_metadata_key: str, optional
    	Key where the eRegulon metadata is stored (in `scplus_obj.uns`)
    key_added: str, optional
    	Key where formated signatures will be stored (in `scplus_obj.uns`)
    """
    region_signatures = {x: list(set(scplus_obj.uns[eRegulon_metadata_key][scplus_obj.uns[eRegulon_metadata_key].Region_signature_name == x]['Region'])) for x in list(set(scplus_obj.uns[eRegulon_metadata_key].Region_signature_name))}
    gene_signatures = {x: list(set(scplus_obj.uns[eRegulon_metadata_key][scplus_obj.uns[eRegulon_metadata_key].Gene_signature_name == x]['Gene'])) for x in list(set(scplus_obj.uns[eRegulon_metadata_key].Gene_signature_name))}

    if not key_added in scplus_obj.uns.keys():
        scplus_obj.uns[key_added] = {}
        
    scplus_obj.uns[key_added]['Gene_based'] = gene_signatures
    scplus_obj.uns[key_added]['Region_based'] = region_signatures 
        
    
def score_eRegulons(scplus_obj: 'SCENICPLUS',
                    ranking: 'CistopicImputedFeatures',
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
    eRegulons_metadata_key: `class::CistopicImputedFeatures`
    	Precomputed region/gene ranking.
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
    
    scplus_obj.uns[key_added][key] = signature_enrichment(ranking,
                        scplus_obj.uns[eRegulon_signatures_key][key],
                        enrichment_type = 'gene',
                        auc_threshold = auc_threshold,
                        normalize = normalize,
                        n_cpu = n_cpu)

def binarize_AUC(scplus_obj: 'SCENICPLUS', 
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