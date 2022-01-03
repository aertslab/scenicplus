from pycisTopic.diff_features import *
from pycisTopic.signature_enrichment import *

def get_eRegulons_as_signatures(scplus_obj: 'SCENICPLUS',
                              eRegulon_metadata_key: str ='eRegulon_metadata', 
                              key_added: str = 'eRegulon_signatures'):
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
