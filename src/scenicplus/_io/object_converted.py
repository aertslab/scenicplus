"""Convert from SCENIC+ class to MuData

Given a SCENIC+ class this function will convert to a MuData object.

"""
from scenicplus.scenicplus_class import SCENICPLUS
from mudata import MuData, AnnData
from pycistarget._io import dict_motif_enrichment_results_to_mudata
import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import Tuple

def scenicplus_object_to_mudata(
    scplus_obj: SCENICPLUS,
    search_space_key: str = 'search_space',
    region_to_gene_key: str = 'region_to_gene',
    TF_to_gene_key: str = 'TF2G_adj',
    eRegulon_AUC_key: str = 'eRegulon_AUC',
    eRegulon_metadata_key: str = 'eRegulon_metadata') -> Tuple[MuData, MuData]:
    """
    Convert scplus_obj to MuData

    Parameters
    ----------
        scplus_obj: SCENICPLUS 
            a scenicplus object
        search_space_key: str = 'search_space' 
            key under which the search space is stored in .uns
        region_to_gene_key: str = 'region_to_gene' 
            key under which the region to gene importances are stored in .uns
        TF_to_gene_key: str = 'TF2G_adj' 
            key under which the TF to gene importances are stored in .uns
        eRegulon_AUC_key: str = 'eRegulon_AUC' 
            key under which the eRegulon AUC values are stored in .uns
        eRegulon_metadata_key: str = 'eRegulon_metadata' 
            key under which the eRegulon metadata is stored in .uns
    
    Returns
    -------
        Tuple[MuData, MuData]
            Mudata with gene expression/region accessibility data and eRegulons and MuData containing motif enrichment results.
    
    """
    not_stored = set(scplus_obj.uns.keys()) - set([search_space_key, region_to_gene_key, TF_to_gene_key, eRegulon_AUC_key, eRegulon_metadata_key])
    print(
        f"Following items in scplus_obj.uns will not be stored, store them seperatly if you want to keep them.\n\t{', '.join(not_stored)}")
    mudata_constructor = {}
    
    #Construct ACC AnnData
    adata_ACC = AnnData(
        X = scplus_obj.X_ACC.T, dtype = np.int32,
        obs = pd.DataFrame(index = scplus_obj.cell_names),
        var = scplus_obj.metadata_regions.infer_objects())
    mudata_constructor['ACC'] = adata_ACC
    
    #Construct EXP AnnData
    adata_EXP = AnnData(
        X = scplus_obj.X_EXP, dtype = np.int32,
        obs = pd.DataFrame(index = scplus_obj.cell_names),
        var = scplus_obj.metadata_genes.infer_objects())
    mudata_constructor['EXP'] = adata_EXP

    #Construct eRegulon AUC AnnDatas
    adata_AUC_region = AnnData(
        X = np.array(scplus_obj.uns[eRegulon_AUC_key]['Region_based'], dtype = np.float32), dtype = np.float32,
        obs = pd.DataFrame(index = scplus_obj.uns[eRegulon_AUC_key]['Region_based'].index),
        var = pd.DataFrame(index = scplus_obj.uns[eRegulon_AUC_key]['Region_based'].columns))
    mudata_constructor['AUC_target_regions'] = adata_AUC_region
    adata_AUC_gene = AnnData(
        X = np.array(scplus_obj.uns[eRegulon_AUC_key]['Gene_based'], dtype = np.float32), dtype = np.float32,
        obs = pd.DataFrame(index = scplus_obj.uns[eRegulon_AUC_key]['Gene_based'].index),
        var = pd.DataFrame(index = scplus_obj.uns[eRegulon_AUC_key]['Gene_based'].columns))
    mudata_constructor['AUC_target_genes'] = adata_AUC_gene

    #construct uns
    uns = OrderedDict()
    uns['search_space'] = scplus_obj.uns[search_space_key].explode('Distance').infer_objects()
    uns['region_to_gene'] = scplus_obj.uns[region_to_gene_key].explode('Distance').infer_objects()
    uns['TF_to_gene'] = scplus_obj.uns[TF_to_gene_key].infer_objects()
    uns['eRegulon_metadata'] = scplus_obj.uns[eRegulon_metadata_key].infer_objects()

    mdata = MuData(
        mudata_constructor,
        obs = scplus_obj.metadata_cell.infer_objects(),
        obsm = {key: np.array(scplus_obj.dr_cell[key], dtype = np.float32) for key in scplus_obj.dr_cell.keys()},
        uns = uns)

    mdata_menr = dict_motif_enrichment_results_to_mudata(scplus_obj.menr)

    return mdata, mdata_menr
