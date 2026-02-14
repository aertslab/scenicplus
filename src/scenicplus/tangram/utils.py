import scanpy as sc
import anndata as ann
import pandas as pd
import numpy as np
import loompy as lp
import json
import re
import anndata as ann


def get_auc_anndata_from_scenicplus(
        scplus_obj,
        regulon_entity: str = 'Gene_based'
        ):

    """
    Returns simple anndata that can be used for projecting AUC values to spatial data
    """

    adata_scplus = ann.AnnData(X=scplus_obj.uns['eRegulon_AUC'][regulon_entity])

    return adata_scplus


def get_anndata_from_loom(
        path_loom: str
        ):

    """
    Returns simple anndata containing expression data from loom
    """

    loom = lp.connect(path_loom, validate = False, mode = 'r')
    exprMat = pd.DataFrame( loom[:,:], index=loom.ra['Gene'], columns=loom.ca['CellID']).T
    loom.close()

    adata = ann.AnnData(X=exprMat)

    return adata

    

def update_spatial_loom_with_eRegulons(adata_tg_auc: ann.AnnData,
                                  path_scenicplus_loom: str,
                                  path_spatial_loom: str
                                  ):
    """
    Function for updating spatial loom with eRegulon information, will write to 'path_spatial_loom'
    """

    # open scenic plus loom (read only)
    loom_scplus = lp.connect(path_scenicplus_loom, validate=False, mode='r')

    # get metadata
    metadata = json.loads(loom_scplus.attrs['MetaData'])

    # open and update spatial loom
    loom = lp.connect(path_spatial_loom, validate=False, mode='r+')

    # add regulons thresholds as metadata
    metadata_spatial = json.loads(loom.attrs['MetaData'])
    metadata_spatial['regulonThresholds'] = metadata['regulonThresholds']
    loom.attrs['MetaData'] = json.dumps(metadata_spatial)

    # format eRegulon names, gene based
    ereg_names = [ re.sub(r'_\([0-9]+g\)', '', ereg) for ereg in adata_tg_auc.var_names ]
    # format eRegulon names, region based
    ereg_names = [ re.sub(r'_\([0-9]+r\)', '', ereg) for ereg in ereg_names ]
    
    # add AUC values, regulons
    df_ereg = pd.DataFrame(adata_tg_auc.X, columns=ereg_names, index=adata_tg_auc.obs_names)
    loom.ca['RegulonsAUC'] = np.array([tuple(row) for row in df_ereg.values],
                                      dtype=np.dtype(list(zip(df_ereg.columns, df_ereg.dtypes))))
    loom.ra['Regulons'] = loom_scplus.ra['Regulons']
    loom.ra['linkedGene'] = loom_scplus.ra['linkedGene']
    
    # close spatial loom
    loom.close()
    
    # close scenicplus loom
    loom_scplus.close()
