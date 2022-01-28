import os
import warnings
import tangram as tg
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ann

def project_genes(adata_scplus: ann.AnnData,
                adata_map: ann.AnnData,
                adata_spatial: ann.AnnData,
                ):

    """
    Function for projecting genes values to spatial data using a tangram mapping (has to be computed in advance)
    """


    # save original regulon names as tangram sets them to lowercase
    dict_genename = {}
    for g in adata_scplus.var.index:
        dict_genename[g.lower()] = g

    # project eRegulons
    ad_ge = tg.project_genes(adata_map, adata_scplus)

    # create new anndata
    adata_gex = ann.AnnData(X=ad_ge.X, obs=adata_spatial.obs, var=ad_ge.var, uns=adata_spatial.uns,
                            obsm=adata_spatial.obsm, obsp=adata_spatial.obsp)

    # get original gene names from scRNAseq as tangram appears to rename them to lowercase etc.
    adata_gex.var['tangram_gene'] = adata_gex.var.index

    new_index = []
    for g in adata_gex.var.index:
        if g in dict_genename:
            new_index.append(dict_genename[g])
        else:
            new_index.append(g)
    adata_gex.var.index = new_index
    adata_gex.var['Gene'] = new_index

    return adata_gex
