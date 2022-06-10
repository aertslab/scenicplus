"""Calculate differentially expressed genes (DEGs) and differentially accessible regions (DARs).

"""

import anndata
import scanpy as sc
from typing import Optional, List
import logging
import numpy as np
import pandas as pd
import sys

from .scenicplus_class import SCENICPLUS

pd.options.mode.chained_assignment = None


def _format_df(df, key, adjpval_thr, log2fc_thr):
    """
    A helper function to format differential test results
    """
    df.index = df['names']
    df = df[['logfoldchanges', 'pvals_adj']]
    df.columns = ['Log2FC', 'Adjusted_pval']
    df['Contrast'] = key
    df.index.name = None
    df = df.loc[df['Adjusted_pval'] <= adjpval_thr]
    df = df.loc[df['Log2FC'] >= log2fc_thr]
    df = df.sort_values(
        ['Log2FC', 'Adjusted_pval'], ascending=[False, True]
    )
    return df


def get_differential_features(scplus_obj: SCENICPLUS,
                              variable,
                              use_hvg: Optional[bool] = True,
                              contrast_type: Optional[List] = ['DARs', 'DEGs'],
                              adjpval_thr: Optional[float] = 0.05,
                              log2fc_thr: Optional[float] = np.log2(1.5),
                              min_cells: Optional[int] = 2
                              ):
    """
    Get DARs of DEGs given reference variable. 

    Parameters
    ---------
    scplus_obj: `class::SCENICPLUS`
        A SCENICPLUS object.
    variable: str
        Variable to compute DARs/DEGs by (has to be included in scplus_obj.metadata_cell)
    use_hvg: bool, optional
        Whether to use only highly variable genes/regions
    contrast_type: list, optional
        Wheter to compute DARs and/or DEGs per variable
    adjpval_thr: float, optional
        P-value threshold
    log2fc_thr
        Log2FC threshold
    """
    # Create logger
    level = logging.INFO
    log_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    log = logging.getLogger('SCENIC+')

    for contrast in contrast_type:
        log.info('Calculating ' + contrast + ' for variable ' + variable)
        if contrast == 'DEGs':
            adata = anndata.AnnData(X=scplus_obj.X_EXP.copy(), obs=pd.DataFrame(
                index=scplus_obj.cell_names), var=pd.DataFrame(index=scplus_obj.gene_names))
            min_disp = 0.5
        if contrast == 'DARs':
            adata = anndata.AnnData(X=scplus_obj.X_ACC.copy().T, obs=pd.DataFrame(
                index=scplus_obj.cell_names), var=pd.DataFrame(index=scplus_obj.region_names))
            min_disp = 0.05
        adata.obs = scplus_obj.metadata_cell

        # remove annotations with less than 'min_cells'
        label_count = adata.obs[variable].value_counts()
        keeplabels = [label for label, count in zip(label_count.index, label_count.values) if count >= min_cells]
        keepcellids = [cellid for cellid in adata.obs.index if adata.obs[variable][cellid] in keeplabels]
        adata = adata[keepcellids]
        
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        if use_hvg:
            sc.pp.highly_variable_genes(
                adata, min_mean=0.0125, max_mean=3, min_disp=min_disp, max_disp=np.inf)
            var_features = adata.var.highly_variable[adata.var.highly_variable].index.tolist(
            )
            adata = adata[:, var_features]
            log.info('There are ' + str(len(var_features)) +
                     ' variable features')        
        
        sc.tl.rank_genes_groups(
            adata, variable, method='wilcoxon', corr_method='bonferroni')
        groups = adata.uns['rank_genes_groups']['names'].dtype.names
        diff_dict = {group: _format_df(sc.get.rank_genes_groups_df(
            adata, group=group), group, adjpval_thr, log2fc_thr) for group in groups}
        if contrast not in scplus_obj.uns.keys():
            scplus_obj.uns[contrast] = {}
        scplus_obj.uns[contrast][variable] = diff_dict
        log.info('Finished calculating ' + contrast +
                 ' for variable ' + variable)
