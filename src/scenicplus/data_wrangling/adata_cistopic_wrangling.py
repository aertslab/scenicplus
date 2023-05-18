from mudata import AnnData, MuData
from pycisTopic.cistopic_class import CistopicObject
from pycisTopic.diff_features import impute_accessibility
import pandas as pd
import logging
import sys
from typing import (
    Mapping, Any, Callable)

def process_multiome_data(
        GEX_anndata: AnnData,
        cisTopic_obj: CistopicObject,
        use_raw_for_GEX_anndata: bool = True,
        imputed_acc_kwargs: Mapping[str, Any] = {'scale_factor': 10**6},
        bc_transform_func: Callable = lambda x: x) -> MuData:
    # Create logger
    level = logging.INFO
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger('Ingesting multiome data')
   
    GEX_anndata = GEX_anndata.raw.to_adata() if use_raw_for_GEX_anndata else GEX_anndata

    # Use bc_transform_func to map scRNA-seq barcodes to scATAC-seq barcodes
    GEX_cell_names = [
        bc_transform_func(bc) for bc in GEX_anndata.obs_names]
    GEX_anndata.obs_names = GEX_cell_names
    ACC_cell_names = list(cisTopic_obj.cell_names.copy())
    
    # get cells with high quality (HQ cells) chromatin accessbility AND gene expression profile
    common_cells = list(set(GEX_cell_names) & set(ACC_cell_names))
    if len(common_cells) == 0:
        raise Exception(
            "No cells found which are present in both assays, check input and consider using `bc_transform_func`!")
    log.info(f"Found {len(common_cells)} multiome cells.")

    # Subset and impute accessibility
    imputed_acc_obj = impute_accessibility(
        cisTopic_obj, selected_cells=common_cells, **imputed_acc_kwargs)
    # Subset and get gene expression data
    X_EXP = GEX_anndata[common_cells].X.copy()
    
    assert all([
        rna_bc == atac_bc 
        for rna_bc, atac_bc in zip(
        GEX_anndata[common_cells].to_df().index, imputed_acc_obj.cell_names)]), \
            "rna and atac bc don't match"

    # Get and subset metadata from anndata and cisTopic object
    GEX_gene_metadata = GEX_anndata.var.copy(deep=True)
    GEX_cell_metadata = GEX_anndata.obs.copy(deep=True).loc[
        common_cells]
    ACC_region_metadata = cisTopic_obj.region_data.copy(deep=True).loc[
        imputed_acc_obj.feature_names]
    ACC_cell_metadata = cisTopic_obj.cell_data.copy(deep=True).loc[
        common_cells]

    # get cell dimensionality reductions
    GEX_dr_cell = {
        key: pd.DataFrame(
            index = GEX_cell_names,
            data=GEX_anndata.obsm[key]).loc[common_cells].to_numpy()
        for key in GEX_anndata.obsm.keys()}
    ACC_dr_cell = {
        key: cisTopic_obj.projections['cell'][key].loc[common_cells].to_numpy()
        for key in cisTopic_obj.projections['cell'].keys()}

    mudata = MuData(
        {
            'scRNA': AnnData(
                X=X_EXP, obs=GEX_cell_metadata,
                var=GEX_gene_metadata, obsm=GEX_dr_cell),
            'scATAC': AnnData(
                X=imputed_acc_obj.mtx.T, obs=ACC_cell_metadata,
                var=ACC_region_metadata, obsm=ACC_dr_cell)
        }
    )

    return mudata
