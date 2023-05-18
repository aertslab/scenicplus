import pathlib
from typing import (
    Callable, Union, Dict, List)
import pickle
import mudata
import logging
import sys

from scenicplus.data_wrangling.adata_cistopic_wrangling import (
    process_multiome_data, process_non_multiome_data)
from scenicplus.data_wrangling.cistarget_wrangling import get_and_merge_cistromes

# Create logger
level = logging.INFO
format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
handlers = [logging.StreamHandler(stream=sys.stdout)]
logging.basicConfig(level=level, format=format, handlers=handlers)
log = logging.getLogger('SCENIC+')

def prepare_GEX_ACC(
        cisTopic_obj_fname: pathlib.Path,
        GEX_anndata_fname: pathlib.Path,
        out_file: pathlib.Path,
        use_raw_for_GEX_anndata: bool,
        is_multiome: bool,
        bc_transform_func: Union[None, Callable],
        key_to_group_by: Union[None, str],
        nr_metacells: Union[int, Dict[str, int], None],
        nr_cells_per_metacells: Union[int, Dict[str, int]]) -> None:
    log.info("Reading cisTopic object.")
    cisTopic_obj = pickle.load(open(cisTopic_obj_fname, "rb"))
    log.info("Reading gene expression AnnData.")
    GEX_anndata = mudata.read(GEX_anndata_fname.__str__())
    if is_multiome:
        mdata = process_multiome_data(
            GEX_anndata=GEX_anndata,
            cisTopic_obj=cisTopic_obj,
            use_raw_for_GEX_anndata=use_raw_for_GEX_anndata,
            bc_transform_func=bc_transform_func)
    else:
        mdata = process_non_multiome_data(
            GEX_anndata=GEX_anndata,
            cisTopic_obj=cisTopic_obj,
            key_to_group_by=key_to_group_by,
            use_raw_for_GEX_anndata=use_raw_for_GEX_anndata,
            nr_metacells=nr_metacells,
            nr_cells_per_metacells=nr_cells_per_metacells)
    mdata.write_h5mu(out_file)

def prepare_motif_enrichment_results(
        menr_fname: pathlib.Path,
        multiome_mudata_fname: pathlib.Path,
        out_file_direct_annotation: pathlib.Path,
        out_file_extended_annotation: pathlib.Path,
        direct_annotation: List[str],
        extended_annotation: List[str]) -> None:
    log.info("Reading motif enrichment results.")
    menr = pickle.load(open(menr_fname, 'rb'))
    log.info("Reading multiome MuData.")
    mdata = mudata.read(multiome_mudata_fname.__str__())
    log.info("Getting cistromes.")
    adata_direct_cistromes, adata_extended_cistromes = get_and_merge_cistromes(
        menr=menr,
        scplus_regions=set(mdata['scATAC'].var_names),
        direct_annotation=direct_annotation,
        extended_annotation=extended_annotation)
    # Get transcription factor names from cistromes
    # Later, to calculate TF-to-gene relationships these TFs will be used.
    TFs = list(
        set([*adata_direct_cistromes.var_names, *adata_extended_cistromes.var_names]) & \
        set(mdata['scRNA'].var_names))
    log.info(f"Labling {len(TFs)} genes as transcription factors.")
    mdata['scRNA'].var['is_TF'] = False
    mdata['scRNA'].var.loc[TFs, 'is_TF'] = True
    log.info(
        f"Saving modified multiome MuData to: {multiome_mudata_fname.__str__()}")
    mdata.write_h5mu(multiome_mudata_fname)
    if len(direct_annotation) > 0:
        log.info(
            f"Writing direct cistromes to: {out_file_direct_annotation.__str__()}")
        adata_direct_cistromes.write_h5ad(out_file_direct_annotation.__str__())
    if len(extended_annotation) > 0:
        log.info(
            f"Writing extended cistromes to: {out_file_extended_annotation.__str__()}")
        adata_extended_cistromes.write_h5ad(out_file_extended_annotation.__str__())