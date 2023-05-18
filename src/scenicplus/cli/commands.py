import pathlib
from typing import (
    Callable, Union, Dict, List,
    Tuple)
import pickle
import mudata
import logging
import sys
import pandas as pd

from scenicplus.data_wrangling.adata_cistopic_wrangling import (
    process_multiome_data, process_non_multiome_data)
from scenicplus.data_wrangling.cistarget_wrangling import get_and_merge_cistromes
from scenicplus.data_wrangling.gene_search_space import (
    download_gene_annotation_and_chromsizes, get_search_space)

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
        out_file_tf_names: pathlib.Path,
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
    log.info(f"Found {len(TFs)} TFs.")
    log.info(f"Saving TF names to: {out_file_tf_names.__str__()}")
    with open(out_file_tf_names, "w") as f:
        for TF in TFs:
            _ = f.write(TF)
            _ = f.write("\n")
    if len(direct_annotation) > 0:
        log.info(
            f"Writing direct cistromes to: {out_file_direct_annotation.__str__()}")
        adata_direct_cistromes.write_h5ad(out_file_direct_annotation.__str__())
    if len(extended_annotation) > 0:
        log.info(
            f"Writing extended cistromes to: {out_file_extended_annotation.__str__()}")
        adata_extended_cistromes.write_h5ad(out_file_extended_annotation.__str__())

def download_gene_annotation_chromsizes(
        species: str,
        biomart_host: str,
        use_ucsc_chromosome_style: bool,
        genome_annotation_out_fname: pathlib.Path,
        chromsizes_out_fname: pathlib.Path):
    result = download_gene_annotation_and_chromsizes(
            species=species,
            biomart_host=biomart_host,
            use_ucsc_chromosome_style=use_ucsc_chromosome_style)
    if type(result) is tuple:
        annot, chromsizes = result 
        log.info(f"Saving chromosome sizes to: {chromsizes_out_fname.__str__()}")
        chromsizes.to_csv(
            chromsizes_out_fname,
            sep = "\t", header = True, index = False)
    else:
        annot: pd.DataFrame = result
        log.info(
            "Chrosomome sizes was not found, please provide this information manually.")
    log.info(f"Saving genome annotation to: {genome_annotation_out_fname.__str__()}")
    annot.to_csv(
        genome_annotation_out_fname,
        sep = "\t", header = True, index = False)

def get_search_space_command(
        multiome_mudata_fname: pathlib.Path,
        gene_annotation_fname: pathlib.Path,
        chromsizes_fname: pathlib.Path,
        out_fname: pathlib.Path,
        use_gene_boundaries: bool,
        upstream: Tuple[int, int],
        downstream: Tuple[int, int],
        extend_tss: Tuple[int, int],
        remove_promoters: bool):
    log.info("Reading data")
    mdata = mudata.read(multiome_mudata_fname.__str__())
    gene_annotation=pd.read_table(gene_annotation_fname)
    chromsizes=pd.read_table(chromsizes_fname)
    search_space = get_search_space(
        scplus_region=set(mdata['scATAC'].var_names),
        scplus_genes=set(mdata['scRNA'].var_names),
        gene_annotation=gene_annotation,
        chromsizes=chromsizes,
        use_gene_boundaries=use_gene_boundaries,
        upstream=upstream,
        downstream=downstream,
        extend_tss=extend_tss,
        remove_promoters=remove_promoters)
    log.info(f"Writing search space to: {out_fname.__str__()}")
    search_space.to_csv(
        out_fname, sep = "\t",
        header = True, index = False)
