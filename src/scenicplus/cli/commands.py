# TODO: split these commands over multiple files. 
#       this will limit the amount of import time required.
#       Or, put the imports inside the function calls.

# General imports
import pathlib
from typing import (
    Callable, Union, Dict, List,
    Tuple, Literal)
import pickle
import mudata
import logging
import sys
import pandas as pd

# SCENIC+ imports
from scenicplus.data_wrangling.adata_cistopic_wrangling import (
    process_multiome_data, 
    process_non_multiome_data)
from scenicplus.data_wrangling.cistarget_wrangling import get_and_merge_cistromes
from scenicplus.data_wrangling.gene_search_space import (
    download_gene_annotation_and_chromsizes, 
    get_search_space)
from scenicplus.TF_to_gene import calculate_TFs_to_genes_relationships
from scenicplus.enhancer_to_gene import calculate_regions_to_genes_relationships
from scenicplus.grn_builder.gsea_approach import build_grn
from scenicplus.grn_builder.modules import eRegulon


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

def infer_TF_to_gene(
        multiome_mudata_fname: pathlib.Path,
        tf_names_fname: pathlib.Path,
        temp_dir: pathlib.Path,
        adj_out_fname: pathlib.Path,
        method: Literal['GBM', 'RF'],
        n_cpu: int,
        seed: int):
    log.info("Reading multiome MuData.")
    mdata = mudata.read(multiome_mudata_fname.__str__())
    with open(tf_names_fname, "r") as f:
         tf_names = f.read().split('\n')
    log.info(f"Using {len(tf_names)} TFs.")
    adj = calculate_TFs_to_genes_relationships(
        df_exp_mtx=mdata["scRNA"].to_df(),
        tf_names = tf_names,
        temp_dir = temp_dir,
        method = method,
        n_cpu = n_cpu,
        seed = seed)
    log.info(f"Saving TF to gene adjacencies to: {adj_out_fname.__str__()}")
    adj.to_csv(
        adj_out_fname,
        sep='\t', header = True, index = False)

def infer_region_to_gene(
        multiome_mudata_fname: pathlib.Path,
        search_space_fname: pathlib.Path,
        temp_dir: pathlib.Path,
        adj_out_fname: pathlib.Path,
        importance_scoring_method: Literal['RF', 'ET', 'GBM'],
        correlation_scoring_method: Literal['PR', 'SR'],
        mask_expr_dropout: bool,
        n_cpu: int):
    log.info("Reading multiome MuData.")
    mdata = mudata.read(multiome_mudata_fname.__str__())
    log.info("Reading search space")
    search_space = pd.read_table(search_space_fname)
    adj = calculate_regions_to_genes_relationships(
        df_exp_mtx = mdata['scRNA'].to_df(),
        df_acc_mtx = mdata['scATAC'].to_df(),
        search_space = search_space,
        temp_dir = temp_dir,
        mask_expr_dropout = mask_expr_dropout,
        importance_scoring_method = importance_scoring_method,
        correlation_scoring_method = correlation_scoring_method,
        n_cpu = n_cpu)
    log.info(f"Saving region to gene adjacencies to {adj_out_fname.__str__()}")
    adj.to_csv(
        adj_out_fname,
        sep='\t', header = True, index = False)

def _format_egrns(
        eRegulons: List[eRegulon],
        tf_to_gene: pd.DataFrame):
    """
    A function to format eRegulons to a pandas dataframe
    """
    REGION_TO_GENE_COLUMNS = [
        'Region',
        'Gene',
        'importance',
        'rho',
        'importance_x_rho',
        'importance_x_abs_rho']
    eRegulons_formatted = []
    for ereg in eRegulons:
        TF = ereg.transcription_factor
        is_extended = ereg.is_extended
        region_to_gene = pd.DataFrame(
            ereg.regions2genes,
            columns=REGION_TO_GENE_COLUMNS)
        n_target_regions = len(set(region_to_gene['Region']))
        n_target_genes = len(set(region_to_gene['Gene']))
        # TF_[extended,direct]_[+,-]/[+,-]
        eRegulon_name = TF + '_' + \
            ('extended' if is_extended else 'direct') + '_' + \
            ('+' if 'positive tf2g' in ereg.context else '-') + '/' + \
            ('+' if 'positive r2g' in ereg.context else '-')
        # TF_[extended,direct]_[+,-]/[+,-]_(nr)
        region_signature_name = eRegulon_name + '_' + f'({n_target_regions}r)'
        # TF_[extended,direct]_[+,-]/[+,-]_(ng)
        gene_signature_name = eRegulon_name + '_' + f'({n_target_genes}g)'
        # construct dataframe
        region_to_gene['TF'] = TF
        region_to_gene['is_extended'] = is_extended
        region_to_gene['eRegulon_name'] = eRegulon_name
        region_to_gene['Gene_signature_name'] = gene_signature_name
        region_to_gene['Region_signature_name'] = region_signature_name
        eRegulons_formatted.append(region_to_gene)
    eRegulon_metadata = pd.concat(eRegulons_formatted)
    eRegulon_metadata.merge(
        right=tf_to_gene.rename({'target': 'Gene'}, axis = 1), #TODO: rename col beforehand!
        how='left',
        on= ['TF', 'Gene'],
        suffixes=['_R2G', '_TF2G'])
    return eRegulon_metadata

def infer_grn(
        TF_to_gene_adj_fname: pathlib.Path,
        region_to_gene_adj_fname: pathlib.Path,
        cistromes_fname: pathlib.Path,
        eRegulon_out_fname: pathlib.Path,
        is_extended: bool,
        temp_dir: pathlib.Path,
        order_regions_to_genes_by: str,
        order_TFs_to_genes_by: str,
        gsea_n_perm: int,
        quantiles: List[float],
        top_n_regionTogenes_per_gene: List[float],
        top_n_regionTogenes_per_region: List[float],
        binarize_using_basc: bool,
        min_regions_per_gene: int,
        rho_dichotomize_tf2g: bool,
        rho_dichotomize_r2g: bool,
        rho_dichotomize_eregulon: bool,
        keep_only_activating: bool,
        rho_threshold: float,
        min_target_genes: int,
        n_cpu: int):
    log.info("Loading TF to gene adjacencies.")
    tf_to_gene = pd.read_table(TF_to_gene_adj_fname)

    log.info("Loading region to gene adjacencies.")
    region_to_gene = pd.read_table(region_to_gene_adj_fname)

    log.info("Loading cistromes.")
    cistromes = mudata.read(cistromes_fname.__str__())

    eRegulons = build_grn(
        tf_to_gene=tf_to_gene,
        region_to_gene=region_to_gene,
        cistromes=cistromes,
        is_extended=is_extended,
        temp_dir=temp_dir.__str__(),
        order_regions_to_genes_by=order_regions_to_genes_by,
        order_TFs_to_genes_by=order_TFs_to_genes_by,
        gsea_n_perm=gsea_n_perm,
        quantiles=quantiles,
        top_n_regionTogenes_per_gene=top_n_regionTogenes_per_gene,
        top_n_regionTogenes_per_region=top_n_regionTogenes_per_region,
        binarize_using_basc=binarize_using_basc,
        min_regions_per_gene=min_regions_per_gene,
        rho_dichotomize_tf2g=rho_dichotomize_tf2g,
        rho_dichotomize_r2g=rho_dichotomize_r2g,
        rho_dichotomize_eregulon=rho_dichotomize_eregulon,
        keep_only_activating=keep_only_activating,
        rho_threshold=rho_threshold,
        NES_thr=0,
        adj_pval_thr=1,
        min_target_genes=min_target_genes,
        n_cpu=n_cpu,
        merge_eRegulons=True,
        disable_tqdm=False)

    log.info("Formatting eGRN as table.")
    eRegulon_metadata = _format_egrns(
        eRegulons=eRegulons,
        tf_to_gene=tf_to_gene)
    log.info(f"Saving network to {eRegulon_out_fname.__str__()}")
    eRegulon_metadata.to_csv(
        eRegulon_out_fname,
        sep='\t', header=True, index=False)

# TODO: add command for triplet score
