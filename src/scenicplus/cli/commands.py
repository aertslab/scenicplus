# General imports
import logging
import os
import pathlib
import pickle
import shutil
import sys
from typing import Callable, Dict, Iterator, List, Literal, Optional, Tuple, Union

import joblib
import mudata
import pandas as pd
import pyranges as pr
from importlib_resources import files
from pycistarget.motif_enrichment_cistarget import cisTarget
from pycistarget.motif_enrichment_dem import DEM

from scenicplus.grn_builder.modules import eRegulon

# Create logger
level = logging.INFO
format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
handlers = [logging.StreamHandler(stream=sys.stdout)]
logging.basicConfig(level=level, format=format, handlers=handlers)
log = logging.getLogger("SCENIC+")

def init_snakemake_folder(
        out_dir: pathlib.Path):
    """
    Function to create a snakemake folder structure in the output directory.

    Parameters
    ----------
    out_dir : pathlib.Path
        The output directory where the snakemake folder structure will be created.

    """
    log.info(f"Creating snakemake folder in: {out_dir.__str__()}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    os.makedirs(os.path.join(out_dir, "Snakemake"))
    os.makedirs(os.path.join(out_dir, "Snakemake", "config"))
    os.makedirs(os.path.join(out_dir, "Snakemake", "workflow"))
    config_file = str(files("scenicplus.snakemake").joinpath("config.yaml"))
    snakefile = str(files("scenicplus.snakemake").joinpath("Snakefile"))
    shutil.copy(config_file, os.path.join(out_dir, "Snakemake", "config", "config.yaml"))
    shutil.copy(snakefile, os.path.join(out_dir, "Snakemake", "workflow", "Snakefile"))

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
    """
    Prepare the gene expression and accessibility data for SCENIC+ analysis.

    Parameters
    ----------
    cisTopic_obj_fname : pathlib.Path
        The path to the cisTopic object file.
    GEX_anndata_fname : pathlib.Path
        The path to the gene expression AnnData file.
    out_file : pathlib.Path
        The path to the output file.
    use_raw_for_GEX_anndata : bool
        Whether to use the raw data for the gene expression AnnData.
    is_multiome : bool
        Whether the data is multiome or not.
    bc_transform_func : Union[None, Callable]
        The function to transform the barcodes from scATAC to scRNA.
    key_to_group_by : Union[None, str]
        The key to group the data by.
    nr_metacells : Union[int, Dict[str, int], None]
        The number of metacells.
    nr_cells_per_metacells : Union[int, Dict[str, int]]
        The number of cells per metacell.

    """
    from scenicplus.data_wrangling.adata_cistopic_wrangling import (
        process_multiome_data,
        process_non_multiome_data,
    )
    log.info("Reading cisTopic object.")
    with open(cisTopic_obj_fname, "rb") as f:
        cisTopic_obj = pickle.load(f)
    log.info("Reading gene expression AnnData.")
    GEX_anndata = mudata.read(GEX_anndata_fname.__str__())
    if is_multiome:
        if bc_transform_func is None:
            bc_transform_func = lambda x: x  # noqa: E731
        mdata = process_multiome_data(
            GEX_anndata=GEX_anndata,
            cisTopic_obj=cisTopic_obj,
            use_raw_for_GEX_anndata=use_raw_for_GEX_anndata,
            bc_transform_func=bc_transform_func)
    else:
        if key_to_group_by is None:
            raise ValueError(
                "key_to_group_by must be provided for non-multiome data."
            )
        mdata = process_non_multiome_data(
            GEX_anndata=GEX_anndata,
            cisTopic_obj=cisTopic_obj,
            key_to_group_by=key_to_group_by,
            use_raw_for_GEX_anndata=use_raw_for_GEX_anndata,
            nr_metacells=nr_metacells,
            nr_cells_per_metacells=nr_cells_per_metacells)
    mdata.write_h5mu(out_file)

def _run_cistarget_single_region_set(
        cistarget_db_fname,
        region_set,
        fraction_overlap_w_cistarget_database,
        name,
        species,
        auc_threshold,
        nes_threshold,
        rank_threshold,
        path_to_motif_annotations,
        annotation_version,
        annotations_to_use,
        motif_similarity_fdr,
        orthologous_identity_threshold) -> cisTarget:
    """Helper function to run cisTarget on a single region set."""
    from pycistarget.motif_enrichment_cistarget import cisTargetDatabase
    ctx_db = cisTargetDatabase(
        fname=cistarget_db_fname,
        region_sets=region_set,
        name="cistarget",
        fraction_overlap=fraction_overlap_w_cistarget_database)
    cistarget_result = cisTarget(
        region_set=region_set,
        name=name,
        species=species,
        auc_threshold=auc_threshold,
        nes_threshold=nes_threshold,
        rank_threshold=rank_threshold,
        path_to_motif_annotations=path_to_motif_annotations,
        annotation_version=annotation_version,
        annotation_to_use=annotations_to_use,
        motif_similarity_fdr=motif_similarity_fdr,
        orthologous_identity_threshold=orthologous_identity_threshold,
    )
    cistarget_result.run_ctx(ctx_db)
    return cistarget_result

def run_motif_enrichment_cistarget(
        region_set_folder: pathlib.Path,
        cistarget_db_fname: pathlib.Path,
        output_fname_cistarget_result: pathlib.Path,
        n_cpu: int,
        fraction_overlap_w_cistarget_database: float,
        auc_threshold: float,
        nes_threshold: float,
        rank_threshold: float,
        path_to_motif_annotations: str,
        annotation_version: str,
        motif_similarity_fdr: float,
        orthologous_identity_threshold: float,
        temp_dir: pathlib.Path,
        species: Literal[
            "homo_sapiens", "mus_musculus", "drosophila_melanogaster"],
        annotations_to_use: List[str],
        write_html: bool = True,
        output_fname_cistarget_html: Optional[pathlib.Path] = None) -> None:
    """
    Run motif enrichment using cistarget algorithm.

    Parameters
    ----------
    region_set_folder: pathlib.Path
        Path to directory containing region sets.
    cistarget_db_fname: pathlib.Path
        Path to ranking database.
    output_fname_cistarget_result: pathlib.Path
        Path to store output.
    n_cpu: int
        Number of parallel processes to run
    fraction_overlap_w_cistarget_database: float
        Fraction of bps that should overlap with the regions used for
        creating the cistarget database
    auc_threshold: float
        AUC threshold for calling significant motif enrichment.
    nes_threshold: float
        NES threshold for calling significant motif enrichment.
    rank_threshold: float
        Fraction of the ranking used for calculating AUC values.
    path_to_motif_annotations: str
        Path to motif annotations.
    annotation_version: str
        Version of the motif annotations.
    motif_similarity_fdr: float
        FDR threshold for motif similarity.
    orthologous_identity_threshold: float
        Threshold for orthologous identity.
    temp_dir: pathlib.Path
        Path to temporary directory.
    species: Literal[
        "homo_sapiens", "mus_musculus", "drosophila_melanogaster"]
        species name.
    annotations_to_use: List[str]
        List of annotations to use
    write_html: bool = True
        Wether to write a html output file or not
    output_fname_cistarget_html: Optional[pathlib.Path]
        Where to store the html output file.

    """
    region_set_dict: Dict[str, pr.PyRanges] = {}
    log.info(f"Reading region sets from: {region_set_folder}")
    for region_set_subfolder in os.listdir(region_set_folder):
        if os.path.isdir(os.path.join(region_set_folder, region_set_subfolder)):
            log.info(f"Reading all .bed files in: {region_set_subfolder}")
            if any(
                f.endswith(".bed")
                    for f in
                    os.listdir(
                        os.path.join(
                            region_set_folder, region_set_subfolder
                        )
                    )
                ):
                for f in os.listdir(os.path.join(region_set_folder, region_set_subfolder)):
                    if f.endswith(".bed"):
                        key_name = region_set_subfolder + "_" + f.replace(".bed", "")
                        if key_name in region_set_dict:
                            raise ValueError(
                                f"non unique folder/file combination: {key_name}"
                            )
                        region_set_dict[key_name] = pr.read_bed(
                                os.path.join(
                                        region_set_folder, region_set_subfolder, f
                                ),
                                as_df=False
                        )

    cistarget_results: List[cisTarget] = joblib.Parallel(
        n_jobs=n_cpu,
        temp_folder=temp_dir
    )(
        joblib.delayed(
            _run_cistarget_single_region_set
        )(
            name = key,
            region_set=region_set_dict[key],
            cistarget_db_fname=cistarget_db_fname,
            fraction_overlap_w_cistarget_database=fraction_overlap_w_cistarget_database,
            species=species,
            auc_threshold=auc_threshold,
            nes_threshold=nes_threshold,
            rank_threshold=rank_threshold,
            path_to_motif_annotations=path_to_motif_annotations,
            annotation_version=annotation_version,
            annotations_to_use=annotations_to_use,
            motif_similarity_fdr=motif_similarity_fdr,
            orthologous_identity_threshold=orthologous_identity_threshold
        )
        for key in region_set_dict
    )
    # Write results to file
    if write_html:
        log.info(f"Writing html to: {output_fname_cistarget_html}")
        all_motif_enrichment_df = pd.concat(
            ctx_result.motif_enrichment for ctx_result in cistarget_results
        )
        all_motif_enrichment_df.to_html(
            buf = output_fname_cistarget_html,
            escape = False,
            col_space = 80
        )
    log.info(f"Writing output to: {output_fname_cistarget_result}")
    for cistarget_result in cistarget_results:
        if len(cistarget_result.motif_enrichment) > 0:
            cistarget_result.write_hdf5(
                path = output_fname_cistarget_result,
                mode = "a"
            )

def _run_dem_single_region_set(
        foreground_region_sets,
        background_region_sets,
        dem_db_fname,
        max_bg_regions,
        genome_annotation,
        balance_number_of_promoters,
        promoter_space,
        seed,
        fraction_overlap_w_dem_database,
        name,
        species,
        adjpval_thr,
        log2fc_thr,
        mean_fg_thr,
        motif_hit_thr,
        path_to_motif_annotations,
        annotation_version,
        annotations_to_use,
        motif_similarity_fdr,
        orthologous_identity_threshold) -> DEM:
    """Helper function to run DEM on a single region set."""
    from pycistarget.motif_enrichment_dem import (
        DEMDatabase,
        get_foreground_and_background_regions,
    )
    # Get foreground and background regions for DEM analysis
    foreground_regions, background_regions = get_foreground_and_background_regions(
        foreground_region_sets = foreground_region_sets,
        background_region_sets = background_region_sets,
        max_bg_regions = max_bg_regions,
        genome_annotation = genome_annotation,
        balance_number_of_promoters = balance_number_of_promoters,
        promoter_space = promoter_space,
        seed = seed)
    # Load DEM database
    dem_db = DEMDatabase(
        dem_db_fname,
        fraction_overlap=fraction_overlap_w_dem_database)
    # Setup DEM analysis
    dem_result = DEM(
        foreground_regions = foreground_regions,
        background_regions = background_regions,
        name = name,
        species = species,
        adjpval_thr = adjpval_thr,
        log2fc_thr = log2fc_thr,
        mean_fg_thr = mean_fg_thr,
        motif_hit_thr = motif_hit_thr,
        path_to_motif_annotations = path_to_motif_annotations,
        annotation_version = annotation_version,
        annotation_to_use = annotations_to_use,
        motif_similarity_fdr = motif_similarity_fdr,
        orthologous_identity_threshold = orthologous_identity_threshold)
    # Run DEM analysis
    dem_result.run(dem_db)
    return dem_result

def _get_foreground_background(
        region_set_dict: Dict[str, Dict[str, pr.PyRanges]]
    ) -> Iterator[Tuple[str, List[pr.PyRanges], List[pr.PyRanges]]]:
    """Helper function to get foreground and background regions for DEM analysis."""
    for key in region_set_dict:
        for subkey_fg in region_set_dict[key]:
            foreground = [region_set_dict[key][subkey_fg]]
            background = [
                region_set_dict[key][subkey_bg]
                for subkey_bg in region_set_dict[key]
                if subkey_bg != subkey_fg
            ]
            yield (key + "_" + subkey_fg + "_vs_all", foreground, background)

def run_motif_enrichment_dem(
        region_set_folder: pathlib.Path,
        dem_db_fname: pathlib.Path,
        output_fname_dem_html: pathlib.Path,
        output_fname_dem_result: pathlib.Path,
        n_cpu: int,
        temp_dir: pathlib.Path,
        species: Literal[
                "homo_sapiens", "mus_musculus", "drosophila_melanogaster"],
        fraction_overlap_w_dem_database: float = 0.4,
        max_bg_regions: Optional[int] = None,
        path_to_genome_annotation: Optional[str] = None,
        balance_number_of_promoters: bool = True,
        promoter_space: int = 1_000,
        adjpval_thr: float = 0.05,
        log2fc_thr: float = 1.0,
        mean_fg_thr: float = 0.0,
        motif_hit_thr: Optional[float] = None,
        path_to_motif_annotations: Optional[str] = None,
        annotation_version: str = "v10nr_clust",
        annotations_to_use: tuple = ("Direct_annot", "Orthology_annot"),
        motif_similarity_fdr: float = 0.001,
        orthologous_identity_threshold: float = 0.0,
        seed: int = 555,
        write_html: bool = True):
    """
    Run motif enrichment using DEM algorithm.

    region_set_folder: pathlib.Path
        Path to directory containing region set bed files.
    dem_db_fname: pathlib.Path
        Path to score database
    output_fname_dem_html: pathlib.Path
        Path to store html output.
    output_fname_dem_result: pathlib.Path
        Path to store output.
    n_cpu: int
        Number of parallel processes to run.
    temp_dir: pathlib.Path
        Path to temporary directory.
    species: Literal[
        "homo_sapiens", "mus_musculus", "drosophila_melanogaster"]
        species name.
    fraction_overlap_w_dem_database: float
        Fraction of bps that should overlap with region in DEM db.
    max_bg_regions: Optional[int]
        Maximum number of background regions to use.
    path_to_genome_annotation: Optional[str]
        Path to genome annotation file.
    balance_number_of_promoters: bool
        Wether to balance the number of promoter regions in foreground
        and background set.
    promoter_space: int
        Number of bps up- and downstream of the TSS to consider as promoter.
    adjpval_thr: float
        Adjusted p-values threshold.
    log2fc_thr: float
        Log2FC threshold.
    mean_fg_thr: float
        Threshold on the mean of the motif score on the foreground regions.
    motif_hit_thr: Optional[float]
        Threshold on the motif score to consider motif hits.
    path_to_motif_annotations: Optional[str]
        Path to the motif annotation file.
    annotation_version: str
        Annotation version to use.
    annotations_to_use: list
        List of annotations to use.
    motif_similarity_fdr: float
        FDR threshold for motif similarity.
    orthologous_identity_threshold:
        Threshold for orthologous identity.
    seed: int
        Seed for random number generator.
    write_html: bool
        Wether to write a html output file or not.

    """
    from pycistarget.motif_enrichment_dem import (
        DEM,
    )
    region_set_dict: Dict[str, pr.PyRanges] = {}
    log.info(f"Reading region sets from: {region_set_folder}")
    for region_set_subfolder in os.listdir(region_set_folder):
        if os.path.isdir(os.path.join(region_set_folder, region_set_subfolder)):
            region_set_dict[region_set_subfolder] = {}
            log.info(f"Reading all .bed files in: {region_set_subfolder}")
            if any(
                f.endswith(".bed")
                    for f in
                    os.listdir(
                        os.path.join(
                            region_set_folder, region_set_subfolder
                        )
                    )
                ):
                for f in os.listdir(os.path.join(region_set_folder, region_set_subfolder)):
                    if f.endswith(".bed"):
                        region_set_dict[region_set_subfolder][f.replace(".bed", "")] = pr.read_bed(
                                os.path.join(
                                        region_set_folder, region_set_subfolder, f
                                ),
                                as_df=False
                        )
    # Read genome annotation, if needed
    if path_to_genome_annotation is not None:
        genome_annotation = pd.read_table(path_to_genome_annotation)
    else:
        genome_annotation = None

    dem_results: List[DEM] = joblib.Parallel(
        n_jobs=n_cpu,
        temp_folder=temp_dir
    )(
        joblib.delayed(
            _run_dem_single_region_set
        )(
            foreground_region_sets=foreground_region_sets,
            background_region_sets=background_region_sets,
            name=name,
            dem_db_fname=dem_db_fname,
            max_bg_regions=max_bg_regions,
            genome_annotation=genome_annotation,
            balance_number_of_promoters=balance_number_of_promoters,
            promoter_space=promoter_space,
            seed=seed,
            fraction_overlap_w_dem_database=fraction_overlap_w_dem_database,
            species=species,
            adjpval_thr=adjpval_thr,
            log2fc_thr=log2fc_thr,
            mean_fg_thr=mean_fg_thr,
            motif_hit_thr=motif_hit_thr,
            path_to_motif_annotations=path_to_motif_annotations,
            annotation_version=annotation_version,
            annotations_to_use=annotations_to_use,
            motif_similarity_fdr=motif_similarity_fdr,
            orthologous_identity_threshold=orthologous_identity_threshold
        )
        for name, foreground_region_sets, background_region_sets in _get_foreground_background(region_set_dict)
    )
    if write_html:
        log.info(f"Writing html to: {output_fname_dem_html}")
        all_motif_enrichment_df = pd.concat(
            ctx_result.motif_enrichment for ctx_result in dem_results
        )
        all_motif_enrichment_df.to_html(
            buf = output_fname_dem_html,
            escape = False,
            col_space = 80
        )
    log.info(f"Writing output to: {output_fname_dem_result}")
    for dem_result in dem_results:
        if len(dem_result.motif_enrichment) > 0:
            dem_result.write_hdf5(
                path = output_fname_dem_result,
                mode = "a"
            )

def prepare_motif_enrichment_results(
        paths_to_motif_enrichment_results: List[str],
        multiome_mudata_fname: pathlib.Path,
        out_file_direct_annotation: pathlib.Path,
        out_file_extended_annotation: pathlib.Path,
        out_file_tf_names: pathlib.Path,
        direct_annotation: List[str],
        extended_annotation: List[str]) -> None:
    """
    Prepare motif enrichment results for SCENIC+ analysis.

    Parameters
    ----------
    paths_to_motif_enrichment_results : List[str]
        List of paths to motif enrichment results.
    multiome_mudata_fname : pathlib.Path
        Path to multiome MuData file.
    out_file_direct_annotation : pathlib.Path
        Path to store TF cistromes with direct annotation.
    out_file_extended_annotation : pathlib.Path
        Path to store TF cistromes with extended annotation.
    out_file_tf_names : pathlib.Path
        Path to store TF names.
    direct_annotation : List[str]
        List of annotations to use for direct annotation.
    extended_annotation : List[str]
        List of annotations to use for extended annotation.

    """
    from scenicplus.data_wrangling.cistarget_wrangling import get_and_merge_cistromes
    log.info("Reading multiome MuData.")
    mdata = mudata.read(multiome_mudata_fname.__str__())
    log.info("Getting cistromes.")
    adata_direct_cistromes, adata_extended_cistromes = get_and_merge_cistromes(
        paths_to_motif_enrichment_results=paths_to_motif_enrichment_results,
        scplus_regions=set(mdata["scATAC"].var_names),
        direct_annotation=direct_annotation,
        extended_annotation=extended_annotation)
    # Get transcription factor names from cistromes
    # Later, to calculate TF-to-gene relationships these TFs will be used.
    TFs = list(
        {
            *adata_direct_cistromes.var_names,
            *adata_extended_cistromes.var_names
        } & set(mdata["scRNA"].var_names))
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
    """
    Download gene annotation and chromosome sizes.

    Parameters
    ----------
    species : str
        Species name.
    biomart_host : str
        Host of biomart.
    use_ucsc_chromosome_style : bool
        Whether to use UCSC chromosome style.
    genome_annotation_out_fname : pathlib.Path
        Path to store genome annotation.
    chromsizes_out_fname : pathlib.Path
        Path to store chromosome sizes.

    """
    from scenicplus.data_wrangling.gene_search_space import (
        download_gene_annotation_and_chromsizes,
    )
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
        annot = result
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
    """
    Format the search space for SCENIC+ analysis.

    Parameters
    ----------
    multiome_mudata_fname : pathlib.Path
        Path to multiome MuData file.
    gene_annotation_fname : pathlib.Path
        Path to gene annotation file.
    chromsizes_fname : pathlib.Path
        Path to chromosome sizes file.
    out_fname : pathlib.Path
        Path to store output.
    use_gene_boundaries : bool
        Whether to use gene boundaries.
    upstream : Tuple[int, int]
        Minimum and maximum (up until another gene) number of bps upstream of
        TSS to include in the search space.
    downstream : Tuple[int, int]
        Minimum and maximum (up until another gene) number of bps downstream of
        gene's end to include in the search space.
    extend_tss : Tuple[int, int]
        Number of bps to extend the TSS to define promoter region.
    remove_promoters : bool
        Whether to remove promoters.

    """
    from scenicplus.data_wrangling.gene_search_space import get_search_space
    log.info("Reading data")
    mdata = mudata.read(multiome_mudata_fname.__str__())
    gene_annotation=pd.read_table(gene_annotation_fname)
    chromsizes=pd.read_table(chromsizes_fname)
    search_space = get_search_space(
        scplus_region=set(mdata["scATAC"].var_names),
        scplus_genes=set(mdata["scRNA"].var_names),
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
        method: Literal["GBM", "RF"],
        n_cpu: int,
        seed: int):
    from scenicplus.TF_to_gene import calculate_TFs_to_genes_relationships
    log.info("Reading multiome MuData.")
    mdata = mudata.read(multiome_mudata_fname.__str__())
    with open(tf_names_fname) as f:
         tf_names = f.read().split("\n")
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
        sep="\t", header = True, index = False)

def infer_region_to_gene(
        multiome_mudata_fname: pathlib.Path,
        search_space_fname: pathlib.Path,
        temp_dir: pathlib.Path,
        adj_out_fname: pathlib.Path,
        importance_scoring_method: Literal["RF", "ET", "GBM"],
        correlation_scoring_method: Literal["PR", "SR"],
        mask_expr_dropout: bool,
        n_cpu: int):
    """
    Infer region to gene relationships.

    Parameters
    ----------
    multiome_mudata_fname : pathlib.Path
        Path to multiome MuData file.
    search_space_fname : pathlib.Path
        Path to search space file.
    temp_dir : pathlib.Path
        Path to temporary directory.
    adj_out_fname : pathlib.Path
        Path to store output.
    importance_scoring_method : Literal["RF", "ET", "GBM"]
        Method to score importance.
    correlation_scoring_method : Literal["PR", "SR"]
        Method to score correlation.
    mask_expr_dropout : bool
        Whether to mask expression dropout.
    n_cpu : int
        Number of parallel processes to run.

    """
    from scenicplus.enhancer_to_gene import calculate_regions_to_genes_relationships
    log.info("Reading multiome MuData.")
    mdata = mudata.read(multiome_mudata_fname.__str__())
    log.info("Reading search space")
    search_space = pd.read_table(search_space_fname)
    adj = calculate_regions_to_genes_relationships(
        df_exp_mtx = mdata["scRNA"].to_df(),
        df_acc_mtx = mdata["scATAC"].to_df(),
        search_space = search_space,
        temp_dir = temp_dir,
        mask_expr_dropout = mask_expr_dropout,
        importance_scoring_method = importance_scoring_method,
        correlation_scoring_method = correlation_scoring_method,
        n_cpu = n_cpu)
    log.info(f"Saving region to gene adjacencies to {adj_out_fname.__str__()}")
    adj.to_csv(
        adj_out_fname,
        sep="\t", header = True, index = False)

def _format_egrns(
        eRegulons: List[eRegulon],
        tf_to_gene: pd.DataFrame):
    """Helper function to format eRegulons to a pandas dataframe."""
    REGION_TO_GENE_COLUMNS = [
        "Region",
        "Gene",
        "importance",
        "rho",
        "importance_x_rho",
        "importance_x_abs_rho"
    ]
    eRegulons_formatted = []
    for ereg in eRegulons:
        TF = ereg.transcription_factor
        is_extended = ereg.is_extended
        region_to_gene = pd.DataFrame(
            ereg.regions2genes,
            columns=REGION_TO_GENE_COLUMNS)
        n_target_regions = len(set(region_to_gene["Region"]))
        n_target_genes = len(set(region_to_gene["Gene"]))
        # TF_[extended,direct]_[+,-]/[+,-]
        eRegulon_name = TF + "_" + \
            ("extended" if is_extended else "direct") + "_" + \
            ("+" if "positive tf2g" in ereg.context else "-") + "/" + \
            ("+" if "positive r2g" in ereg.context else "-")
        # TF_[extended,direct]_[+,-]/[+,-]_(nr)
        region_signature_name = eRegulon_name + "_" + f"({n_target_regions}r)"
        # TF_[extended,direct]_[+,-]/[+,-]_(ng)
        gene_signature_name = eRegulon_name + "_" + f"({n_target_genes}g)"
        # construct dataframe
        region_to_gene["TF"] = TF
        region_to_gene["is_extended"] = is_extended
        region_to_gene["eRegulon_name"] = eRegulon_name
        region_to_gene["Gene_signature_name"] = gene_signature_name
        region_to_gene["Region_signature_name"] = region_signature_name
        eRegulons_formatted.append(region_to_gene)
    eRegulon_metadata = pd.concat(eRegulons_formatted)
    eRegulon_metadata = eRegulon_metadata.merge(
        right=tf_to_gene.rename({"target": "Gene"}, axis = 1), #TODO: rename col beforehand!
        how="left",
        on= ["TF", "Gene"],
        suffixes=["_R2G", "_TF2G"])
    return eRegulon_metadata

def infer_grn(
        TF_to_gene_adj_fname: pathlib.Path,
        region_to_gene_adj_fname: pathlib.Path,
        cistromes_fname: pathlib.Path,
        eRegulon_out_fname: pathlib.Path,
        ranking_db_fname: str,
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
    """
    Infer gene regulatory network.

    Parameters
    ----------
    TF_to_gene_adj_fname : pathlib.Path
        Path to TF to gene adjacency file.
    region_to_gene_adj_fname : pathlib.Path
        Path to region to gene adjacency file.
    cistromes_fname : pathlib.Path
        Path to cistromes file.
    eRegulon_out_fname : pathlib.Path
        Path to store output.
    ranking_db_fname : str
        Path to ranking database.
    is_extended : bool
        Whether to use extended cistromes.
    temp_dir : pathlib.Path
        Path to temporary directory.
    order_regions_to_genes_by : str
        Order regions to genes by.
    order_TFs_to_genes_by : str
        Order TFs to genes by.
    gsea_n_perm : int
        Number of permutations for GSEA.
    quantiles : List[float]
        List of quantiles used for binarizing region-to-gene adjacencies.
    top_n_regionTogenes_per_gene : List[float]
        List of top n regions per gene, used for binarizing
        region-to-gene adjacencies.
    top_n_regionTogenes_per_region : List[float]
        List of top n regions per region, used for binarizing
        region-to-gene adjacencies.
    binarize_using_basc : bool
        Whether to binarize region-to-gene adjacencies using BASC.
    min_regions_per_gene : int
        Minimum number of regions per gene.
    rho_dichotomize_tf2g : bool
        Whether to dichotomize TF-to-gene adjacencies.
    rho_dichotomize_r2g : bool
        Whether to dichotomize region-to-gene adjacencies.
    rho_dichotomize_eregulon : bool
        Whether to dichotomize eRegulons.
    keep_only_activating : bool
        Whether to keep only activating eRegulons.
    rho_threshold : float
        Threshold for dichotomizing.
    min_target_genes : int
        Minimum number of target genes.
    n_cpu : int
        Number of parallel processes to run.

    """
    from scenicplus.grn_builder.gsea_approach import build_grn
    from scenicplus.triplet_score import calculate_triplet_score
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

    log.info("Calculating triplet ranking.")
    eRegulon_metadata = calculate_triplet_score(
        cistromes=cistromes,
        eRegulon_metadata=eRegulon_metadata,
        ranking_db_fname=ranking_db_fname)

    log.info(f"Saving network to {eRegulon_out_fname.__str__()}")
    eRegulon_metadata.to_csv(
        eRegulon_out_fname,
        sep="\t", header=True, index=False)

def calculate_auc(
        eRegulons_fname: pathlib.Path,
        multiome_mudata_fname: pathlib.Path,
        out_file: pathlib.Path,
        n_cpu: int = 1):
    """
    Calculate eRegulon enrichment scores.

    Parameters
    ----------
    eRegulons_fname : pathlib.Path
        Path to eRegulons file.
    multiome_mudata_fname : pathlib.Path
        Path to multiome MuData file.
    out_file : pathlib.Path
        Path to store output.
    n_cpu : int
        Number of parallel processes to run.

    """
    from scenicplus.eregulon_enrichment import score_eRegulons
    log.info("Reading data.")
    mdata = mudata.read(multiome_mudata_fname.__str__())
    eRegulons = pd.read_table(eRegulons_fname)
    log.info("Calculating enrichment scores.")
    gene_region_AUC = score_eRegulons(
        eRegulons=eRegulons,
        gex_mtx=mdata["scRNA"].to_df(),
        acc_mtx=mdata["scATAC"].to_df(),
        n_cpu=n_cpu)
    mdata_AUC = mudata.MuData(
        {
            "Gene_based": mudata.AnnData(X=gene_region_AUC["Gene_based"]),
            "Region_based": mudata.AnnData(X=gene_region_AUC["Region_based"])
        })
    log.info(f"Writing file to {out_file.__str__()}")
    mdata_AUC.write_h5mu(out_file.__str__())

def create_scplus_mudata(
        multiome_mudata_fname: pathlib.Path,
        e_regulon_auc_direct_mudata_fname: pathlib.Path,
        e_regulon_auc_extended_mudata_fname: pathlib.Path,
        e_regulon_metadata_direct_fname: pathlib.Path,
        e_regulon_metadata_extended_fname: pathlib.Path,
        out_file: pathlib.Path):
    from scenicplus.scenicplus_mudata import ScenicPlusMuData
    log.info("Reading multiome MuData.")
    acc_gex_mdata = mudata.read(multiome_mudata_fname.__str__())
    log.info("Reading AUC values.")
    e_regulon_auc_direct = mudata.read(e_regulon_auc_direct_mudata_fname.__str__())
    e_regulon_auc_extended = mudata.read(e_regulon_auc_extended_mudata_fname.__str__())
    log.info("Reading eRegulon metadata.")
    e_regulon_metadata_direct = pd.read_table(e_regulon_metadata_direct_fname)
    e_regulon_metadata_extended = pd.read_table(e_regulon_metadata_extended_fname)
    log.info("Generating MuData object.")
    scplus_mdata = ScenicPlusMuData(
        acc_gex_mdata=acc_gex_mdata,
        e_regulon_auc_direct=e_regulon_auc_direct,
        e_regulon_auc_extended=e_regulon_auc_extended,
        e_regulon_metadata_direct=e_regulon_metadata_direct,
        e_regulon_metadata_extended=e_regulon_metadata_extended)
    log.info(f"Writing file to {out_file.__str__()}")
    scplus_mdata.write_h5mu(out_file.__str__())
