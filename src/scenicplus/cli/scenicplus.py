import sys
import argparse 
import pathlib
from scenicplus.cli import gfx

_DESCRIPTION = "Single-Cell Enhancer-driven gene regulatory Network Inference and Clustering"

def _function(arg: str):
    if not arg.startswith("lambda"):
        raise ValueError("Argument has to be a lambda function definition!")
    return eval(arg)

"""
Functions to create data preparation parsers.
"""

def add_parser_for_prepare_GEX_and_ACC_data(subparser:argparse._SubParsersAction):
    parser:argparse.ArgumentParser = subparser.add_parser(
        name = "prepare_GEX_ACC",
        add_help = True,
        description="""
        Prepare scRNA-seq, scATAC-seq data. Returns a MuData file
        containing linked gene expression and chromatin accessibility data.""")
    def command_prepare_GEX_ACC(arg):
        from scenicplus.cli.commands import prepare_GEX_ACC
        prepare_GEX_ACC(
            cisTopic_obj_fname=arg.cisTopic_obj_fname,
            GEX_anndata_fname=arg.GEX_anndata_fname,
            out_file=arg.out_file,
            use_raw_for_GEX_anndata=(not arg.do_not_use_raw_for_GEX_anndata),
            is_multiome=(not arg.is_not_multiome),
            bc_transform_func=_function(arg.bc_transform_func),
            key_to_group_by=arg.key_to_group_by,
            nr_metacells=arg.nr_metacells,
            nr_cells_per_metacells=arg.nr_cells_per_metacells)
    parser.set_defaults(func=command_prepare_GEX_ACC)
    # Required arguments
    parser.add_argument(
        "--cisTopic_obj_fname", dest="cisTopic_obj_fname",
        action="store", type=pathlib.Path, required=True,
        help="Path to cisTopic object pickle file.")
    parser.add_argument(
        "--GEX_anndata_fname", dest="GEX_anndata_fname",
        action="store", type=pathlib.Path, required=True,
        help="Path to gene expression AnnData h5ad file.")
    parser.add_argument(
        "--out_file", dest="out_file",
        action="store", type=pathlib.Path, required=True,
        help="Out file name (MuData h5mu file).")
    # Optional arguments
    parser.add_argument(
        "--do_not_use_raw_for_GEX_anndata", dest="do_not_use_raw_for_GEX_anndata",
        action="store_true", default=False,
        help="Do not use raw gene expression counts. Default is False")
    parser.add_argument(
        "--is_not_multiome", dest="is_not_multiome",
        action="store_true", default=False,
        help="Data is not multiome. Default is False")
    parser.add_argument(
        "--bc_transform_func", dest="bc_transform_func",
        action="store", type=str,
        default="lambda x: x",
        help="lambda function to transform gene expression cell barcodes into chromatin accessibility barcodes. Default is lambda x: x")
    parser.add_argument(
        "--key_to_group_by", dest="key_to_group_by",
        action="store", type=str,
        default=None,
        help="""For non multi_ome_mode, use this cell metadata key to generate metacells from scRNA-seq and scATAC-seq. 
        Key should be common in scRNA-seq and scATAC-seq side.""")
    parser.add_argument(
        "--nr_metacells", dest="nr_metacells",
        action="store", type=int,
        default=None,
        help="""For non multi_ome_mode, use this number of meta cells to link scRNA-seq and scATAC-seq
        If this is a single integer the same number of metacells will be used for all annotations.
        By default this number is determined automatically so that each cell is sampled at maximum twice.""")
    parser.add_argument(
        "--nr_cells_per_metacells", dest="nr_cells_per_metacells",
        action="store", type=int,
        default=10,
        help="""For non multi_ome_mode, use this number of cells per metacell to link scRNA-seq and scATAC-seq.
        If this is a single integer the same number of cells will be used for all annotations.
        Default is 10""")

def add_parser_for_prepare_menr_data(subparser:argparse._SubParsersAction):
    parser:argparse.ArgumentParser = subparser.add_parser(
        name = "prepare_menr",
        add_help = True,
        description="""
        Prepare motif enrichment data. Returns two AnnData files
        containing cistroms based on direct and extended motif-to-TF annotations.
        Also updates the multiome MuData indicating which genes are TFs.""")
    def prepare_menr_data(arg):
        if len(arg.direct_annotation) > 0 and arg.out_file_direct_annotation is None:
            raise ValueError("Please provide path for --direct_annotation!")
        if len(arg.extended_annotation) > 0 and arg.out_file_extended_annotation is None:
            raise ValueError("Please provide path for --extended_annotation!")
        from scenicplus.cli.commands import prepare_motif_enrichment_results
        prepare_motif_enrichment_results(
            menr_fname=arg.menr_fname,
            multiome_mudata_fname=arg.multiome_mudata_fname,
            out_file_direct_annotation=arg.out_file_direct_annotation,
            out_file_extended_annotation=arg.out_file_extended_annotation,
            out_file_tf_names=arg.out_file_tf_names,
            direct_annotation=arg.direct_annotation,
            extended_annotation=arg.extended_annotation)
    parser.set_defaults(func=prepare_menr_data)
    # Required arguments
    parser.add_argument(
        "--menr_fname", dest="menr_fname",
        action="store", type=pathlib.Path, required=True,
        help="Path to motif enrichment result pickle file (from pycistarget).")
    parser.add_argument(
        "--multiome_mudata_fname", dest="multiome_mudata_fname",
        action="store", type=pathlib.Path, required=True,
        help="Path to multiome MuData object (from scenicplus prepare_GEX_ACC).")
    parser.add_argument(
        "--out_file_tf_names", dest="out_file_tf_names",
        action="store", type=pathlib.Path, required=True,
        help="Out file name for TF names (txt file).")
    # Optional arguments
    parser.add_argument(
        "--out_file_direct_annotation", dest="out_file_direct_annotation",
        action="store", type=pathlib.Path, required=False,
        help="Out file name for direct cistromes (AnnData h5ad file).")
    parser.add_argument(
        "--out_file_extended_annotation", dest="out_file_extended_annotation",
        action="store", type=pathlib.Path, required=False,
        help="Out file name for extended cistromes (AnnData h5ad file).")
    parser.add_argument(
        "--direct_annotation", dest="direct_annotation",
        action="store", type=str, required=False, nargs='+',
        default=['Direct_annot'],
        help="Annotations to use as direct. Default is 'Direct_annot'")
    parser.add_argument(
        "--extended_annotation", dest="extended_annotation",
        action="store", type=str, required=False, nargs='+',
        default=['Orthology_annot'],
        help="Annotations to use as extended. Default is 'Orthology_annot'")

def add_parser_for_download_genome_annotations(subparser:argparse._SubParsersAction):
    parser:argparse.ArgumentParser = subparser.add_parser(
        name = "download_genome_annotations",
        add_help = True,
        description="""
        Download genome annotation and chromsizes and save to tsv""")
    def download_command(arg):
        from scenicplus.cli.commands import download_gene_annotation_chromsizes
        download_gene_annotation_chromsizes(
            species=arg.species,
            genome_annotation_out_fname=arg.genome_annotation_out_fname,
            chromsizes_out_fname=arg.chromsizes_out_fname,
            biomart_host=arg.biomart_host,
            use_ucsc_chromosome_style=(not arg.do_not_use_ucsc_chromosome_style))
    parser.set_defaults(func=download_command)
    # Required arguments
    parser.add_argument(
        "--species", dest="species",
        action="store", type=pathlib.Path, required=True,
        help="Species name (e.g. hsapies).")
    parser.add_argument(
        "--genome_annotation_out_fname", dest="genome_annotation_out_fname",
        action="store", required=True,
        help="Out file name for genome annotation (tsv).")
    parser.add_argument(
        "--chromsizes_out_fname", dest="chromsizes_out_fname",
        action="store", required=True,
        help="Out file name for chromosome sizes (tsv).")
    # Optional arguments
    parser.add_argument(
        "--biomart_host", dest="biomart_host",
        action="store", type=str, required=False,
        default = "http://www.ensembl.org",
        help="Biomart host name")
    parser.add_argument(
        "--do_not_use_ucsc_chromosome_style", dest="do_not_use_ucsc_chromosome_style",
        action="store_true",
        help="Do not use UCSC chromosome style names.")

def add_parser_for_search_space(subparser:argparse._SubParsersAction):
    parser:argparse.ArgumentParser = subparser.add_parser(
        name = "search_spance",
        add_help = True,
        description="""
        Get search space for each gene. Returns tsv with search spance""")
    def search_space(arg):
        from scenicplus.cli.commands import get_search_space_command
        get_search_space_command(
            multiome_mudata_fname=arg.multiome_mudata_fname,
            gene_annotation_fname=arg.gene_annotation_fname,
            chromsizes_fname=arg.chromsizes_fname,
            out_fname=arg.out_fname,
            use_gene_boundaries=arg.use_gene_boundaries,
            upstream=arg.upstream,
            downstream=arg.downstream,
            extend_tss=arg.extend_tss,
            remove_promoters=arg.remove_promoters)
    parser.set_defaults(func=search_space)
    # Required arguments
    parser.add_argument(
        "--multiome_mudata_fname", dest="multiome_mudata_fname",
        action="store", type=pathlib.Path, required=True,
        help="Path to multiome MuData object (from scenicplus prepare_GEX_ACC).")
    parser.add_argument(
        "--gene_annotation_fname", dest="gene_annotation_fname",
        action="store", type=pathlib.Path, required=True,
        help="Path to gene annotation tsv (from scenicplus download_genome_annotations).")
    parser.add_argument(
        "--chromsizes_fname", dest="chromsizes_fname",
        action="store", type=pathlib.Path, required=True,
        help="Path to chromosome sizes tsv (from scenicplus download_genome_annotations).")
    parser.add_argument(
        "--out_fname", dest="out_fname",
        action="store", type=pathlib.Path, required=True,
        help="Out file name for gene search space (tsv).")
    # Optional arguments
    parser.add_argument(
        "--use_gene_boundaries", dest="use_gene_boundaries",
        action="store_true",
        help="Whether to use the whole search space or stop when encountering another gene.")
    parser.add_argument(
        "--upstream", dest="upstream",
        action="store", type=int, required=False,
        nargs=2, default=[1000, 150000],
        help="""Search space upstream. The minimum (first position) means that even if there is a gene right next to it these
                bp will be taken. The second position indicates the maximum distance.
                Default is 1000 150000""")
    parser.add_argument(
        "--downstream", dest="downstream",
        action="store", type=int, required=False,
        nargs=2, default=[1000, 150000],
        help="""Search space downstream. The minimum (first position) means that even if there is a gene right next to it these
                bp will be taken. The second position indicates the maximum distance.
                Default is 1000 150000""")
    parser.add_argument(
        "--extend_tss", dest="extend_tss",
        action="store", type=int, required=False,
        nargs=2, default=[10, 10],
        help="Space around the TSS consider as promoter. Default is 10 10")
    parser.add_argument(
        "--remove_promoters", dest="remove_promoters",
        action="store_true",
        help="Whether to remove promoters from the search space or not.")


"""
Functions to create GRN inference parsers.
"""

def add_parser_for_infer_TF_to_gene(subparser:argparse._SubParsersAction):
    parser:argparse.ArgumentParser = subparser.add_parser(
        name = "TF_to_gene",
        add_help = True,
        description="""
        Infer TF-to-gene relationships""")
    def TF_to_gene(arg):
        from scenicplus.cli.commands import infer_TF_to_gene
        infer_TF_to_gene(
            multiome_mudata_fname=arg.multiome_mudata_fname,
            tf_names_fname=arg.tf_names,
            temp_dir=arg.temp_dir,
            adj_out_fname=arg.out_tf_to_gene_adjacencies,
            method=arg.method,
            n_cpu=arg.n_cpu,
            seed=arg.seed)
    parser.set_defaults(func=TF_to_gene)
    # Required arguments
    parser.add_argument(
        "--multiome_mudata_fname", dest="multiome_mudata_fname",
        action="store", type=pathlib.Path, required=True,
        help="Path to multiome MuData object (from scenicplus prepare_GEX_ACC).")
    parser.add_argument(
        "--tf_names", dest="tf_names",
        action="store", type=pathlib.Path, required=True,
        help="Path TF names (from scenicplus prepare_menr).")
    parser.add_argument(
        "--temp_dir", dest="temp_dir",
        action="store", type=pathlib.Path, required=True,
        help="Path temp dir.")
    parser.add_argument(
        "--out_tf_to_gene_adjacencies", dest="out_tf_to_gene_adjacencies",
        action="store", type=pathlib.Path, required=True,
        help="Out file name to store TF to gene adjacencies (tsv)")
    
    # Optional arguments
    parser.add_argument(
        "--method", dest="method",
        action="store", choices = ["GBM", "RF"], required=False,
        default = "GBM",
        help="Regression method to use, either GBM (Gradient Boosting Machine) or RF (Random Forrest). Default is GBM")
    parser.add_argument(
        "--n_cpu", dest="n_cpu",
        action="store", type=int, required=False,
        default=1,
        help="Number of cores to use. Default is 1.")
    parser.add_argument(
        "--seed", dest="seed",
        action="store", type=int, required=False,
        default=666,
        help="Seed to use. Default is 666.")

def add_parser_for_infer_region_to_gene(subparser:argparse._SubParsersAction):
    parser:argparse.ArgumentParser = subparser.add_parser(
        name = "region_to_gene",
        add_help = True,
        description="""
        Infer region-to-gene relationships""")
    def TF_to_gene(arg):
        from scenicplus.cli.commands import infer_region_to_gene
        infer_region_to_gene(
            multiome_mudata_fname=arg.multiome_mudata_fname,
            search_space_fname=arg.search_space_fname,
            temp_dir=arg.temp_dir,
            adj_out_fname=arg.out_region_to_gene_adjacencies,
            importance_scoring_method=arg.importance_scoring_method,
            correlation_scoring_method=arg.correlation_scoring_method,
            mask_expr_dropout=arg.mask_expr_dropout,
            n_cpu = arg.n_cpu)
    parser.set_defaults(func=TF_to_gene)
    # Required arguments
    parser.add_argument(
        "--multiome_mudata_fname", dest="multiome_mudata_fname",
        action="store", type=pathlib.Path, required=True,
        help="Path to multiome MuData object (from scenicplus prepare_GEX_ACC).")
    parser.add_argument(
        "--search_space_fname", dest="search_space_fname",
        action="store", type=pathlib.Path, required=True,
        help="Path to search space dataframe (from scenicplus search_spance).")
    parser.add_argument(
        "--temp_dir", dest="temp_dir",
        action="store", type=pathlib.Path, required=True,
        help="Path temp dir.")
    parser.add_argument(
        "--out_region_to_gene_adjacencies", dest="out_region_to_gene_adjacencies",
        action="store", type=pathlib.Path, required=True,
        help="Path to store region to gene adjacencies (tsv).")

    # Optional arguments
    parser.add_argument(
        "--importance_scoring_method", dest="importance_scoring_method",
        action="store", choices = ['RF', 'ET', 'GBM'], required=False,
        default = "GBM",
        help="Regression method to use, either GBM (Gradient Boosting Machine), RF (Random Forrest) or ET (Extra Trees). Default is GBM.")
    parser.add_argument(
        "--correlation_scoring_method", dest="correlation_scoring_method",
        action="store", choices = ['PR', 'SR'], required=False,
        default = "SR",
        help="Correlation method to use, either PR (Pearson correlation) or SR (Spearman Rank correlation). Default is SR.")
    parser.add_argument(
        "--mask_expr_dropout", dest="mask_expr_dropout",
        action="store_true",
        help="Whether to mask expression dropouts. Default is False.")
    parser.add_argument(
        "--n_cpu", dest="n_cpu",
        action="store", type=int, required=False,
        default=1,
        help="Number of cores to use. Default is 1.")

def add_parser_for_infer_egrn(subparser:argparse._SubParsersAction):
    parser:argparse.ArgumentParser = subparser.add_parser(
        name = "eGRN",
        add_help = True,
        description="""
        Infer enhancer-driven Gene Regulatory Network (eGRN)""")
    def eGRN(arg):
        from scenicplus.cli.commands import infer_grn
        infer_grn(
            TF_to_gene_adj_fname=arg.TF_to_gene_adj_fname,
            region_to_gene_adj_fname=arg.region_to_gene_adj_fname,
            cistromes_fname=arg.cistromes_fname,
            eRegulon_out_fname=arg.eRegulon_out_fname,
            is_extended=arg.is_extended,
            temp_dir=arg.temp_dir,
            order_regions_to_genes_by=arg.order_regions_to_genes_by,
            order_TFs_to_genes_by=arg.order_TFs_to_genes_by,
            gsea_n_perm=arg.gsea_n_perm,
            quantiles=arg.quantiles,
            top_n_regionTogenes_per_gene=arg.top_n_regionTogenes_per_gene,
            top_n_regionTogenes_per_region=arg.top_n_regionTogenes_per_region,
            binarize_using_basc=not(arg.do_not_binarize_using_basc),
            min_regions_per_gene=arg.min_regions_per_gene,
            rho_dichotomize_tf2g=not(arg.do_not_rho_dichotomize_tf2g),
            rho_dichotomize_r2g=not(arg.do_not_rho_dichotomize_r2g),
            rho_dichotomize_eregulon=not(arg.do_not_rho_dichotomize_eRegulon),
            keep_only_activating=arg.keep_only_activating_eRegulons,
            rho_threshold=arg.rho_threshold,
            min_target_genes=arg.min_target_genes,
            n_cpu=arg.n_cpu)
            
    parser.set_defaults(func=eGRN)
    # Required arguments
    parser.add_argument(
        "--TF_to_gene_adj_fname", dest="TF_to_gene_adj_fname",
        action="store", type=pathlib.Path, required=True,
        help="Path to TF-to-gene adjacencies (.tsv) from scenicplus TF_to_gene.")
    parser.add_argument(
        "--region_to_gene_adj_fname", dest="region_to_gene_adj_fname",
        action="store", type=pathlib.Path, required=True,
        help="Path to region-to-gene adjacencies (.tsv) from scenicplus region_to_gene.")
    parser.add_argument(
        "--cistromes_fname", dest="cistromes_fname",
        action="store", type=pathlib.Path, required=True,
        help="Path to either direct or extended cistromes (.h5ad) from scenicplus prepare_menr.")
    parser.add_argument(
        "--eRegulon_out_fname", dest="eRegulon_out_fname",
        action="store", type=pathlib.Path, required=True,
        help="Path to save eRegulon dataframe (.tsv)")
    parser.add_argument(
        "--temp_dir", dest="temp_dir",
        action="store", type=pathlib.Path, required=True,
        help="Path to temp dir.")
    # Optional arguments
    parser.add_argument(
        "--is_extended", dest="is_extended",
        action="store_true",
        help="Use this when cistromes are based on extended annotation. Default is False.")
    parser.add_argument(
        "--order_regions_to_genes_by", dest="order_regions_to_genes_by",
        action="store", type=str, required=False,
        default="importance",
        help="Column by which to order the region-to-gene links. Default is 'importance'.")
    parser.add_argument(
        "--order_TFs_to_genes_by", dest="order_TFs_to_genes_by",
        action="store", type=str, required=False,
        default="importance",
        help="Column by which to order the TF-to-gene links. Default is 'importance'.")
    parser.add_argument(
        "--gsea_n_perm", dest="gsea_n_perm",
        action="store", type=int, required=False,
        default=1000,
        help="Number of permutations to run for GSEA. Default is 1000.")
    parser.add_argument(
        "--quantiles", dest="quantiles",
        action="store", type=float, required=False,
        nargs="*", default=[0.85, 0.90, 0.95],
        help="Quantiles for thresholding region-to-gene links. Default is [0.85, 0.90, 0.95]")
    parser.add_argument(
        "--top_n_regionTogenes_per_gene", dest="top_n_regionTogenes_per_gene",
        action="store", type=int, required=False,
        nargs="*", default=[5, 10, 15],
        help="Top n region-to-gene links per gene for thresholding region-to-gene links. Default is [5, 10, 15]")
    parser.add_argument(
        "--top_n_regionTogenes_per_region", dest="top_n_regionTogenes_per_region",
        action="store", type=int, required=False,
        nargs="*", default=[],
        help="Top n region-to-gene links per region for thresholding region-to-gene links. Default is []")
    parser.add_argument(
        "--do_not_binarize_using_basc", dest="do_not_binarize_using_basc",
        action="store_true",
        help="Don't use BASC to binarize region to gene links. By default BASC is used.")
    parser.add_argument(
        "--min_regions_per_gene", dest="min_regions_per_gene",
        action="store", type=int, required=False,
        default=0,
        help="Minimum regions per gene. Default is 0.")
    parser.add_argument(
        "--do_not_rho_dichotomize_tf2g", dest="do_not_rho_dichotomize_tf2g",
        action="store_true",
        help="Don't split positive and negative TF-to-gene links. By default they are split.")
    parser.add_argument(
        "--do_not_rho_dichotomize_r2g", dest="do_not_rho_dichotomize_r2g",
        action="store_true",
        help="Don't split positive and negative region-to-gene links. By default they are split.")
    parser.add_argument(
        "--do_not_rho_dichotomize_eRegulon", dest="do_not_rho_dichotomize_eRegulon",
        action="store_true",
        help="Don't split positive and negative eRegulons. By default they are split.")
    parser.add_argument(
        "--keep_only_activating_eRegulons", dest="keep_only_activating_eRegulons",
        action="store_true",
        help="Keep only activating eRegulons. By default both activating and repressive eRegulons are kept.")
    parser.add_argument(
        "--rho_threshold", dest="rho_threshold",
        action="store", type=float, required=False,
        default=0.05,
        help="Threshold on correlation coefficient used for splitting positive and negative interactions. Default is 0.05")
    parser.add_argument(
        "--min_target_genes", dest="min_target_genes",
        action="store", type=int, required=False,
        default=10,
        help="Minimum number of target genes per eRegulon, eRegulon with a lower number of target genes will be discarded. Default is 10")
    parser.add_argument(
        "--n_cpu", dest="n_cpu",
        action="store", type=int, required=False,
        default=1,
        help="Number of cores to use. Default is 1.")

def create_argument_parser():
    parser = argparse.ArgumentParser(
        description=_DESCRIPTION)
    subparsers = parser.add_subparsers()

    """
    Data preparation parsers
    """
    preprocessing_parser = subparsers.add_parser(
        "prepare_data", 
        description="Prepare gene expression, chromatin accessibility and motif enrichment data.")
    preprocessing_subparsers = preprocessing_parser.add_subparsers()
    # Create data preparation parsers
    add_parser_for_prepare_GEX_and_ACC_data(preprocessing_subparsers)
    add_parser_for_prepare_menr_data(preprocessing_subparsers)
    add_parser_for_download_genome_annotations(preprocessing_subparsers)
    add_parser_for_search_space(preprocessing_subparsers)

    """
    GRN inference parsers
    """
    inference_parser = subparsers.add_parser(
        "grn_inference",
        description="Infer Enhancer driven Gene Regulatory Networks.")
    inference_subparsers = inference_parser.add_subparsers()
    # Create inference parsers
    add_parser_for_infer_TF_to_gene(inference_subparsers)
    add_parser_for_infer_region_to_gene(inference_subparsers)
    add_parser_for_infer_egrn(inference_subparsers)
    return parser

def main(argv=None) -> int:
    #parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args(args=argv)
    if not hasattr(args, "func"):
        print(gfx.logo)
        parser.print_help()
    else:
        args.func(args)
    return 0

if __name__ == '__main__':
    sys.exit(main())