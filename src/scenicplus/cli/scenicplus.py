import sys
import argparse 
import pathlib
from scenicplus.cli import gfx

_DESCRIPTION = "Single-Cell Enhancer-driven gene regulatory Network Inference and Clustering"

def _function(arg: str):
    if not arg.startswith("lambda"):
        raise ValueError("Argument has to be a lambda function definition!")
    return eval(arg)

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
        help="Do not use raw gene expression counts.")
    parser.add_argument(
        "--is_not_multiome", dest="is_not_multiome",
        action="store_true", default=False,
        help="Data is not multiome")
    parser.add_argument(
        "--bc_transform_func", dest="bc_transform_func",
        action="store", type=str,
        default="lambda x: x",
        help="lambda function to transform gene expression cell barcodes into chromatin accessibility barcodes")
    parser.add_argument(
        "--key_to_group_by", dest="key_to_group_by",
        action="store", type=str,
        default=None,
        help="""For non multi_ome_mode, use this cell metadata key to generate metacells from scRNA-seq and scATAC-seq. 
        Key should be common in scRNA-seq and scATAC-seq side""")
    parser.add_argument(
        "--nr_metacells", dest="nr_metacells",
        action="store", type=int,
        default=None,
        help="""For non multi_ome_mode, use this number of meta cells to link scRNA-seq and scATAC-seq
        If this is a single integer the same number of metacells will be used for all annotations.""")
    parser.add_argument(
        "--nr_cells_per_metacells", dest="nr_cells_per_metacells",
        action="store", type=int,
        default=10,
        help="""For non multi_ome_mode, use this number of cells per metacell to link scRNA-seq and scATAC-seq.
        If this is a single integer the same number of cells will be used for all annotations.""")

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
        help="Annotations to use as direct.")
    parser.add_argument(
        "--extended_annotation", dest="extended_annotation",
        action="store", type=str, required=False, nargs='+',
        default=['Orthology_annot'],
        help="Annotations to use as extended.")

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
        action="store", default=False, required=True,
        help="Out file name for genome annotation (tsv).")
    parser.add_argument(
        "--chromsizes_out_fname", dest="chromsizes_out_fname",
        action="store", default=False, required=True,
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
                bp will be taken. The second position indicates the maximum distance.""")
    parser.add_argument(
        "--downstream", dest="downstream",
        action="store", type=int, required=False,
        nargs=2, default=[1000, 150000],
        help="""Search space downstream. The minimum (first position) means that even if there is a gene right next to it these
                bp will be taken. The second position indicates the maximum distance.""")
    parser.add_argument(
        "--extend_tss", dest="extend_tss",
        action="store", type=int, required=False,
        nargs=2, default=[10, 10],
        help="Space around the TSS consider as promoter.")
    parser.add_argument(
        "--remove_promoters", dest="remove_promoters",
        action="store_true",
        help="Whether to remove promoters from the search space or not.")

def create_argument_parser():
    parser = argparse.ArgumentParser(
        description=_DESCRIPTION)
    prepare_subparsers = parser.add_subparsers(help="Prepare data")
    add_parser_for_prepare_GEX_and_ACC_data(prepare_subparsers)
    add_parser_for_prepare_menr_data(prepare_subparsers)
    add_parser_for_download_genome_annotations(prepare_subparsers)
    add_parser_for_search_space(prepare_subparsers)
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