import sys
import argparse 
import pathlib
from scenicplus.cli import gfx
from scenicplus.cli.commands import (
    prepare_GEX_ACC
)

_DESCRIPTION = "Single-Cell Enhancer-driven gene regulatory Network Inference and Clustering"

def _function(arg: str):
    if not arg.startswith("lambda"):
        raise ValueError("Argument has to be a lambda function definition!")
    return eval(arg)

def prepare_GEX_and_ACC_data(subparser:argparse._SubParsersAction):
    parser:argparse.ArgumentParser = subparser.add_parser(
        name = "prepare_GEX_ACC",
        add_help = True,
        description="""
        Prepare scRNA-seq, scATAC-seq data. Returns a MuData file
        containing linked gene expression and chromatin accessibility data.""")
    def command_prepare_GEX_ACC(arg):
        prepare_GEX_ACC(
            cisTopic_obj_fname=arg.cisTopic_obj_fname,
            GEX_anndata_fname=arg.GEX_anndata_fname,
            out_file=arg.out_file,
            use_raw_for_GEX_anndata=(not arg.dont_use_raw_for_GEX_anndata),
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
        help="Out file name (MuData h5ad file).")
    # Optional arguments
    parser.add_argument(
        "--dont_use_raw_for_GEX_anndata", dest="dont_use_raw_for_GEX_anndata",
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

def create_argument_parser():
    parser = argparse.ArgumentParser(
        description=_DESCRIPTION)
    subparsers = parser.add_subparsers(help="sub-command help")
    prepare_GEX_and_ACC_data(subparsers)
    return parser

def main(argv=None) -> int:
    #parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args(args=argv)
    print(gfx.logo)
    if not hasattr(args, "func"):
        parser.print_help()
    else:
        args.func(args)
    return 0

if __name__ == '__main__':
    sys.exit(main())