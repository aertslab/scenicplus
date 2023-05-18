import sys
import argparse 
from scenicplus.cli import gfx

_DESCRIPTION = "Single-Cell Enhancer-driven gene regulatory Network Inference and Clustering"

def create_argument_parser():
    parser = argparse.ArgumentParser(
        description=_DESCRIPTION)
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