import argparse
import logging
from .gfx import logo, logo_r2g

def region_to_gene_command(args):
    print(logo_r2g)
    expression_mtx_fname = args.expression_mtx_fname.name
    imputed_accessibility_obj_fname = args.imputed_accessibility_obj_fname.name
    search_space_fname = args.search_space_fname.name
    scale_factor_ATAC_normalization = args.scale_factor_ATAC_normalization
    scale_factor_RNA_normalization = args.scale_factor_RNA_normalization
    importance_scoring_method = args.importance_scoring_method
    correlation_scoring_method = args.correlation_scoring_method
    output = args.output
    n_cpu = args.n_cpu
    temp_dir = args.temp_dir
    object_store_memory = args.object_store_memory
    memory = args.memory

    #Imports
    import sys
    import pandas as pd
    import numpy  as np
    import pickle
    from pycisTopic.diff_features import impute_accessibility, normalize_scores
    import loompy as lp
    from scanpy import AnnData
    import scanpy as sc
    import os
    from scenicplus.enhancer_to_gene import calculate_regions_to_genes_relationships, get_search_space, rank_aggregation, export_to_UCSC_interact
    from scenicplus.enhancer_to_gene import RF_KWARGS, ET_KWARGS, GBM_KWARGS
    KWARGS_dict = {'RF': RF_KWARGS, 'ET': ET_KWARGS, 'GBM': GBM_KWARGS} #TODO: add SGBM
    import pyranges as pr
    from scenicplus.utils import region_names_to_coordinates

    #Create logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('R2G')

    log.info('Loading imputed accessbility object.')
    with open(imputed_accessibility_obj_fname, 'rb') as f:
        imputed_acc_obj = pickle.load(f)
    
    log.info('Normalizing region cell probabilities ...')
    norm_imputed_acc_obj = normalize_scores(imputed_acc_obj, scale_factor = scale_factor_ATAC_normalization)

    log.info('Loading expression matrix.')
    with open(expression_mtx_fname, 'rb') as f:
        exprMat = pickle.load(f)
    
    log.info('Normalizing expression data ...')
    adata = AnnData(exprMat)
    sc.pp.normalize_total(adata, target_sum=scale_factor_RNA_normalization)
    sc.pp.log1p(adata)

    #convert accessbility and gene expression data to dataframes
    df_exp_mat = adata.to_df()
    df_acc_mat = pd.DataFrame(norm_imputed_acc_obj.mtx.T, 
                    index = norm_imputed_acc_obj.cell_names, 
                    columns = norm_imputed_acc_obj.feature_names)
    
    log.info('setting expression and accessbility data to common index.')
    cells_common_exp_acc = set(df_exp_mat.index) & set(df_acc_mat.index)
    df_exp_mat = df_exp_mat.loc[cells_common_exp_acc]
    df_acc_mat = df_acc_mat.loc[cells_common_exp_acc]

    log.info('Loading search space')
    search_space = pd.read_csv(search_space_fname, sep = '\t')

    region_to_gene = calculate_regions_to_genes_relationships(
        imputed_acc_mtx=df_acc_mat,
        expr_mtx=df_exp_mat,
        search_space=search_space,
        mask_expr_dropout=False,
        importance_scoring_method = importance_scoring_method,
        correlation_scoring_method= correlation_scoring_method,
        importance_scoring_kwargs=KWARGS_dict[importance_scoring_method],
        ray_n_cpu=n_cpu,
        _temp_dir = temp_dir,
        object_store_memory = object_store_memory,
        _memory = memory
    )
    log.info('Writing to file: {}'.format(output))
    region_to_gene.to_csv(output, header = True, index = False, sep = '\t')
    log.info('Done!')
    


def create_argument_parser():
    parser = argparse.ArgumentParser(description = '')

    subparsers = parser.add_subparsers(help='sub-command help')

    #----------------------------------------------------------------------------#
    # Create parser for calculating enhancer to gene importances and correlation #
    #----------------------------------------------------------------------------#
    parser_r2g       = subparsers.add_parser('r2g', help = 'Derive region to gene links.')
    parser_r2g.add_argument(
        '--expression_mtx_fname',
        type = argparse.FileType('r'),
        help = 'Path to pickle file containing expression matrix'
    )
    parser_r2g.add_argument(
        '--imputed_accessibility_obj_fname',
        type = argparse.FileType('r'),
        help = 'Path to pickle file containing imputed accessibility object'
    )
    parser_r2g.add_argument(
        '--search_space_fname',
        type = argparse.FileType('r'),
        help = 'Path to tsv file containing region to gene search space.'
    )
    parser_r2g.add_argument(
        '--scale_factor_ATAC_normalization',
        type = int,
        default = 10**4,
        help = '[OPTIONAL] Scale factor to use when normalizing region cell probabilities.'
    )
    parser_r2g.add_argument(
        '--scale_factor_RNA_normalization',
        type = int,
        default = 10**4,
        help = '[OPTIONAL] Scale factor to use when normalizing gene expression values.'
    )
    parser_r2g.add_argument(
        '--importance_scoring_method',
        type = str,
        choices = ['RF', 'ET', 'GBM'],
        default = 'RF',
        help = '[OPTIONAL] Method to use to calculate region to gene importance scores. (RF: Random Forrest, ET: Extra Trees, GBM: Gradient Boosting Machine). Default is RF.'
    )
    parser_r2g.add_argument(
        '--correlation_scoring_method',
        type = str,
        choices = ['grp_requiredSR', 'PR'],
        default = 'SR',
        help = '[OPTIONAL] Method to use to calculate region to gene correlation coefficients. (SR: Spearmanr, PR: Pearsonr). Default is SR.'
    )
    parser_r2g.add_argument(
        '-o',
        '--output',
        type = str,
        help = 'Output folder where region to gene tsv and bigbed interact file will be written.'
    )
    parser_r2g.add_argument(
        '-n',
        '--n_cpu',
        type = int,
        help = 'Number of cores to use.'
    )
    parser_r2g.add_argument(
        '--temp_dir',
        type = str,
        help = 'temp directory for ray.'
    )
    
    parser_r2g.add_argument(
        '--object_store_memory',
        type = int,
        default = 12e+10,
        help = 'object store memory for ray.'
    )
    
    parser_r2g.add_argument(
        '--memory',
        type = int,
        default = 18e+10,
        help = 'memory for ray.'
    )
    parser_r2g.set_defaults(func=region_to_gene_command)
    return parser

def main(argv=None):
    print(logo)
    # Parse arguments.
    parser = create_argument_parser()
    args = parser.parse_args(args=argv)
    if not hasattr(args, 'func'):
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()