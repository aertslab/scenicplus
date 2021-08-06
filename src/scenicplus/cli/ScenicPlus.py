import argparse
import logging
from .gfx import logo, logo_r2g

def region_to_gene_command(args):
    #TODO: add argument validation!
    print(logo_r2g)
    
    #------------#
    # Parse args #
    #------------#

    #general args:
    f_expr_loom                     = args.expression_loom_fname.name
    f_cisTopic_obj                  = args.cisTopic_obj_fname.name
    d_output                        = args.output
    ray_n_cpu                       = args.n_cpu
    ray_temp_dir                    = args.temp_dir
    species                         = args.species
    assembly                        = args.assembly

    #preprocessing args:
    scale_factor_imputation         = args.scale_factor_imputation
    scale_factor_ATAC_normalization = args.scale_factor_ATAC_normalization
    scale_factor_RNA_normalization  = args.scale_factor_RNA_normalization

    #search space args:
    use_gene_boundaries             = args.use_gene_boundaries
    upstream_min                    = args.upstream_min
    upstream_max                    = args.upstream_max
    downstream_min                  = args.downstream_min
    downstream_max                  = args.downstream_max
    extend_TSS_downstream           = args.extend_TSS_downstream
    extend_TSS_upstream             = args.extend_TSS_upstream
    remove_distal_promoters         = args.remove_distal_promoters

    #region to gene args:
    regions_bed                     = args.regions.name
    importance_scoring_method       = args.importance_scoring_method
    correlation_scoring_method      = args.correlation_scoring_method

    #rank aggregation args:
    rank_aggr_group                 = args.rank_aggr_group
    rank_aggr_distance_method       = args.rank_aggr_distance_method
    rank_aggr_ceil                  = args.rank_aggr_ceil
    rank_aggr_threshold             = args.rank_aggr_threshold

    #export to bigbed args:
    path_bedToBigBed                = args.path_bedToBigBed
    ucsc_track_name                 = args.ucsc_track_name
    ucsc_description                = args.ucsc_description
    cmap_neg                        = args.cmap_neg
    cmap_pos                        = args.cmap_pos
    vmin                            = args.vmin
    vmax                            = args.vmax

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

    #open cistopic object, calculate region cell prob. and normalize
    log.info('Loading cisTopic object ...')
    with open(f_cisTopic_obj, 'rb') as f:
        cistopic_obj = pickle.load(f)

    log.info('Calculating region cell probabilities ...')
    imputed_acc_obj = impute_accessibility(cistopic_obj, selected_cells=None, selected_regions=None, scale_factor=scale_factor_imputation)

    log.info('Normalizing region cell probabilities ...')
    norm_imputed_acc_obj = normalize_scores(imputed_acc_obj, scale_factor = scale_factor_ATAC_normalization)

    #open expression data and normalize
    log.info('Loading expression data ...')
    with lp.connect(f_expr_loom, mode = 'r+', validate = False) as lf:
        exprMat = pd.DataFrame(lf[:,:], index=lf.ra.Gene, columns=lf.ca.CellID).T
    
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

    if regions_bed is not None:
        log.info('Reading: {}'.format(regions_bed))
        pr_regions = pr.read_bed(regions_bed)
        log.info('Using provided regions for calculating region to gene links.')
    else:
        log.info('Using all consensus regions for calculating region to gene links.')
        pr_regions = pr.PyRanges(region_names_to_coordinates(cistopic_obj.feature_names))
    
    log.info('Getting search space')
    search_space = get_search_space(
        pr_regions = pr_regions,
        species = species,
        assembly = assembly,
        use_gene_boundaries = use_gene_boundaries,
        upstream = [upstream_min, upstream_max],
        downstream = [downstream_min, downstream_max],
        extend_tss = [extend_TSS_downstream, extend_TSS_upstream],
        remove_promoters = remove_distal_promoters )
    
    region_to_gene = calculate_regions_to_genes_relationships(
        imputed_acc_mtx=df_acc_mat,
        expr_mtx=df_exp_mat,
        search_space=search_space,
        mask_expr_dropout=False,
        importance_scoring_method = importance_scoring_method,
        correlation_scoring_method= correlation_scoring_method,
        importance_scoring_kwargs=KWARGS_dict[importance_scoring_method],
        ray_n_cpu=ray_n_cpu,
        _temp_dir = ray_temp_dir
    )
    
    log.info('Calculating aggregated rank of importance score, correlation coef. and max mean accessbility.')
    grouping_vector = cistopic_obj.cell_data.loc[df_acc_mat.index, rank_aggr_group]
    rank_aggregation(
        region_to_gene = region_to_gene,
        imputed_acc_mtx = df_acc_mat,
        grouping_vector = grouping_vector,
        method = rank_aggr_distance_method,
        ray_n_cpu = ray_n_cpu,
        return_copy = False,
        scale_ceil = rank_aggr_ceil,
        _temp_dir = ray_temp_dir)
    
    f_output_r2g = os.path.join(d_output, 'region_to_gene.tsv')
    log.info('Writing region to gene table to: {}'.format(f_output_r2g))
    region_to_gene.to_csv(f_output_r2g, header = True, index = False, sep = '\t')
    
    log.info('Exporting interaction track')
    _ = export_to_UCSC_interact(
        region_to_gene_df = region_to_gene,
        species = species,
        aggr_rank_score_thr = rank_aggr_threshold,
        outfile = os.path.join(d_output, 'region_to_gene.interact'),
        bigbed_outfile = os.path.join(d_output, 'region_to_gene.inter.bb'),
        path_bedToBigBed = path_bedToBigBed,
        assembly = assembly,
        ucsc_track_name = ucsc_track_name,
        ucsc_description = ucsc_description,
        cmap_neg = cmap_neg,
        cmap_pos = cmap_pos,
        vmin = vmin,
        vmax = vmax

    )

    log.info('Done!')

def create_argument_parser():
    parser = argparse.ArgumentParser(description = '')

    subparsers = parser.add_subparsers(help='sub-command help')

    #----------------------------------------------------------------------------#
    # Create parser for calculating enhancer to gene importances and correlation #
    #----------------------------------------------------------------------------#

    parser_r2g       = subparsers.add_parser('r2g', help = 'Derive region to gene links.')

    grp_gen          = parser_r2g.add_argument_group('General arguments')
    grp_pp           = parser_r2g.add_argument_group('Preprocessing arguments')
    grp_search_space = parser_r2g.add_argument_group('Search Space arguments')
    grp_scoring      = parser_r2g.add_argument_group('Region to Gene scoring arguments')
    grp_rnk          = parser_r2g.add_argument_group('Rank aggregation arguments')
    grp_export       = parser_r2g.add_argument_group('BigBed export arguments')

    #general arguments

    grp_gen.add_argument(
        '--expression_loom_fname',
        type = argparse.FileType('r'),
        help = 'Loom file containing raw expression matrix. e.g. loomfile outputed by pySCENIC'
    )
    grp_gen.add_argument(
        '--cisTopic_obj_fname',
        type = argparse.FileType('r'),
        help = 'Pickled file containing cisTopic object.'
    )
    grp_gen.add_argument(
        '-o',
        '--output',
        type = str,
        help = 'Output folder where region to gene tsv and bigbed interact file will be written.'
    )
    grp_gen.add_argument(
        '-n',
        '--n_cpu',
        type = int,
        help = 'Number of cores to use.'
    )
    grp_gen.add_argument(
        '--temp_dir',
        type = str,
        help = 'temp directory for ray.'
    )
    grp_gen.add_argument(
        '--species',
        type = str,
        help = 'Name of the species on which the data is generated (e.g. hsapiens)'
    )
    grp_gen.add_argument(
        '--assembly',
        type = str,
        help = 'Assembly of the reference genome on which the data is mapped (e.g. hg38)'
    )
    

    #preprocessing arguments
    grp_pp.add_argument(
        '--scale_factor_imputation',
        type = int,
        default = 10**6,
        help = '[OPTIONAL] Scale factor to use when calculating region cell probabilities.'
    )
    grp_pp.add_argument(
        '--scale_factor_ATAC_normalization',
        type = int,
        default = 10**4,
        help = '[OPTIONAL] Scale factor to use when normalizing region cell probabilities.'
    )
    grp_pp.add_argument(
        '--scale_factor_RNA_normalization',
        type = int,
        default = 10**4,
        help = '[OPTIONAL] Scale factor to use when normalizing gene expression values.'
    )

    #search space arguments

    grp_search_space.add_argument(
        '--use_gene_boundaries',
        type = bool,
        default = False,
        help = '[OPTIONAL] Wether or not to stop the search space of a gene when encountering another gene. Default is False' 
    )
    grp_search_space.add_argument(
        '--upstream_min',
        type = int,
        default = 1000,
        help = "[OPTIONAL] Minimal upstream distance each gene's search space should reach. Default is 1kb."
    )
    grp_search_space.add_argument(
        '--upstream_max',
        type = int,
        default = 100000,
        help = "[OPTIONAL] Maximal upstream distance each gene's search space can reach. Default is 100kb."
    )
    grp_search_space.add_argument(
        '--downstream_min',
        type = int,
        default = 1000,
        help = "[OPTIONAL] Minimal downstream distance each gene's search space should reach. Default is 1kb."
    )
    grp_search_space.add_argument(
        '--downstream_max',
        type = int,
        default = 100000,
        help = "[OPTIONAL] Maximal downstream distance each gene's search space can reach. Default is 100kb."
    )
    grp_search_space.add_argument(
        '--extend_TSS_downstream',
        type = int,
        default = 10,
        help = "[OPTIONAL] Number of bp with which the TSS of each gene is extended downstream to get the boundary of the gene's  promoter. Default is 10."
    )
    grp_search_space.add_argument(
        '--extend_TSS_upstream',
        type = int,
        default = 10,
        help = "[OPTIONAL] Number of bp with which the TSS of each gene is extended upstream to get the boundary of the gene's  promoter. Default is 10."
    )
    grp_search_space.add_argument(
        '--remove_distal_promoters',
        type = bool,
        default = False,
        help = "[OPTIONAL] Wether or not to remove distal promoters from each gene's search space. Default is False."
    )

    #region to gene scoring arguments

    grp_scoring.add_argument(
        '-r',
        '--regions',
        type = argparse.FileType('r'),
        default = None,
        help = '[OPTIONAL] Bed file containing regions for which region to gene links. Default is to use all consensus regions.'
    )
    
    grp_scoring.add_argument(
        '--importance_scoring_method',
        type = str,
        choices = ['RF', 'ET', 'GBM'],
        default = 'RF',
        help = '[OPTIONAL] Method to use to calculate region to gene importance scores. (RF: Random Forrest, ET: Extra Trees, GBM: Gradient Boosting Machine). Default is RF.'
    )
    grp_scoring.add_argument(
        '--correlation_scoring_method',
        type = str,
        choices = ['grp_requiredSR', 'PR'],
        default = 'SR',
        help = '[OPTIONAL] Method to use to calculate region to gene correlation coefficients. (SR: Spearmanr, PR: Pearsonr). Default is SR.'
    )

    #aggregated ranking arguments

    grp_rnk.add_argument(
        '--rank_aggr_group', 
        type = str,
        help = "Key sepecifying by which to group cells to calculate mean accessbility values during rank aggregation. This key should be in the cisTopic's object meta data"
    )
    grp_rnk.add_argument(
        '--rank_aggr_distance_method',
        type = str,
        default = 'euclidean',
        choices = ['swap', 'kendalltau', 'spearman', 'spearmanr', 'pearson', 'pearsonr', 'hamming', 'levenshtein', 'winner', 'euclidean', 'winner_distance', 'asymmetrical_winner_distance'],
        help = '[OPTIONAL] Method to calculate distance between different rankings. Default is euclidean.'
    )
    grp_rnk.add_argument(
        '--rank_aggr_ceil',
        type = int,
        default = 1000,
        help = '[OPTIONAL] Maximum value of the rank aggregation score after scaling. Default is 1000.'
    )

    #UCSC export arguments

    grp_export.add_argument(
        '--rank_aggr_threshold',
        type = int,
        default = 600,
        help = '[OPTIONAL] Threshold for the scaled rank aggregation score used for visualizaitonin bigbed export. Default is 600.'
    )
    grp_export.add_argument(
        '--path_bedToBigBed',
        type = str,
        help = 'Path to location of bedToBigBed program'
    )
    grp_export.add_argument(
        '--ucsc_track_name',
        type = str,
        default = 'region_to_gene',
        help = '[OPTIONAL] Name of the interaction track for visualization in UCSC genome browser. Default is "region_to_gene".'
    )
    grp_export.add_argument(
        '--ucsc_description',
        type = str,
        default = 'interaction file for region to gene',
        help = '[OPTIONAL] Description of the interaction track for visualization in UCSC genome browser. Default is "interaction file for region to gene".'
    )
    grp_export.add_argument(
        '--cmap_neg',
        type = str,
        default = 'Reds',
        help = '[OPTIONAL] Matplotlib color map for displaying negative links. Default is Reds.'
    )
    grp_export.add_argument(
        '--cmap_pos',
        type = str,
        default = 'Blues',
        help = '[OPTIONAL] Matplotlib color map for displaying positive links. Default is Blues.'
    )
    grp_export.add_argument(
        '--vmin',
        type = int,
        default = 0,
        help = '[OPTIONAL] Value assigned to the lowest position of the color map. Default is 0.'
    )
    grp_export.add_argument(
        '--vmax',
        type = int,
        default = 1000,
        help = '[OPTIONAL] Value assigned to the highest position of the color map. Default is 1000.'
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