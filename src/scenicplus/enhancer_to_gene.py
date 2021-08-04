#TODO: Add axtra binarization tools (otsu, ...)

import pandas as pd
import numpy  as np
import ray
import warnings
import logging
import time
import sys
import os
import subprocess
import pyranges as pr
from .utils import extend_pyranges, extend_pyranges_with_limits, reduce_pyranges_with_limits_b, calculate_distance_with_limits_join, reduce_pyranges_b, calculate_distance_join
from .utils import coord_to_region_names

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from scipy.stats import pearsonr, spearmanr

RANDOM_SEED = 666

SKLEARN_REGRESSOR_FACTORY = {
    'RF': RandomForestRegressor,
    'ET': ExtraTreesRegressor,
    'GBM': GradientBoostingRegressor
}

SCIPY_CORRELATION_FACTORY = {
    'PR': pearsonr,
    'SR': spearmanr
}

#Parameters from arboreto
# scikit-learn random forest regressor
RF_KWARGS = {
    'n_jobs': 1,
    'n_estimators': 1000,
    'max_features': 'sqrt'
}

# scikit-learn extra-trees regressor
ET_KWARGS = {
    'n_jobs': 1,
    'n_estimators': 1000,
    'max_features': 'sqrt'
}

# scikit-learn gradient boosting regressor
GBM_KWARGS = {
    'learning_rate': 0.01,
    'n_estimators': 500,
    'max_features': 0.1
}

# scikit-learn stochastic gradient boosting regressor
SGBM_KWARGS = {
    'learning_rate': 0.01,
    'n_estimators': 5000,  # can be arbitrarily large
    'max_features': 0.1,
    'subsample': 0.9
}

# Interact auto sql definition
INTERACT_AS = """table interact
"Interaction between two regions"
    (
    string chrom;      "Chromosome (or contig, scaffold, etc.). For interchromosomal, use 2 records"
    uint chromStart;   "Start position of lower region. For interchromosomal, set to chromStart of this region"
    uint chromEnd;     "End position of upper region. For interchromosomal, set to chromEnd of this region"
    string name;       "Name of item, for display.  Usually 'sourceName/targetName' or empty"
    uint score;        "Score from 0-1000."
    double value;      "Strength of interaction or other data value. Typically basis for score"
    string exp;        "Experiment name (metadata for filtering). Use . if not applicable"
    string color;      "Item color.  Specified as r,g,b or hexadecimal #RRGGBB or html color name, as in //www.w3.org/TR/css3-color/#html4."
    string sourceChrom;  "Chromosome of source region (directional) or lower region. For non-directional interchromosomal, chrom of this region."
    uint sourceStart;  "Start position source/lower/this region"
    uint sourceEnd;    "End position in chromosome of source/lower/this region"
    string sourceName;  "Identifier of source/lower/this region"
    string sourceStrand; "Orientation of source/lower/this region: + or -.  Use . if not applicable"
    string targetChrom; "Chromosome of target region (directional) or upper region. For non-directional interchromosomal, chrom of other region"
    uint targetStart;  "Start position in chromosome of target/upper/this region"
    uint targetEnd;    "End position in chromosome of target/upper/this region"
    string targetName; "Identifier of target/upper/this region"
    string targetStrand; "Orientation of target/upper/this region: + or -.  Use . if not applicable"
    )
"""

def get_search_space(pr_regions,
                     species = None,
                     assembly = None,
                     pr_annot = None, 
                     pr_chromsizes = None, 
                     use_gene_boundaries = False, 
                     upstream = [1000, 100000], 
                     downstream = [1000, 100000],
                     extend_tss=[10, 10],
                     remove_promoters = False):
    """
    Get search space surrounding genes to calculate enhancer to gene links

    Parameters
    ----------
    pr_regions: pr.PyRanges
        a :class:`pr.PyRanges` containing regions with which the extended search space should be intersected.
    species: string, optional
        Name of the species (e.g. hsapiens) on whose reference genome the search space should be calculated. This will be used to retrieve gene annotations from biomart. 
        Annotations can also be manually provided using the parameter [pr_annot]. Default: None
    assembly: string, optional
        Name of the assembly (e.g. hg38) of the reference genome on which the search space should be calculated. 
        This will be used to retrieve chromosome sizes from the UCSC genome browser.
        Chromosome sizes can also be manually provided using the parameter [pr_chromsizes]. Default: None  
    pr_annot: pr.PyRanges, optional
        A :class:`pr.PyRanges` containing gene annotation, including Chromosome, Start, End, Strand (as '+' and '-'), Gene name
        and Transcription Start Site. Default: None
    pr_chromsizes: pr.PyRanges, optional
        A :class:`pr.PyRanges` containing size of each chromosome, containing 'Chromosome', 'Start' and 'End' columns. Default: None
    use_gene_boundaries: bool, optional
        Whether to use the whole search space or stop when encountering another gene. Default: False
    upstream: List, optional
        Search space upstream. The minimum (first position) means that even if there is a gene right next to it these
        bp will be taken. The second position indicates the maximum distance. Default: [1000, 100000]
    downstream: List, optional
        Search space downstream. The minimum (first position) means that even if there is a gene right next to it these
        bp will be taken. The second position indicates the maximum distance. Default: [1000, 100000]
    extend_tss: list, optional
        Space around the TSS consider as promoter. Default: [10,10]
    remove_promoters: bool, optional
        Whether to remove promoters from the search space or not. Default: False
    Return
    ------
    pd.DataFrame
        A data frame containing regions in the search space for each gene
    """

    # Create logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('R2G')

    #parameter validation
    if ( (species is None and assembly is None and pr_annot is None and pr_chromsizes is None)
       or (species == assembly is None) and (pr_annot is None or pr_chromsizes is None)
       or (pr_annot == pr_chromsizes is None) and (species is None or assembly is None)
       or (species is not None and assembly is not None and pr_annot is not None and pr_chromsizes is not None) ):
            raise Exception('Either a name of a species and a name of an assembly or a pyranges object containing gene annotation and a pyranges object containing chromosome sizes should be provided!')
    
    extra_cols = set.difference(set(pr_regions.df.columns), set(['Chromosome', 'Start', 'End']))
    if len(extra_cols) > 0:
        Warning('The columns: "{}" will be dropped from pr_regions'.format(', '.join(extra_cols)))
        pr_regions = pr.PyRanges( pr_regions.df[ ['Chromosome', 'Start', 'End'] ] )
    
    #set region names
    pr_regions.Name = coord_to_region_names(pr_regions)

    #GET GENE ANNOTATION AND CHROMSIZES
    if species is not None and assembly is not None:
        #Download gene annotation and chromsizes
        #1. Download gene annotation from biomart
        import pybiomart as pbm
        dataset_name = '{}_gene_ensembl'.format(species)
        server = pbm.Server(host = 'http://www.ensembl.org', use_cache = False)
        mart = server['ENSEMBL_MART_ENSEMBL']
        #check wether dataset can be accessed.
        if dataset_name not in mart.list_datasets()['name'].to_numpy():
            raise Exception('{} could not be found as a dataset in biomart. Check species name or consider manually providing gene annotations!')
        else:
            log.info("Downloading gene annotation from biomart dataset: {}".format(dataset_name))
            dataset = mart[dataset_name]
            annot = dataset.query(attributes=['chromosome_name', 'start_position', 'end_position', 'strand', 'external_gene_name', 'transcription_start_site', 'transcript_biotype'])
            annot['Chromosome/scaffold name'] = 'chr' + annot['Chromosome/scaffold name'].astype(str)
            annot.columns=['Chromosome', 'Start', 'End', 'Strand', 'Gene','Transcription_Start_Site', 'Transcript_type']
            annot = annot[annot.Transcript_type == 'protein_coding']
            annot.Strand[annot.Strand == 1] = '+'
            annot.Strand[annot.Strand == -1] = '-'
            annot = pr.PyRanges(annot.dropna(axis = 0))

        #2. Download chromosome sizes from UCSC genome browser
        import requests
        target_url = 'http://hgdownload.cse.ucsc.edu/goldenPath/{asm}/bigZips/{asm}.chrom.sizes'.format(asm = assembly)
        #check wether url exists
        request = requests.get(target_url)
        if request.status_code == 200:
            log.info("Downloading chromosome sizes from: {}".format(target_url))
            chromsizes=pd.read_csv(target_url, sep='\t', header=None)
            chromsizes.columns=['Chromosome', 'End']
            chromsizes['Start']=[0]*chromsizes.shape[0]
            chromsizes=chromsizes.loc[:,['Chromosome', 'Start', 'End']]
            chromsizes=pr.PyRanges(chromsizes)
        else:
            raise Exception('The assembly {} could not be found in http://hgdownload.cse.ucsc.edu/goldenPath/. Check assembly name or consider manually providing chromosome sizes!'.format(assembly))
    else:
        #Manually provided gene annotation and chromsizes
        annot = pr_annot
        chromsizes = pr_chromsizes
    
    #Add gene width
    if annot.df['Gene'].isnull().to_numpy().any():
        annot = pr.PyRanges(annot.df.fillna(value={'Gene': 'na'}))
    annot.Gene_width = abs(annot.End - annot.Start).astype(np.int32)

    #Prepare promoter annotation
    pd_promoters = annot.df.loc[:, ['Chromosome', 'Transcription_Start_Site', 'Strand', 'Gene']]
    pd_promoters['Transcription_Start_Site'] = (
        pd_promoters.loc[:, 'Transcription_Start_Site']
    ).astype(np.int32)
    pd_promoters['End'] = (pd_promoters.loc[:, 'Transcription_Start_Site']).astype(np.int32)
    pd_promoters.columns = ['Chromosome', 'Start', 'Strand', 'Gene', 'End']
    pd_promoters = pd_promoters.loc[:, ['Chromosome', 'Start', 'End', 'Strand', 'Gene']]
    pr_promoters = pr.PyRanges(pd_promoters)
    log.info('Extending promoter annotation to {} bp upstream and {} downstream'.format( str(extend_tss[0]), str(extend_tss[1]) ))
    pr_promoters = extend_pyranges(pr_promoters, extend_tss[0], extend_tss[1])

    if use_gene_boundaries:
        log.info('Calculating gene boundaries [use_gene_boundaries = True]')
        #add chromosome limits
        chromsizes_begin_pos = chromsizes.df.copy()
        chromsizes_begin_pos['End'] = 1
        chromsizes_begin_pos['Strand'] = '+'
        chromsizes_begin_pos['Gene'] = 'Chrom_Begin'
        chromsizes_begin_neg = chromsizes_begin_pos.copy()
        chromsizes_begin_neg['Strand'] = '-'
        chromsizes_end_pos = chromsizes.df.copy()
        chromsizes_end_pos['Start'] = chromsizes_end_pos['End'] - 1
        chromsizes_end_pos['Strand'] = '+'
        chromsizes_end_pos['Gene'] = 'Chrom_End'
        chromsizes_end_neg = chromsizes_end_pos.copy()
        chromsizes_end_neg['Strand'] = '-'
        gene_bound = pr.PyRanges(
            pd.concat(
                [
                    pr_promoters.df,
                    chromsizes_begin_pos,
                    chromsizes_begin_neg,
                    chromsizes_end_pos,
                    chromsizes_end_neg
                ]
            )
        )

        # Get distance to nearest promoter (of a differrent gene)
        annot_nodup = annot[['Chromosome',
                             'Start',
                             'End',
                             'Strand',
                             'Gene',
                             'Gene_width',
                             'Gene_size_weight']].drop_duplicate_positions().copy()
        annot_nodup = pr.PyRanges( annot_nodup.df.drop_duplicates(subset="Gene", keep="first") )

        closest_promoter_upstream = annot_nodup.nearest( gene_bound, overlap=False, how='upstream') 
        closest_promoter_upstream = closest_promoter_upstream[['Chromosome', 'Start', 'End', 'Strand', 'Gene', 'Distance']]

        closest_promoter_downstream = annot_nodup.nearest(gene_bound, overlap=False, how='downstream')
        closest_promoter_downstream = closest_promoter_downstream[['Chromosome', 'Start', 'End', 'Strand', 'Gene', 'Distance']]

        # Add distance information and limit if above/below thresholds
        annot_df = annot_nodup.df
        annot_df = annot_df.set_index('Gene')
        closest_promoter_upstream_df = closest_promoter_upstream.df.set_index('Gene').Distance
        closest_promoter_upstream_df.name = 'Distance_upstream'
        annot_df = pd.concat( [annot_df, closest_promoter_upstream_df], axis=1, sort=False)
        
        closest_promoter_downstream_df = closest_promoter_downstream.df.set_index('Gene').Distance
        closest_promoter_downstream_df.name = 'Distance_downstream'
        annot_df = pd.concat([annot_df, closest_promoter_downstream_df], axis=1, sort=False).reset_index()
        
        annot_df.loc[annot_df.Distance_upstream < upstream[0], 'Distance_upstream'] = upstream[0]
        annot_df.loc[annot_df.Distance_upstream > upstream[1], 'Distance_upstream'] = upstream[1]
        annot_df.loc[annot_df.Distance_downstream < downstream[0], 'Distance_downstream'] = downstream[0]
        annot_df.loc[annot_df.Distance_downstream > downstream[1], 'Distance_downstream'] = downstream[1]
        
        annot_nodup = pr.PyRanges(annot_df.dropna(axis=0))
       
        # Extend to search space
        log.info(
            """Extending search space to: 
            \t\t\t\t\t\tA minimum of {} bp downstream of the start of the gene.
            \t\t\t\t\t\tA minimum of {} bp upstream of the end of the gene.
            \t\t\t\t\t\tA maximum of {} bp downstream or the promoter of the nearest downstream gene.
            \t\t\t\t\t\tA maximum of {} bp upstream of the end of the gene or the promoter of the nearest upstream gene""".format(str(downstream[0]), str(upstream[0]), str(downstream[1]), str(upstream[1])))

        extended_annot = extend_pyranges_with_limits(annot_nodup)
        extended_annot = extended_annot[['Chromosome',
                                         'Start',
                                         'End',
                                         'Strand',
                                         'Gene',
                                         'Gene_width',
                                         'Distance_upstream',
                                         'Distance_downstream']]
    else:
        log.info(
            """Extending search space to:
            \t\t\t\t\t\t{} bp downstream of the start of the gene.
            \t\t\t\t\t\t{} bp upstream of the start of the gene.""".format(str(downstream[1]), str(upstream[1])))
        extended_annot = extend_pyranges(annot, upstream[1], downstream[1])
        extended_annot = extended_annot[['Chromosome', 'Start', 'End', 'Strand', 'Gene', 'Gene_width', 'Gene_size_weight']]
    
    # Format search space
    extended_annot = extended_annot.drop_duplicate_positions()

    log.info('Intersecting with regions.')
    regions_per_gene = pr_regions.join(extended_annot)
    regions_per_gene.Width = abs(regions_per_gene.End - regions_per_gene.Start).astype(np.int32)
    regions_per_gene.Start = round(regions_per_gene.Start + regions_per_gene.Width / 2).astype(np.int32)
    regions_per_gene.End = (regions_per_gene.Start + 1).astype(np.int32)
    # Calculate distance
    log.info('Calculating distances from region to promoter of gene')
    if use_gene_boundaries:
        regions_per_gene = reduce_pyranges_with_limits_b(regions_per_gene)
        regions_per_gene = calculate_distance_with_limits_join(regions_per_gene)
    else:
        regions_per_gene = reduce_pyranges_b(regions_per_gene, upstream[1], downstream[1])
        regions_per_gene = calculate_distance_join(regions_per_gene)
    
    #Remove DISTAL regions overlapping with promoters
    if remove_promoters:
        log.info('Removing DISTAL regions overlapping promoters')
        regions_per_gene_overlapping_genes = regions_per_gene[regions_per_gene.Distance == 0]
        regions_per_gene_distal = regions_per_gene[regions_per_gene.Distance != 0]
        regions_per_gene_distal_wo_promoters = regions_per_gene_distal.overlap(pr_promoters, invert=True)
        regions_per_gene = pr.PyRanges(pd.concat([regions_per_gene_overlapping_genes.df, regions_per_gene_distal_wo_promoters.df]))
    
    return regions_per_gene.df[['Name', 'Gene', 'Distance']]

@ray.remote
def score_regions_to_single_gene_ray(X, y, regressor_type, regressor_kwargs) -> list:
    return score_regions_to_single_gene(X, y, regressor_type, regressor_kwargs)

def score_regions_to_single_gene(X, y, regressor_type, regressor_kwargs) -> list:
    """
    Calculates region to gene importances or region to gene correlations for a single gene
    :param X: numpy array containing matrix of accessibility of regions in search space
    :param y: numpy array containing expression vector
    :param regressor_type: type of regression/correlation analysis. 
           Available regression analysis are: 'RF' (Random Forrest regression), 'ET' (Extra Trees regression), 'GBM' (Gradient Boostin regression).
           Available correlation analysis are: 'PR' (pearson correlation), 'SR' (spearman correlation).
    :param regressor_kwargs: arguments to pass to regression function.
    :returns feature_importance for regression methods and correlation_coef for correlation methods
    """
    if regressor_type in SKLEARN_REGRESSOR_FACTORY.keys():
            from arboreto import core as arboreto_core
            #fit model
            fitted_model = arboreto_core.fit_model( regressor_type = regressor_type, 
                                                    regressor_kwargs = regressor_kwargs, 
                                                    tf_matrix = X, 
                                                    target_gene_expression = y)
            #get importance scores for each feature
            feature_importance = arboreto_core.to_feature_importances(  regressor_type = regressor_type, 
                                                                        regressor_kwargs = regressor_kwargs, 
                                                                        trained_regressor = fitted_model)
            return feature_importance

    if regressor_type in SCIPY_CORRELATION_FACTORY.keys():
        #define correlation method
        correlator = SCIPY_CORRELATION_FACTORY[regressor_type]

        #do correlation and get correlation coef and p value
        correlation_result = np.array([correlator(x, y) for x in X.T])
        correlation_coef = correlation_result[:, 0]
        
        return correlation_coef#, correlation_adj_pval

def score_regions_to_genes(imputed_acc_mtx: pd.DataFrame, 
                           expr_mtx: pd.DataFrame, 
                           search_space,
                           mask_expr_dropout = False,
                           genes = None, 
                           regressor_type = 'GBM',
                           ray_n_cpu = None,
                           regressor_kwargs = GBM_KWARGS,
                           **kwargs) -> dict:
    """
    Wrapper function for score_regions_to_single_gene and score_regions_to_single_gene_ray.
    Calculates region to gene importances or region to gene correlations for multiple genes
    :param imputed_acc_mtx: pandas data frame containing imputed accessibility data, regions as columns and cells as rows
    :param expr_mtx: pandas data frame containing expression data, genes as columns and cells as rows
    :param search space: pandas data frame containing regions (stored in column 'Name') in the search space for each gene (stored in column 'Gene')
    :param genes: list of genes for which to calculate region gene scores. Uses all genes if set to None
    :param regressor_type: type of regression/correlation analysis. 
           Available regression analysis are: 'RF' (Random Forrest regression), 'ET' (Extra Trees regression), 'GBM' (Gradient Boostin regression).
           Available correlation analysis are: 'PR' (pearson correlation), 'SR' (spearman correlation).
    :param regressor_kwargs: arguments to pass to regression function.
    :param **kwargs: additional parameters to pass to ray.init.
    :returns dictionary with genes as keys and importance score or correlation coefficient 
             as values for resp. regression based and correlation based calculations.
    """
    if genes == None:
        warnings.warn("Using all genes for which a search space and gene expression is avaible")
        genes_to_use = list(set.intersection(set(search_space['Gene']), set(expr_mtx.columns)))
    elif not all(np.isin(genes, list(search_space['Gene']))):
        warnings.warn("Not all provided genes are in search space, excluding following genes: {}".format(np.array(genes)[~np.isin(genes, list(search_space['Gene']))]))
        genes_to_use = list(set.intersection(set(search_space['Gene']), set(genes)))
    else:
        genes_to_use = genes

    if ray_n_cpu != None:
        ray.init(num_cpus=ray_n_cpu, **kwargs)
        try:
            jobs = []
            for gene in genes_to_use:
                if mask_expr_dropout:
                    expr = expr_mtx[gene]
                    cell_non_zero = expr.index[expr != 0]
                    expr = expr.loc[cell_non_zero].to_numpy()
                    acc = imputed_acc_mtx.loc[cell_non_zero, search_space.loc[search_space['Gene'] == gene, 'Name'].values].to_numpy()
                else:
                    expr = expr_mtx[gene].to_numpy()
                    acc = imputed_acc_mtx[search_space.loc[search_space['Gene'] == gene, 'Name'].values].to_numpy()
                jobs.append(score_regions_to_single_gene_ray.remote(X = acc, y = expr, regressor_type = regressor_type, regressor_kwargs = regressor_kwargs))

            regions_to_genes = ray.get(jobs)
        except Exception as e:
            print(e)
        finally:
            ray.shutdown()
        regions_to_genes = {gene: regions_to_gene for gene, regions_to_gene in zip(genes_to_use, regions_to_genes)}
    else:
        regions_to_genes = {}
        for gene in genes_to_use:
            if mask_expr_dropout:
                expr = expr_mtx[gene]
                cell_non_zero = expr.index[expr != 0]
                expr = expr.loc[cell_non_zero].to_numpy()
                acc = imputed_acc_mtx.loc[cell_non_zero, search_space.loc[search_space['Gene'] == gene, 'Name'].values].to_numpy()
            else:
                expr = expr_mtx[gene].to_numpy()
                acc = imputed_acc_mtx[search_space.loc[search_space['Gene'] == gene, 'Name'].values].to_numpy()
            regions_to_genes[gene] = score_regions_to_single_gene(X = acc, y = expr, regressor_type = regressor_type, regressor_kwargs = regressor_kwargs)
        
    return regions_to_genes

def calculate_regions_to_genes_relationships(imputed_acc_mtx: pd.DataFrame, 
                                             expr_mtx:pd.DataFrame, 
                                             search_space :pd.DataFrame, 
                                             mask_expr_dropout = False,
                                             genes = None,
                                             importance_scoring_method = 'GBM', 
                                             importance_scoring_kwargs = GBM_KWARGS,
                                             correlation_scoring_method = 'SR',
                                             ray_n_cpu = None,
                                             **kwargs) -> pd.DataFrame:
    """
    Wrapper function for score_regions_to_genes.
    Calculates region to gene relationships using regression machine learning and correlation
    :param imputed_acc_mtx: pandas data frame containing imputed accessibility data, regions as columns and cells as rows
    :param expr_mtx: pandas data frame containing expression data, genes as columns and cells as rows
    :param search space: pandas data frame containing regions (stored in column 'Name') in the search space for each gene (stored in column 'Gene')
    :param genes: list of genes for which to calculate region gene scores
    :param importance_scoring_method: method used to score region to gene importances.
                                      Available regression analysis are: 'RF' (Random Forrest regression), 'ET' (Extra Trees regression), 'GBM' (Gradient Boostin regression).
    :param importance_scoring_kwargs: arguments to pass to the importance scoring function
    :param correlation_scoring_method: method used to calculate region to gene correlations
                                       Available correlation analysis are: 'PR' (pearson correlation), 'SR' (spearman correlation).
    :param ray_n_cpu: num of cpus to use for ray multi-processing. Does not use ray when set to None
    :returns a pandas dataframe with columns: target, region, importance, rho
    """    
    # Create logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('R2G')

    #calulcate region to gene importance
    log.info('Calculating region to gene importances')
    start_time = time.time()
    region_to_gene_importances = score_regions_to_genes(imputed_acc_mtx = imputed_acc_mtx,
                                                        expr_mtx = expr_mtx,
                                                        search_space = search_space,
                                                        mask_expr_dropout = mask_expr_dropout,
                                                        genes = genes,
                                                        regressor_type = importance_scoring_method,
                                                        regressor_kwargs = importance_scoring_kwargs,
                                                        ray_n_cpu = ray_n_cpu,
                                                        **kwargs)
    log.info('Took {} seconds'.format(time.time() - start_time))

    #calculate region to gene correlation
    log.info('Calculating region to gene correlation')
    start_time = time.time()
    region_to_gene_correlation = score_regions_to_genes(imputed_acc_mtx = imputed_acc_mtx,
                                                        expr_mtx = expr_mtx,
                                                        search_space = search_space,
                                                        mask_expr_dropout = mask_expr_dropout,
                                                        genes = genes,
                                                        regressor_type = correlation_scoring_method,
                                                        ray_n_cpu = ray_n_cpu,
                                                        **kwargs)
    log.info('Took {} seconds'.format(time.time() - start_time))

    #transform dictionaries to pandas dataframe
    result_df = pd.concat   (  [ pd.DataFrame(data = {  'target': gene, 
                                                        'region': search_space.loc[search_space['Gene'] == gene, 'Name'].values,
                                                        'importance' : region_to_gene_importances[gene],
                                                        'rho': region_to_gene_correlation[gene]})
                                    for gene in region_to_gene_importances.keys()
                                ]
                            )
    return result_df


def binarize_region_to_gene_importances(region_to_gene: pd.DataFrame, method, ray_n_cpu = None, return_copy = True, **kwargs):
    if return_copy:
        region_to_gene = region_to_gene.copy()
    if method == 'BASC':
        from .BASCA import binarize
        if ray_n_cpu is None:
            res = region_to_gene.groupby('target')['importance'].apply(lambda x: binarize(vect = x, tau = 0.01, n_samples = 999).binarizedMeasurements)
            for idx, target in enumerate(res.index):
                region_to_gene.loc[region_to_gene['target'] == target, 'selected'] = res[idx]
            region_to_gene['selected'] = region_to_gene['selected'] == 1
            if return_copy:
                return region_to_gene
            else:
                return
        else:
            @ray.remote
            def _ray_binarize(vect, tau, n_samples):
                return binarize(vect, tau, n_samples).binarizedMeasurements

            ray.init(num_cpus=ray_n_cpu, **kwargs)
            try:
                #do binarization in parallel
                binarized_results = ray.get([ _ray_binarize.remote(vect = region_to_gene.loc[region_to_gene['target'] == target, 'importance'].to_numpy(), tau = 0.01, n_samples = 999) 
                                            for target in set(region_to_gene['target']) ])
            except Exception as e:
                print(e)
            finally:
                ray.shutdown()
            #put binarized results in dataframe
            for target, binarized_result in zip(set(region_to_gene['target']), binarized_results):
                region_to_gene.loc[region_to_gene['target'] == target, 'selected'] = binarized_result
            region_to_gene['selected'] = region_to_gene['selected'] == 1
            if return_copy:
                return region_to_gene
            else:
                return
            
    if method == 'mean':
        def _gt_mean(vect):
            u = np.mean(vect)
            return [v > u for v in vect]
        res = region_to_gene.groupby('target')['importance'].apply(lambda x: _gt_mean(x))
        for idx, target in enumerate(res.index):
            region_to_gene.loc[region_to_gene['target'] == target, 'selected'] = res[idx]
        if return_copy:
            return region_to_gene
        else:
            return

def export_to_UCSC_interact(region_to_gene_df, 
                            species,  
                            outfile, 
                            bigbed_outfile = None, 
                            path_bedToBigBed = None, 
                            assembly = None, 
                            ucsc_track_name = 'region_to_gene', 
                            ucsc_description = 'interaction file for region to gene', 
                            scale_per_gene = True, 
                            cmap_neg = 'Reds', 
                            cmap_pos = 'Blues'):
    """
    Exports interaction dataframe to UCSC interaction file and (optionally) UCSC bigInteract file.
    :param region_to_gene_df: interaction dataframe obtained from calculate_regions_to_genes_relationships function.
    :param species: e.g. "hsapiens", used to get gene annotation from ensembl.
    :param outfile: path to file to which to write the UCSC interaction (flat text file).
    :param bigbed_outfile (optional): path to file to which to write the UCSC bigInteract file (binary file).
    :param path_bedToBigBed: path to the bedToBigBed program.
    :param assembly: genomic assembly (e.g. hg38) used to get chromosome sizes to convert to bigBed.
    :param ucsc_track_name: name for the UCSC track.
    :param ucsc_description: description for the UCSC track.
    :param scale_per_gene: wether or not to scale region to gene importance scores (scaling is done for each gene seperatly). 
                           These scaled values are used to calculate color codes.
    :param cmap_neg: matplotlib colormap for coloring importance scores where the correlation coefficient is negative.
    :param cmap_pos: matplotlib colormap for coloring importance scores where the correlation coefficient is positive.
    """
    # Create logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('R2G')
    region_to_gene_df = region_to_gene_df.copy()
    region_to_gene_df.columns = ['Gene', 'Region', 'ImportanceScore', 'CorrelationCoef', 'Selected']
    # Get TSS annotation (end-point for links)
    import pybiomart as pbm
    dataset = pbm.Dataset(name=species+'_gene_ensembl',  host='http://www.ensembl.org')
    annot = dataset.query(attributes=['chromosome_name', 'start_position', 'end_position', 'strand', 'external_gene_name', 'transcription_start_site', 'transcript_biotype'])
    annot['Chromosome/scaffold name'] = 'chr' + annot['Chromosome/scaffold name'].astype(str)
    annot.columns=['Chromosome', 'Start', 'End', 'Strand', 'Gene','Transcription_Start_Site', 'Transcript_type']
    annot = annot[annot.Transcript_type == 'protein_coding']
    annot.Strand[annot.Strand == 1] = '+'
    annot.Strand[annot.Strand == -1] = '-'

    
    #get gene to tss mapping, take the one equal to the gene start/end location if possible otherwise take the first one
    annot['TSSeqStartEnd'] = np.logical_or(annot['Transcription_Start_Site'] == annot['Start'], annot['Transcription_Start_Site'] == annot['End'])
    gene_to_tss = annot[['Gene', 'Transcription_Start_Site']].groupby('Gene').agg(lambda x: list(map(str, x)))
    startEndEq = annot[['Gene', 'TSSeqStartEnd']].groupby('Gene').agg(lambda x: list(x))
    gene_to_tss['Transcription_Start_Site'] = [np.array(tss[0])[eq[0]][0] if sum(eq[0]) >= 1 else tss[0][0] for eq, tss in zip(startEndEq.values, gene_to_tss.values)]
    gene_to_tss.columns = ['TSS_Gene']

    #get gene to strand mapping
    gene_to_strand = annot[['Gene', 'Strand']].groupby('Gene').agg(lambda x: list(map(str, x))[0])
    
    #get gene to chromosome mapping (should be the same as the regions mapped to the gene)
    gene_to_chrom = annot[['Gene', 'Chromosome']].groupby('Gene').agg(lambda x: list(map(str, x))[0])

    #add TSS for each gene to region_to_gene_df
    region_to_gene_df = region_to_gene_df.join(gene_to_tss, on = 'Gene')

    #add strand for each gene to region_to_gene_df
    region_to_gene_df = region_to_gene_df.join(gene_to_strand, on = 'Gene')

    #add chromosome for each gene to region_to_gene_df
    region_to_gene_df = region_to_gene_df.join(gene_to_chrom, on = 'Gene')

    score_key = 'ImportanceScore'
    if scale_per_gene:
        #scale and center scores per gene
        groups = region_to_gene_df[['Gene', 'ImportanceScore']].groupby('Gene')
        mean = groups.transform(np.mean)
        std = groups.transform(np.std)
        normalized_score = [((score - mn)/sd)[0] for score, mn, sd in zip(region_to_gene_df['ImportanceScore'].values, mean.values, std.values)]
        region_to_gene_df['ImportanceScoreScaled'] = normalized_score
        score_key = 'ImportanceScoreScaled'
    
    #get chrom, chromStart, chromEnd
    chrom_split               = np.array([name.split(':') for name in region_to_gene_df['Region'].values])
    chrom                     = chrom_split[:, 0]
    chromStart_chromEnd       = chrom_split[:, 1]
    chromStart_chromEnd_split = np.array([name.split('-') for name in chromStart_chromEnd])
    chromStart                = np.array(list(map(int, chromStart_chromEnd_split[:, 0])))
    chromEnd                  = np.array(list(map(int, chromStart_chromEnd_split[:, 1])))

    #get source chrom, chromStart, chromEnd (i.e. middle of regions)
    sourceChrom = chrom
    sourceStart = np.array(list(map(int, chromStart + (chromEnd - chromStart)/2 - 1)))
    sourceEnd   = np.array(list(map(int, chromStart + (chromEnd - chromStart)/2)))

    #get target chrom, chromStart, chromEnd (i.e. TSS)
    targetChrom = region_to_gene_df['Chromosome']
    targetStart = region_to_gene_df['TSS_Gene'].values
    targetEnd   = list(map(str,np.array(list(map(int, targetStart))) + np.array([1 if strand == '+' else -1 for strand in region_to_gene_df['Strand'].values])))


    #get color
    from matplotlib import cm
    # map postive correlation values to color
    region_to_gene_df.loc[region_to_gene_df['CorrelationCoef'] >= 0 , 'color'] = [','.join(map(str, color_list)) 
                                                                                 for color_list 
                                                                                 in getattr(cm, cmap_pos)(region_to_gene_df.loc[region_to_gene_df['CorrelationCoef'] >= 0, score_key], bytes = True)[:,0:3]]
    
    # map negative correlation values to color
    region_to_gene_df.loc[region_to_gene_df['CorrelationCoef'] < 0, 'color'] = [','.join(map(str, color_list)) 
                                                                                for color_list 
                                                                                in getattr(cm, cmap_neg)(region_to_gene_df.loc[region_to_gene_df['CorrelationCoef'] < 0, score_key], bytes = True)[:,0:3]]
    #set color to gray where correlation coef equals nan
    region_to_gene_df['color'] = region_to_gene_df['color'].fillna('55,55,55')
    #get name for regions (add incremental number to gene in range of regions linked to gene)
    counter = 1
    previous_gene = region_to_gene_df['Gene'].values[0]
    names = []
    for gene in region_to_gene_df['Gene'].values:
        if gene != previous_gene:
            counter = 1
        else:
            counter +=1
        names.append(gene + '_' + str(counter))
        previous_gene = gene

    #format final interact dataframe
    df_interact = pd.DataFrame(
                                data = {
                                    'chrom':        chrom,
                                    'chromStart':   chromStart,
                                    'chromEnd':     chromEnd,
                                    'name':         names,
                                    'score':        np.repeat(0, len(region_to_gene_df)),
                                    'value':        region_to_gene_df['ImportanceScore'].values,
                                    'exp':          np.repeat('.', len(region_to_gene_df)),
                                    'color':        region_to_gene_df['color'].values,
                                    'sourceChrom':  sourceChrom,
                                    'sourceStart':  sourceStart,
                                    'sourceEnd':    sourceEnd,
                                    'sourceName':   names,
                                    'sourceStrand': np.repeat('.', len(region_to_gene_df)),
                                    'targetChrom':  targetChrom,
                                    'targetStart':  targetStart,
                                    'targetEnd':    targetEnd,
                                    'targetName':   region_to_gene_df['Gene'].values,
                                    'targetStrand': region_to_gene_df['Strand'].values
                                }
                            )
    #sort dataframe
    df_interact = df_interact.sort_values(by = ['chrom', 'chromStart'])
    #Write interact file
    with open(outfile, 'w') as f:
        f.write('track type=interact name="{}" description="{}" useScore=0 maxHeightPixels=200:100:50 visibility=full\n'.format(ucsc_track_name, ucsc_description))
        df_interact.to_csv(f, header=False, index = False, sep = '\t')

    #write bigInteract file
    if bigbed_outfile != None:
        outfolder = bigbed_outfile.rsplit('/', 1)[0]
        #write bed file without header to tmp file
        df_interact.to_csv(os.path.join(outfolder, 'interact.bed.tmp'), header=False, index = False, sep = '\t')
        
        #check if auto sql definition for interaction file exists in outfolder, otherwise create it
        if not os.path.exists(os.path.join(outfolder, 'interact.as')):
            with open(os.path.join(outfolder, 'interact.as'), 'w') as f:
                f.write(INTERACT_AS)
        #convert interact.bed.tmp to bigBed format
        # bedToBigBed -as=interact.as -type=bed5+13 region_to_gene_no_head.interact https://genome.ucsc.edu/goldenPath/help/hg38.chrom.sizes region_to_gene.inter.bb
        cmds = [
            os.path.join(path_bedToBigBed, 'bedToBigBed'),
            '-as={}'.format(os.path.join(os.path.join(outfolder, 'interact.as'))),
            '-type=bed5+13',
            os.path.join(outfolder, 'interact.bed.tmp'),
            'https://genome.ucsc.edu/goldenPath/help/' + assembly + '.chrom.sizes',
            bigbed_outfile
        ]
        p = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        if p.returncode:
            raise ValueError(
                "cmds: %s\nstderr:%s\nstdout:%s" % (" ".join(cmds), stderr, stdout)
            )
    return df_interact