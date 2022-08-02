"""Link enhancers to genes based on co-occurence of chromatin accessbility of the enhancer and gene expression.

Both linear methods (spearman or pearson correlation) and non-linear methods (random forrest or gradient boosting) are used to link enhancers to genes.

The correlation methods are used to seperate regions which are infered to have a positive influence on gene expression (i.e. positive correlation) 
and regions which are infered to have a negative influence on gene expression (i.e. negative correlation).

"""

import pandas as pd
import numpy as np
import ray
import logging
import time
import sys
import os
import subprocess
import pyranges as pr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from matplotlib import cm
from matplotlib.colors import Normalize
from typing import List

from .utils import extend_pyranges, extend_pyranges_with_limits, reduce_pyranges_with_limits_b
from .utils import calculate_distance_with_limits_join, reduce_pyranges_b, calculate_distance_join
from .utils import coord_to_region_names, region_names_to_coordinates, ASM_SYNONYMS, Groupby, flatten_list
from .scenicplus_class import SCENICPLUS

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

# Parameters from arboreto
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


def get_search_space(SCENICPLUS_obj: SCENICPLUS,
                     species=None,
                     assembly=None,
                     pr_annot=None,
                     pr_chromsizes=None,
                     predefined_boundaries=None,
                     use_gene_boundaries=False,
                     upstream=[1000, 150000],
                     downstream=[1000, 150000],
                     extend_tss=[10, 10],
                     remove_promoters=False,
                     biomart_host='http://www.ensembl.org',
                     inplace=True,
                     key_added='search_space',
                     implode_entries=True):
    """
    Get search space surrounding genes to calculate enhancer to gene links

    Parameters
    ----------
    SCENICPLUS_obj: SCENICPLUS
        a :class:`pr.SCENICPLUS`.
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
    predefined_boundaries: pr.PyRanges, optional
        A :class:`pr.PyRanges` containing predefined genomic domain boundaries (e.g. TAD boundaries) to use as boundaries. If 
        given, use_gene_boundaries will be ignored.
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
    biomart_host: str, optional
        Biomart host to use to download TSS annotation. Please make sure this host matches the expression data (i.e. matching gene names) otherwise a lot of genes are potentially lost.
    inplace: bool, optional
        If set to True, store results into scplus_obj, otherwise return results.
    key_added: str, optional
        Key under which to add the results under scplus.uns.
    implode_entries: bool, optional
        When a gene has multiple start/end sites it has multiple distances and gene width. 
        If this parameter is set to True these multiple entries per region and gene will be put in a list, generating a single entry.
        If this parameter is set to False these multiple entries will be kept.
    Return
    ------
    pd.DataFrame
        A data frame containing regions in the search space for each gene
    """

    # Create logger
    level = logging.INFO
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger('R2G')

    # parameter validation
    if ((species is None and assembly is None and pr_annot is None and pr_chromsizes is None)
       or (species == assembly is None) and (pr_annot is None or pr_chromsizes is None)
       or (pr_annot == pr_chromsizes is None) and (species is None or assembly is None)
       or (species is not None and assembly is not None and pr_annot is not None and pr_chromsizes is not None)):
        raise Exception('Either a name of a species and a name of an assembly or a pyranges object containing gene annotation and a pyranges object containing chromosome sizes should be provided!')

    # get regions
    pr_regions = pr.PyRanges(
        region_names_to_coordinates(SCENICPLUS_obj.region_names))

    # set region names
    pr_regions.Name = coord_to_region_names(pr_regions)

    # GET GENE ANNOTATION AND CHROMSIZES
    if species is not None and assembly is not None:
        # Download gene annotation and chromsizes
        # 1. Download gene annotation from biomart
        import pybiomart as pbm
        dataset_name = '{}_gene_ensembl'.format(species)
        server = pbm.Server(host=biomart_host, use_cache=False)
        mart = server['ENSEMBL_MART_ENSEMBL']
        # check if biomart host is correct
        dataset_display_name = getattr(
            mart.datasets[dataset_name], 'display_name')
        if not (ASM_SYNONYMS[assembly] in dataset_display_name or assembly in dataset_display_name):
            print(
                f'\u001b[31m!! The provided assembly {assembly} does not match the biomart host ({dataset_display_name}).\n Please check biomart host parameter\u001b[0m\nFor more info see: https://m.ensembl.org/info/website/archives/assembly.html')
        # check wether dataset can be accessed.
        if dataset_name not in mart.list_datasets()['name'].to_numpy():
            raise Exception(
                '{} could not be found as a dataset in biomart. Check species name or consider manually providing gene annotations!')
        else:
            log.info(
                "Downloading gene annotation from biomart dataset: {}".format(dataset_name))
            dataset = mart[dataset_name]
            if 'external_gene_name' not in dataset.attributes.keys():
                external_gene_name_query = 'hgnc_symbol'
            else:
                external_gene_name_query = 'external_gene_name'
            if 'transcription_start_site' not in dataset.attributes.keys():
                transcription_start_site_query = 'transcript_start'
            else:
                transcription_start_site_query = 'transcription_start_site'
            annot = dataset.query(attributes=['chromosome_name', 'start_position', 'end_position',
                                  'strand', external_gene_name_query, transcription_start_site_query, 'transcript_biotype'])
            annot.columns = ['Chromosome', 'Start', 'End', 'Strand',
                             'Gene', 'Transcription_Start_Site', 'Transcript_type']
            annot['Chromosome'] = 'chr' + annot['Chromosome'].astype(str)
            annot = annot[annot.Transcript_type == 'protein_coding']
            annot.Strand[annot.Strand == 1] = '+'
            annot.Strand[annot.Strand == -1] = '-'
            annot = pr.PyRanges(annot.dropna(axis=0))
            if not any(['chr' in c for c in SCENICPLUS_obj.region_names]):
                annot.Chromosome = annot.Chromosome.str.replace('chr', '')
            

        # 2. Download chromosome sizes from UCSC genome browser
        import requests
        target_url = 'http://hgdownload.cse.ucsc.edu/goldenPath/{asm}/bigZips/{asm}.chrom.sizes'.format(
            asm=assembly)
        # check wether url exists
        request = requests.get(target_url)
        if request.status_code == 200:
            log.info("Downloading chromosome sizes from: {}".format(target_url))
            chromsizes = pd.read_csv(target_url, sep='\t', header=None)
            chromsizes.columns = ['Chromosome', 'End']
            chromsizes['Start'] = [0]*chromsizes.shape[0]
            chromsizes = chromsizes.loc[:, ['Chromosome', 'Start', 'End']]
            if not any(['chr' in c for c in SCENICPLUS_obj.region_names]):
                annot.Chromosome = annot.Chromosome.str.replace('chr', '')
            chromsizes = pr.PyRanges(chromsizes)
        else:
            raise Exception(
                'The assembly {} could not be found in http://hgdownload.cse.ucsc.edu/goldenPath/. Check assembly name or consider manually providing chromosome sizes!'.format(assembly))
    else:
        # Manually provided gene annotation and chromsizes
        annot = pr_annot
        chromsizes = pr_chromsizes

    # Add gene width
    if annot.df['Gene'].isnull().to_numpy().any():
        annot = pr.PyRanges(annot.df.fillna(value={'Gene': 'na'}))
    annot.Gene_width = abs(annot.End - annot.Start).astype(np.int32)

    # Prepare promoter annotation
    pd_promoters = annot.df.loc[:, ['Chromosome',
                                    'Transcription_Start_Site', 'Strand', 'Gene']]
    pd_promoters['Transcription_Start_Site'] = (
        pd_promoters.loc[:, 'Transcription_Start_Site']
    ).astype(np.int32)
    pd_promoters['End'] = (
        pd_promoters.loc[:, 'Transcription_Start_Site']).astype(np.int32)
    pd_promoters.columns = ['Chromosome', 'Start', 'Strand', 'Gene', 'End']
    pd_promoters = pd_promoters.loc[:, [
        'Chromosome', 'Start', 'End', 'Strand', 'Gene']]
    pr_promoters = pr.PyRanges(pd_promoters)
    log.info('Extending promoter annotation to {} bp upstream and {} downstream'.format(
        str(extend_tss[0]), str(extend_tss[1])))
    pr_promoters = extend_pyranges(pr_promoters, extend_tss[0], extend_tss[1])

    if use_gene_boundaries or predefined_boundaries:
        if predefined_boundaries:
            predefined_boundaries_pos = predefined_boundaries.df.copy()
            predefined_boundaries_neg = predefined_boundaries.df.copy()
            predefined_boundaries_pos['Strand'] = '+'
            predefined_boundaries_neg['Strand'] = '-'
            predefined_boundaries = pr.PyRanges(
                pd.concat(
                    [
                        predefined_boundaries_pos,
                        predefined_boundaries_neg
                    ]
                )
            )
            space = predefined_boundaries
            log.info('Using predefined domains')
            use_gene_boundaries = False
        if use_gene_boundaries:
            space = pr_promoters
            log.info('Calculating gene boundaries')
        # add chromosome limits
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
        annot_nodup = pr.PyRanges(
            annot_nodup.df.drop_duplicates(subset="Gene", keep="first"))

        closest_promoter_upstream = annot_nodup.nearest(
            gene_bound, overlap=False, how='upstream')
        closest_promoter_upstream = closest_promoter_upstream[[
            'Chromosome', 'Start', 'End', 'Strand', 'Gene', 'Distance']]

        closest_promoter_downstream = annot_nodup.nearest(
            gene_bound, overlap=False, how='downstream')
        closest_promoter_downstream = closest_promoter_downstream[[
            'Chromosome', 'Start', 'End', 'Strand', 'Gene', 'Distance']]

        # Add distance information and limit if above/below thresholds
        annot_df = annot_nodup.df
        annot_df = annot_df.set_index('Gene')
        closest_promoter_upstream_df = closest_promoter_upstream.df.set_index(
            'Gene').Distance
        closest_promoter_upstream_df.name = 'Distance_upstream'
        annot_df = pd.concat(
            [annot_df, closest_promoter_upstream_df], axis=1, sort=False)

        closest_promoter_downstream_df = closest_promoter_downstream.df.set_index(
            'Gene').Distance
        closest_promoter_downstream_df.name = 'Distance_downstream'
        annot_df = pd.concat(
            [annot_df, closest_promoter_downstream_df], axis=1, sort=False).reset_index()

        annot_df.loc[annot_df.Distance_upstream <
                     upstream[0], 'Distance_upstream'] = upstream[0]
        annot_df.loc[annot_df.Distance_upstream >
                     upstream[1], 'Distance_upstream'] = upstream[1]
        annot_df.loc[annot_df.Distance_downstream <
                     downstream[0], 'Distance_downstream'] = downstream[0]
        annot_df.loc[annot_df.Distance_downstream >
                     downstream[1], 'Distance_downstream'] = downstream[1]

        annot_nodup = pr.PyRanges(annot_df.dropna(axis=0))

        # Extend to search space
        log.info(
            """Extending search space to: 
            \t\t\t\t\t\tA minimum of {} bp downstream of the end of the gene.
            \t\t\t\t\t\tA minimum of {} bp upstream of the start of the gene.
            \t\t\t\t\t\tA maximum of {} bp downstream of the end of the gene or the promoter of the nearest downstream gene.
            \t\t\t\t\t\tA maximum of {} bp upstream of the start of the gene or the promoter of the nearest upstream gene""".format(str(downstream[0]), str(upstream[0]), str(downstream[1]), str(upstream[1])))

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
            \t\t\t\t\t\t{} bp downstream of the end of the gene.
            \t\t\t\t\t\t{} bp upstream of the start of the gene.""".format(str(downstream[1]), str(upstream[1])))
        annot_nodup = annot[['Chromosome',
                             'Start',
                             'End',
                             'Strand',
                             'Gene',
                             'Gene_width',
                             'Gene_size_weight']].drop_duplicate_positions().copy()
        annot_nodup = pr.PyRanges(
            annot_nodup.df.drop_duplicates(subset="Gene", keep="first"))
        extended_annot = extend_pyranges(annot, upstream[1], downstream[1])
        extended_annot = extended_annot[[
            'Chromosome', 'Start', 'End', 'Strand', 'Gene', 'Gene_width', 'Gene_size_weight']]

    # Format search space
    extended_annot = extended_annot.drop_duplicate_positions()

    log.info('Intersecting with regions.')
    regions_per_gene = pr_regions.join(extended_annot)
    regions_per_gene.Width = abs(
        regions_per_gene.End - regions_per_gene.Start).astype(np.int32)
    regions_per_gene.Start = round(
        regions_per_gene.Start + regions_per_gene.Width / 2).astype(np.int32)
    regions_per_gene.End = (regions_per_gene.Start + 1).astype(np.int32)
    # Calculate distance
    log.info('Calculating distances from region to gene')
    if use_gene_boundaries or predefined_boundaries:
        regions_per_gene = reduce_pyranges_with_limits_b(regions_per_gene)
        regions_per_gene = calculate_distance_with_limits_join(
            regions_per_gene)
    else:
        regions_per_gene = reduce_pyranges_b(
            regions_per_gene, upstream[1], downstream[1])
        regions_per_gene = calculate_distance_join(regions_per_gene)

    # Remove DISTAL regions overlapping with promoters
    if remove_promoters:
        log.info('Removing DISTAL regions overlapping promoters')
        regions_per_gene_overlapping_genes = regions_per_gene[regions_per_gene.Distance == 0]
        regions_per_gene_distal = regions_per_gene[regions_per_gene.Distance != 0]
        regions_per_gene_distal_wo_promoters = regions_per_gene_distal.overlap(
            pr_promoters, invert=True)
        regions_per_gene = pr.PyRanges(pd.concat(
            [regions_per_gene_overlapping_genes.df, regions_per_gene_distal_wo_promoters.df]))

    regions_per_gene = pr.PyRanges(regions_per_gene.df.drop_duplicates())

    if implode_entries:
        log.info('Imploding multiple entries per region and gene')
        df = regions_per_gene.df
        default_columns = ['Chromosome', 'Start',
                           'End', 'Name', 'Strand', 'Gene']
        agg_dict_func1 = {column: lambda x: x.tolist()[0]
                          for column in default_columns}
        agg_dict_func2 = {column: lambda x: x.tolist()
                          for column in set(list(df.columns)) - set(default_columns)}
        agg_dict_func = {**agg_dict_func1, **agg_dict_func2}
        df = df.groupby(['Gene', 'Name'], as_index=False).agg(agg_dict_func)
        regions_per_gene = pr.PyRanges(df)
    else:
        Warning(
            'Not imploding might cause error when calculating region to gene links.')

    log.info('Done!')
    if inplace:
        SCENICPLUS_obj.uns[key_added] = regions_per_gene.df[[
            'Name', 'Gene', 'Distance']]
    else:
        return regions_per_gene.df[['Name', 'Gene', 'Distance']]


@ray.remote
def _score_regions_to_single_gene_ray(X, y, gene_name, region_names, regressor_type, regressor_kwargs) -> list:
    return _score_regions_to_single_gene(X, y, gene_name, region_names, regressor_type, regressor_kwargs)


def _score_regions_to_single_gene(X, y, gene_name, region_names, regressor_type, regressor_kwargs) -> list:
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
        # fit model
        fitted_model = arboreto_core.fit_model(regressor_type=regressor_type,
                                               regressor_kwargs=regressor_kwargs,
                                               tf_matrix=X,
                                               target_gene_expression=y)
        # get importance scores for each feature
        feature_importance = arboreto_core.to_feature_importances(regressor_type=regressor_type,
                                                                  regressor_kwargs=regressor_kwargs,
                                                                  trained_regressor=fitted_model)
        return pd.Series(feature_importance, index=region_names), gene_name

    if regressor_type in SCIPY_CORRELATION_FACTORY.keys():
        # define correlation method
        correlator = SCIPY_CORRELATION_FACTORY[regressor_type]

        # do correlation and get correlation coef and p value
        correlation_result = np.array([correlator(x, y) for x in X.T])
        correlation_coef = correlation_result[:, 0]

        # , correlation_adj_pval
        return pd.Series(correlation_coef, index=region_names), gene_name


def _score_regions_to_genes(SCENICPLUS_obj: SCENICPLUS,
                           search_space: pd.DataFrame,
                           mask_expr_dropout=False,
                           genes=None,
                           regressor_type='GBM',
                           ray_n_cpu=None,
                           regressor_kwargs=GBM_KWARGS,
                           **kwargs) -> dict:
    """
    Wrapper function for score_regions_to_single_gene and score_regions_to_single_gene_ray.
    Calculates region to gene importances or region to gene correlations for multiple genes
    :param SCENICPLUS_obj: instance of SCENICPLUS class containing expression data and chromatin accessbility data
    :param genes: list of genes for which to calculate region gene scores. Uses all genes if set to None
    :param regressor_type: type of regression/correlation analysis. 
           Available regression analysis are: 'RF' (Random Forrest regression), 'ET' (Extra Trees regression), 'GBM' (Gradient Boostin regression).
           Available correlation analysis are: 'PR' (pearson correlation), 'SR' (spearman correlation).
    :param regressor_kwargs: arguments to pass to regression function.
    :param **kwargs: additional parameters to pass to ray.init.
    :returns dictionary with genes as keys and importance score or correlation coefficient 
             as values for resp. regression based and correlation based calculations.
    """
    # Check overlaps with search space (Issue #1)
    search_space = search_space[search_space['Name'].isin(
        SCENICPLUS_obj.region_names)]
    if genes is None:
        genes_to_use = list(set.intersection(
            set(search_space['Gene']), set(SCENICPLUS_obj.gene_names)))
    elif not all(np.isin(genes, list(search_space['Gene']))):
        genes_to_use = list(set.intersection(
            set(search_space['Gene']), set(genes)))
    else:
        genes_to_use = genes
    # get expression and chromatin accessibility dataframes only once
    EXP_df = SCENICPLUS_obj.to_df(layer='EXP')
    ACC_df = SCENICPLUS_obj.to_df(layer='ACC')
    if ray_n_cpu != None:
        ray.init(num_cpus=ray_n_cpu, **kwargs)
        try:
            jobs = []
            for gene in tqdm(genes_to_use, total=len(genes_to_use), desc='initializing'):
                regions_in_search_space = search_space.loc[search_space['Gene']
                                                           == gene, 'Name'].values
                if mask_expr_dropout:
                    expr = EXP_df[gene]
                    cell_non_zero = expr.index[expr != 0]
                    expr = expr.loc[cell_non_zero].to_numpy()
                    acc = ACC_df.loc[regions_in_search_space,
                                     cell_non_zero].T.to_numpy()
                else:
                    expr = EXP_df[gene].to_numpy()
                    acc = ACC_df.loc[regions_in_search_space].T.to_numpy()
                # Check-up for genes with 1 region only, related to issue 2
                if acc.ndim == 1:
                    acc = acc.reshape(-1, 1)
                jobs.append(_score_regions_to_single_gene_ray.remote(X=acc,
                                                                    y=expr,
                                                                    gene_name=gene,
                                                                    region_names=regions_in_search_space,
                                                                    regressor_type=regressor_type,
                                                                    regressor_kwargs=regressor_kwargs))

            # add progress bar, adapted from: https://github.com/ray-project/ray/issues/8164
            def to_iterator(obj_ids):
                while obj_ids:
                    finished_ids, obj_ids = ray.wait(obj_ids)
                    for finished_id in finished_ids:
                        yield ray.get(finished_id)
            regions_to_genes = {}
            for importance, gene_name in tqdm(to_iterator(jobs),
                                              total=len(jobs),
                                              desc=f'Running using {ray_n_cpu} cores',
                                              smoothing=0.1):
                regions_to_genes[gene_name] = importance
        except Exception as e:
            print(e)
        finally:
            ray.shutdown()
    else:
        regions_to_genes = {}
        for gene in tqdm(genes_to_use, total=len(genes_to_use), desc=f'Running using a single core'):
            regions_in_search_space = search_space.loc[search_space['Gene']
                                                       == gene, 'Name'].values
            if mask_expr_dropout:
                expr = EXP_df[gene]
                cell_non_zero = expr.index[expr != 0]
                expr = expr.loc[cell_non_zero].to_numpy()
                acc = ACC_df.loc[regions_in_search_space,
                                 cell_non_zero].T.to_numpy()
            else:
                expr = EXP_df[gene].to_numpy()
                acc = ACC_df.loc[regions_in_search_space].T.to_numpy()
            # Check-up for genes with 1 region only, related to issue 2
            if acc.ndim == 1:
                acc = acc.reshape(-1, 1)
            regions_to_genes[gene], _ = _score_regions_to_single_gene(X=acc,
                                                                     y=expr,
                                                                     gene_name=gene,
                                                                     region_names=regions_in_search_space,
                                                                     regressor_type=regressor_type,
                                                                     regressor_kwargs=regressor_kwargs)

    return regions_to_genes


def calculate_regions_to_genes_relationships(SCENICPLUS_obj: SCENICPLUS,
                                             search_space_key: str = 'search_space',
                                             mask_expr_dropout: bool = False,
                                             genes: List[str] = None,
                                             importance_scoring_method: str = 'GBM',
                                             importance_scoring_kwargs: dict = GBM_KWARGS,
                                             correlation_scoring_method: str = 'SR',
                                             ray_n_cpu: int = None,
                                             add_distance: bool = True,
                                             key_added: str = 'region_to_gene',
                                             inplace: bool = True,
                                             **kwargs):
    """
    Calculates region to gene relationships using non-linear regression methods and correlation

    Parameters
    ----------
    SCENICPLUS_obj: SCENICPLUS
        instance of SCENICPLUS class containing expression data and chromatin accessbility data
    search_space_key: str = 'search_space'
        a key in SCENICPLUS_obj.uns.keys pointing to a dataframe containing the search space surounding each gene.
    mask_expr_dropout: bool = False
        Wether or not to exclude cells which have zero counts for a gene from the calculations
    genes: List[str] None
        list of genes for which to calculate region gene scores. Default is None, i.e. all genes
    importance_scoring_method: str = GBM
        method used to score region to gene importances. Available regression analysis are: 'RF' (Random Forrest regression), 'ET' (Extra Trees regression), 'GBM' (Gradient Boostin regression).
    importance_scoring_kwargs: dict = GBM_KWARGS
        arguments to pass to the importance scoring function
    correlation_scoring_method: str = SR
        method used to calculate region to gene correlations. Available correlation analysis are: 'PR' (pearson correlation), 'SR' (spearman correlation).
    ray_n_cpu: int = None
        number of cores to use for ray multi-processing. Does not use ray when set to None
    add_distance: bool = True
        Wether or not to return region to gene distances
    key_added: str = region_to_gene
        Key in SCENICPLUS_obj.uns under which to store region to gene links, only stores when inplace = True
    inplace: bool = True
        Wether or not store the region to gene links in the SCENICPLUS_obj, if False a pd.DataFrame will be returned.
    """
    # Create logger
    level = logging.INFO
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger('R2G')
    if search_space_key not in SCENICPLUS_obj.uns.keys():
        raise Exception(
            f'key {search_space_key} not found in SCENICPLUS_obj.uns, first get search space using function: "get_search_space"')

    search_space = SCENICPLUS_obj.uns[search_space_key]
    # Check overlaps with search space (Issue #1)
    search_space = search_space[search_space['Name'].isin(
        SCENICPLUS_obj.region_names)]

    # calulcate region to gene importance
    log.info(
        f'Calculating region to gene importances, using {importance_scoring_method} method')
    start_time = time.time()
    region_to_gene_importances = _score_regions_to_genes(SCENICPLUS_obj,
                                                        search_space=search_space,
                                                        mask_expr_dropout=mask_expr_dropout,
                                                        genes=genes,
                                                        regressor_type=importance_scoring_method,
                                                        regressor_kwargs=importance_scoring_kwargs,
                                                        ray_n_cpu=ray_n_cpu,
                                                        **kwargs)
    log.info('Took {} seconds'.format(time.time() - start_time))

    # calculate region to gene correlation
    log.info(
        f'Calculating region to gene correlation, using {correlation_scoring_method} method')
    start_time = time.time()
    region_to_gene_correlation = _score_regions_to_genes(SCENICPLUS_obj,
                                                        search_space=search_space,
                                                        mask_expr_dropout=mask_expr_dropout,
                                                        genes=genes,
                                                        regressor_type=correlation_scoring_method,
                                                        ray_n_cpu=ray_n_cpu,
                                                        **kwargs)
    log.info('Took {} seconds'.format(time.time() - start_time))

    # transform dictionaries to pandas dataframe
    try:
        result_df = pd.concat([pd.DataFrame(data={'target': gene,
                                                  'region': region_to_gene_importances[gene].index.to_list(),
                                                  'importance': region_to_gene_importances[gene].to_list(),
                                                  'rho': region_to_gene_correlation[gene].loc[
                                                      region_to_gene_importances[gene].index.to_list()].to_list()})
                               for gene in region_to_gene_importances.keys()
                               ]
                              )
        result_df = result_df.reset_index()
        result_df = result_df.drop('index', axis=1)
        result_df['importance_x_rho'] = result_df['rho'] * \
            result_df['importance']
        result_df['importance_x_abs_rho'] = abs(
            result_df['rho']) * result_df['importance']

    except:
        print('An error occured!')
        return region_to_gene_importances, region_to_gene_correlation

    if add_distance:
        # TODO: use consistent column names
        search_space_rn = search_space.rename(
            {'Name': 'region', 'Gene': 'target'}, axis=1).copy()
        result_df = result_df.merge(search_space_rn, on=['region', 'target'])
        #result_df['Distance'] = result_df['Distance'].map(lambda x: x[0])
    log.info('Done!')

    if inplace:
        SCENICPLUS_obj.uns[key_added] = result_df
    else:
        return result_df


def export_to_UCSC_interact(SCENICPLUS_obj: SCENICPLUS,
                            species: str,
                            outfile: str,
                            region_to_gene_key: str =' region_to_gene',
                            pbm_host:str = 'http://www.ensembl.org',
                            bigbed_outfile:str = None,
                            path_bedToBigBed: str= None,
                            assembly: str = None,
                            ucsc_track_name: str = 'region_to_gene',
                            ucsc_description: str = 'interaction file for region to gene',
                            cmap_neg: str = 'Reds',
                            cmap_pos: str = 'Greens',
                            key_for_color: str = 'importance',
                            vmin: int = 0,
                            vmax: int = 1,
                            scale_by_gene: bool = True,
                            subset_for_eRegulons_regions: bool = True,
                            eRegulons_key: str = 'eRegulons') -> pd.DataFrame:
    """
    Exports interaction dataframe to UCSC interaction file and (optionally) UCSC bigInteract file.

    Parameters
    ----------
    SCENICPLUS_obj: SCENICPLUS
        An instance of class scenicplus_class.SCENICPLUS containing region to gene links in .uns.
    species: str
        Species corresponding to your datassets (e.g. hsapiens)
    outfile: str
        Path to output file 
    region_to_gene_key: str =' region_to_gene'
        Key in SCENICPLUS_obj.uns.keys() under which to find region to gene links.
    pbm_host:str = 'http://www.ensembl.org'
        Url of biomart host relevant for your assembly.
    bigbed_outfile:str = None
        Path to which to write the bigbed output.
    path_bedToBigBed: str= None
        Path to bedToBigBed program, used to convert bed file to bigbed format. Can be downloaded from http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/bedToBigBed
    assembly: str = None
        String identifying the assembly of your dataset (e.g. hg39).
    ucsc_track_name: str = 'region_to_gene'
        Name of the exported UCSC track
    ucsc_description: str = 'interaction file for region to gene'
        Description of the exported UCSC track
    cmap_neg: str = 'Reds'
        Matplotlib colormap used to color negative region to gene links.
    cmap_pos: str = 'Greens'
        Matplotlib colormap used to color positive region to gene links.
    key_for_color: str = 'importance'
        Key pointing to column in region to gene links used to map cmap colors to.
    vmin: int = 0  
        vmin of region to gene link colors.
    vmax: int = 1
        vmax of region to gene link colors.
    scale_by_gene: bool = True
        Boolean specifying wether to scale importance scores of regions linking to the same gene from 0 to 1
    subset_for_eRegulons_regions: bool = True
        Boolean specifying wether or not to subset region to gene links for regions and genes in eRegulons.
    eRegulons_key: str = 'eRegulons'
        key in SCENICPLUS_obj.uns.keys() under which to find eRegulons.
    
    Returns
    -------
    pd.DataFrame with region to gene links formatted in the UCSC interaction format.
    """
    # Create logger
    level = logging.INFO
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger('R2G')

    if region_to_gene_key not in SCENICPLUS_obj.uns.keys():
        raise Exception(
            f'key {region_to_gene_key} not found in SCENICPLUS_obj.uns, first calculate region to gene relationships using function: "calculate_regions_to_genes_relationships"')

    region_to_gene_df = SCENICPLUS_obj.uns[region_to_gene_key].copy()

    if subset_for_eRegulons_regions:
        if eRegulons_key not in SCENICPLUS_obj.uns.keys():
            raise ValueError(
                f'key {eRegulons_key} not found in SCENICPLUS_obj.uns.keys()')
        eRegulon_regions = list(set(flatten_list(
            [ereg.target_regions for ereg in SCENICPLUS_obj.uns[eRegulons_key]])))
        region_to_gene_df.index = region_to_gene_df['region']
        region_to_gene_df = region_to_gene_df.loc[eRegulon_regions].reset_index(
            drop=True)

    # Rename columns to be in line with biomart annotation
    region_to_gene_df.rename(columns={'target': 'Gene'}, inplace=True)

    # Get TSS annotation (end-point for links)
    log.info('Downloading gene annotation from biomart, using dataset: {}'.format(
        species+'_gene_ensembl'))
    import pybiomart as pbm
    dataset = pbm.Dataset(name=species+'_gene_ensembl',  host=pbm_host)
    annot = dataset.query(attributes=['chromosome_name', 'start_position', 'end_position',
                          'strand', 'external_gene_name', 'transcription_start_site', 'transcript_biotype'])
    annot.columns = ['Chromosome', 'Start', 'End', 'Strand',
                     'Gene', 'Transcription_Start_Site', 'Transcript_type']
    annot['Chromosome'] = 'chr' + \
        annot['Chromosome'].astype(str)
    annot = annot[annot.Transcript_type == 'protein_coding']
    annot.Strand[annot.Strand == 1] = '+'
    annot.Strand[annot.Strand == -1] = '-'
    if not any(['chr' in c for c in SCENICPLUS_obj.region_names]):
        annot.Chromosome = annot.Chromosome.str.replace('chr', '')

    log.info('Formatting data ...')
    # get gene to tss mapping, take the one equal to the gene start/end location if possible otherwise take the first one
    annot['TSSeqStartEnd'] = np.logical_or(
        annot['Transcription_Start_Site'] == annot['Start'], annot['Transcription_Start_Site'] == annot['End'])
    gene_to_tss = annot[['Gene', 'Transcription_Start_Site']].groupby(
        'Gene').agg(lambda x: list(map(str, x)))
    startEndEq = annot[['Gene', 'TSSeqStartEnd']
                       ].groupby('Gene').agg(lambda x: list(x))
    gene_to_tss['Transcription_Start_Site'] = [np.array(tss[0])[eq[0]][0] if sum(
        eq[0]) >= 1 else tss[0][0] for eq, tss in zip(startEndEq.values, gene_to_tss.values)]
    gene_to_tss.columns = ['TSS_Gene']

    # get gene to strand mapping
    gene_to_strand = annot[['Gene', 'Strand']].groupby(
        'Gene').agg(lambda x: list(map(str, x))[0])

    # get gene to chromosome mapping (should be the same as the regions mapped to the gene)
    gene_to_chrom = annot[['Gene', 'Chromosome']].groupby(
        'Gene').agg(lambda x: list(map(str, x))[0])

    # add TSS for each gene to region_to_gene_df
    region_to_gene_df = region_to_gene_df.join(gene_to_tss, on='Gene')

    # add strand for each gene to region_to_gene_df
    region_to_gene_df = region_to_gene_df.join(gene_to_strand, on='Gene')

    # add chromosome for each gene to region_to_gene_df
    region_to_gene_df = region_to_gene_df.join(gene_to_chrom, on='Gene')

    # get chrom, chromStart, chromEnd
    region_to_gene_df.dropna(axis=0, how='any', inplace=True)
    arr = region_names_to_coordinates(region_to_gene_df['region']).to_numpy()
    chrom, chromStart, chromEnd = np.split(arr, 3, 1)
    chrom = chrom[:, 0]
    chromStart = chromStart[:, 0]
    chromEnd = chromEnd[:, 0]

    # get source chrom, chromStart, chromEnd (i.e. middle of regions)
    sourceChrom = chrom
    sourceStart = np.array(
        list(map(int, chromStart + (chromEnd - chromStart)/2 - 1)))
    sourceEnd = np.array(
        list(map(int, chromStart + (chromEnd - chromStart)/2)))

    # get target chrom, chromStart, chromEnd (i.e. TSS)
    targetChrom = region_to_gene_df['Chromosome']
    targetStart = region_to_gene_df['TSS_Gene'].values
    targetEnd = list(map(str, np.array(list(map(int, targetStart))) + np.array(
        [1 if strand == '+' else -1 for strand in region_to_gene_df['Strand'].values])))

    # get color
    norm = Normalize(vmin=vmin, vmax=vmax)
    if scale_by_gene:
        grouper = Groupby(
            region_to_gene_df.loc[region_to_gene_df['rho'] >= 0, 'Gene'].to_numpy())
        scores = region_to_gene_df.loc[region_to_gene_df['rho']
                                       >= 0, key_for_color].to_numpy()
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap_pos)

        def _value_to_color(scores):
            S = (scores - scores.min()) / (scores.max() - scores.min())
            return [','.join([str(x) for x in mapper.to_rgba(s, bytes=True)][0:3]) for s in S]

        colors_pos = np.zeros(len(scores), dtype='object')
        for idx in grouper.indices:
            colors_pos[idx] = _value_to_color(scores[idx])

        grouper = Groupby(
            region_to_gene_df.loc[region_to_gene_df['rho'] < 0, 'Gene'].to_numpy())
        scores = region_to_gene_df.loc[region_to_gene_df['rho']
                                       < 0, key_for_color].to_numpy()
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap_neg)

        def _value_to_color(scores):
            S = (scores - scores.min()) / (scores.max() - scores.min())
            return [','.join([str(x) for x in mapper.to_rgba(s, bytes=True)][0:3]) for s in S]

        colors_neg = np.zeros(len(scores), dtype='object')
        for idx in grouper.indices:
            colors_neg[idx] = _value_to_color(scores[idx])

    else:
        scores = region_to_gene_df.loc[region_to_gene_df['rho']
                                       >= 0, key_for_color].to_numpy()
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap_pos)
        colors_pos = [
            ','.join([str(x) for x in mapper.to_rgba(s, bytes=True)][0:3]) for s in scores]

        scores = region_to_gene_df.loc[region_to_gene_df['rho']
                                       < 0, key_for_color].to_numpy()
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap_neg)
        colors_neg = [
            ','.join([str(x) for x in mapper.to_rgba(s, bytes=True)][0:3]) for s in scores]

    region_to_gene_df.loc[region_to_gene_df['rho'] >= 0, 'color'] = colors_pos
    region_to_gene_df.loc[region_to_gene_df['rho'] < 0,  'color'] = colors_neg
    region_to_gene_df['color'] = region_to_gene_df['color'].fillna('55,55,55')

    # get name for regions (add incremental number to gene in range of regions linked to gene)
    counter = 1
    previous_gene = region_to_gene_df['Gene'].values[0]
    names = []
    for gene in region_to_gene_df['Gene'].values:
        if gene != previous_gene:
            counter = 1
        else:
            counter += 1
        names.append(gene + '_' + str(counter))
        previous_gene = gene

    # format final interact dataframe
    df_interact = pd.DataFrame(
        data={
            'chrom':        chrom,
            'chromStart':   chromStart,
            'chromEnd':     chromEnd,
            'name':         names,
            'score':        (1000*(region_to_gene_df['importance'].values - np.min(region_to_gene_df['importance'].values))/np.ptp(region_to_gene_df['importance'].values)).astype(int) ,
            'value':        region_to_gene_df['importance'].values,
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
    # sort dataframe
    df_interact = df_interact.sort_values(by=['chrom', 'chromStart'])
    # Write interact file
    log.info('Writing data to: {}'.format(outfile))
    with open(outfile, 'w') as f:
        f.write('track type=interact name="{}" description="{}" useScore=0 maxHeightPixels=200:100:50 visibility=full\n'.format(
            ucsc_track_name, ucsc_description))
        df_interact.to_csv(f, header=False, index=False, sep='\t')

    # write bigInteract file
    if bigbed_outfile != None:
        log.info('Writing data to: {}'.format(bigbed_outfile))
        outfolder = bigbed_outfile.rsplit('/', 1)[0]
        # write bed file without header to tmp file
        df_interact.to_csv(os.path.join(
            outfolder, 'interact.bed.tmp'), header=False, index=False, sep='\t')

        # check if auto sql definition for interaction file exists in outfolder, otherwise create it
        if not os.path.exists(os.path.join(outfolder, 'interact.as')):
            with open(os.path.join(outfolder, 'interact.as'), 'w') as f:
                f.write(INTERACT_AS)
        # convert interact.bed.tmp to bigBed format
        # bedToBigBed -as=interact.as -type=bed5+13 region_to_gene_no_head.interact https://genome.ucsc.edu/goldenPath/help/hg38.chrom.sizes region_to_gene.inter.bb
        cmds = [
            os.path.join(path_bedToBigBed, 'bedToBigBed'),
            '-as={}'.format(os.path.join(os.path.join(outfolder, 'interact.as'))),
            '-type=bed5+13',
            os.path.join(outfolder, 'interact.bed.tmp'),
            'https://hgdownload.cse.ucsc.edu/goldenpath/' + assembly + '/bigZips/' + assembly + '.chrom.sizes',
            bigbed_outfile
        ]
        p = subprocess.Popen(cmds, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        if p.returncode:
            raise ValueError(
                "cmds: %s\nstderr:%s\nstdout:%s" % (
                    " ".join(cmds), stderr, stdout)
            )
    return df_interact
