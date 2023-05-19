import pyranges as pr
import logging
from scenicplus.utils import (calculate_distance_join, calculate_distance_with_limits_join, 
                              extend_pyranges,
                              extend_pyranges_with_limits,
                              reduce_pyranges_b, reduce_pyranges_with_limits_b,
                              region_names_to_coordinates, coord_to_region_names)
import pandas as pd
import numpy as np
from typing import Set, Tuple, Union
import pybiomart as pbm
import re
import requests
import time
import sys
from scenicplus.scenicplus_class import SCENICPLUS
import requests

def get_ensembl_annotation(species: str,
                           biomart_host='http://www.ensembl.org',
                            add_prefix='chr'
                           ):
    """
    Get gene annotation from ENSEMBL
    Parameters
    ----------
    species: string, optional
        Name of the species (e.g. hsapiens) on whose reference genome the search space should be calculated. This will be used to retrieve gene annotations from biomart. 
        Annotations can also be manually provided using the parameter [pr_annot]. Default: None
    biomart_host: str, optional
        Biomart host to use to download TSS annotation. Please make sure this host matches the expression data (i.e. matching gene names) otherwise a lot of genes are potentially lost.
    add_prefix: prefix to add to ENSEMBL chromosome names, default is 'chr' to match UCSC chromosome names, 
        set to for None for non-chromosome-level assemblies

    Return
    ------
    pr.PyRanges
        annotation
    """
        
    # GET GENE ANNOTATION AND CHROMSIZES
    # Download gene annotation and chromsizes
    # 1. Download gene annotation from biomart
    dataset_name = '{}_gene_ensembl'.format(species)
    server = pbm.Server(host=biomart_host, use_cache=False)
    mart = server['ENSEMBL_MART_ENSEMBL']

    # check wether dataset can be accessed.
    if dataset_name not in mart.list_datasets()['name'].to_numpy():
        raise Exception(
            '{} could not be found as a dataset in biomart. Check species name or consider manually providing gene annotations!')
    else:
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
        if add_prefix is not None:
            annot['Chromosome'] = add_prefix + annot['Chromosome'].astype(str)
        annot = annot[annot.Transcript_type == 'protein_coding']
        annot.Strand[annot.Strand == 1] = '+'
        annot.Strand[annot.Strand == -1] = '-'
        annot = pr.PyRanges(annot.dropna(axis=0))
    
    return annot

def get_ucsc_chromsizes(assembly: str):
    """
    Get chromsizes from UCSC

    Parameters
    ----------
    assembly: string, optional
        Name of the assembly (e.g. hg38) of the reference genome on which the search space should be calculated. 
        This will be used to retrieve chromosome sizes from the UCSC genome browser.
        Chromosome sizes can also be manually provided using the parameter [pr_chromsizes]. Default: None  

    Return
    ------
    pr.PyRanges
        chromsizes
    """
    
    target_url = 'http://hgdownload.cse.ucsc.edu/goldenPath/{asm}/bigZips/{asm}.chrom.sizes'.format(
            asm=assembly)
    
    # check wether url exists
    request = requests.get(target_url)
    if request.status_code == 200:
        chromsizes = pd.read_csv(target_url, sep='\t', header=None)
        chromsizes.columns = ['Chromosome', 'End']
        chromsizes['Start'] = [0]*chromsizes.shape[0]
        chromsizes = chromsizes.loc[:, ['Chromosome', 'Start', 'End']]
        chromsizes = pr.PyRanges(chromsizes)
    else:
        raise Exception(
            'The assembly {} could not be found in http://hgdownload.cse.ucsc.edu/goldenPath/. Check assembly name or consider manually providing chromosome sizes!'.format(assembly))
    
    return chromsizes


def get_search_space(SCENICPLUS_obj: SCENICPLUS,
                     annot: pr.PyRanges,
                     chromsizes: pr.PyRanges,
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

    # get regions
    pr_regions = pr.PyRanges(
        region_names_to_coordinates(SCENICPLUS_obj.region_names))

    # set region names
    pr_regions.Name = coord_to_region_names(pr_regions)

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
