import pandas as pd
import pyranges as pr
import numpy as np
import networkx as nx
from ctxcore.genesig import Regulon
import logging
import sys

def extend_pyranges(pr_obj: pr.PyRanges,
                    upstream: int,
                    downstream: int):
    """
    A helper function to extend coordinates downstream/upstream in a pyRanges given upstream and downstream
    distances.
    """
    # Split per strand
    positive_pr = pr_obj[pr_obj.Strand == '+']
    negative_pr = pr_obj[pr_obj.Strand == '-']
    # Extend space
    if len(positive_pr) > 0:
        positive_pr.Start = (positive_pr.Start - upstream).astype(np.int32)
        positive_pr.End = (positive_pr.End + downstream).astype(np.int32)
    if len(negative_pr) > 0:
        negative_pr.Start = (negative_pr.Start - downstream).astype(np.int32)
        negative_pr.End = (negative_pr.End + upstream).astype(np.int32)
    extended_pr = pr.PyRanges(
        pd.concat([positive_pr.df, negative_pr.df], axis=0, sort=False))
    return extended_pr

def extend_pyranges_with_limits(pr_obj: pr.PyRanges):
    """
    A helper function to extend coordinates downstream/upstream in a pyRanges with Distance_upstream and
    Distance_downstream columns.
    """
    # Split per strand
    positive_pr = pr_obj[pr_obj.Strand == '+']
    negative_pr = pr_obj[pr_obj.Strand == '-']
    # Extend space
    if len(positive_pr) > 0:
        positive_pr.Start = (positive_pr.Start - positive_pr.Distance_upstream).astype(np.int32)
        positive_pr.End = (positive_pr.End + positive_pr.Distance_downstream).astype(np.int32)
    if len(negative_pr) > 0:
        negative_pr.Start = (negative_pr.Start - negative_pr.Distance_downstream).astype(np.int32)
        negative_pr.End = (negative_pr.End + negative_pr.Distance_upstream).astype(np.int32)
    extended_pr = pr.PyRanges(pd.concat([positive_pr.df, negative_pr.df], axis=0, sort=False))
    return extended_pr

def reduce_pyranges_with_limits_b(pr_obj: pr.PyRanges):
    """
    A helper function to reduce coordinates downstream/upstream in a pyRanges with Distance_upstream and
    Distance_downstream columns.
    """
    # Split per strand
    positive_pr = pr_obj[pr_obj.Strand == '+']
    negative_pr = pr_obj[pr_obj.Strand == '-']
    # Extend space
    if len(positive_pr) > 0:
        positive_pr.Start_b = (positive_pr.Start_b + positive_pr.Distance_upstream).astype(np.int32)
        positive_pr.End_b = (positive_pr.End_b - positive_pr.Distance_downstream).astype(np.int32)
    if len(negative_pr) > 0:
        negative_pr.Start_b = (negative_pr.Start_b + negative_pr.Distance_downstream).astype(np.int32)
        negative_pr.End_b = (negative_pr.End_b - negative_pr.Distance_upstream).astype(np.int32)
    extended_pr = pr.PyRanges(pd.concat([positive_pr.df, negative_pr.df], axis=0, sort=False))
    return extended_pr

def calculate_distance_with_limits_join(pr_obj: pr.PyRanges):
    """
    A helper function to calculate distances between regions and genes, returning information on what is the relative
    distance to the TSS and end of the gene.
    """
    # Split per strand
    pr_obj_df = pr_obj.df
    distance_df = pd.DataFrame(
        [
            pr_obj_df.Start_b -
            pr_obj_df.Start,
            pr_obj_df.End_b -
            pr_obj_df.Start,
            pr_obj_df.Strand
        ],
        index=['start_dist', 'end_dist', 'strand'])
    distance_df = distance_df.transpose()
    distance_df.loc[:, 'min_distance'] = abs(distance_df.loc[:, ['start_dist', 'end_dist']].transpose()).min()
    distance_df.strand[distance_df.strand == '+'] = 1
    distance_df.strand[distance_df.strand == '-'] = -1
    distance_df.loc[:, 'location'] = 0
    distance_df.loc[(distance_df.start_dist > 0) & (distance_df.end_dist > 0), 'location'] = 1
    distance_df.loc[(distance_df.start_dist < 0) & (distance_df.end_dist < 0), 'location'] = -1
    distance_df.loc[:, 'location'] = distance_df.loc[:, 'location'] * distance_df.loc[:, 'strand']
    pr_obj.Distance = distance_df.loc[:, 'location'] * distance_df.loc[:, 'min_distance'].astype(np.int32)
    pr_obj = pr_obj[['Chromosome',
                     'Start',
                     'End',
                     'Strand',
                     'Name',
                     'Gene',
                     'Gene_width',
                     'Distance',
                     'Distance_upstream',
                     'Distance_downstream']]
    return pr_obj

def reduce_pyranges_b(pr_obj: pr.PyRanges,
                      upstream: int,
                      downstream: int):
    """
    A helper function to reduce coordinates downstream/upstream in a pyRanges given upstream and downstream
    distances.
    """
    # Split per strand
    positive_pr = pr_obj[pr_obj.Strand == '+']
    negative_pr = pr_obj[pr_obj.Strand == '-']
    # Extend space
    if len(positive_pr) > 0:
        positive_pr.Start_b = (positive_pr.Start_b + upstream).astype(np.int32)
        positive_pr.End_b = (positive_pr.End_b - downstream).astype(np.int32)
    if len(negative_pr) > 0:
        negative_pr.Start_b = (negative_pr.Start_b + downstream).astype(np.int32)
        negative_pr.End_b = (negative_pr.End_b - upstream).astype(np.int32)
    extended_pr = pr.PyRanges(pd.concat([positive_pr.df, negative_pr.df], axis=0, sort=False))
    return extended_pr

def calculate_distance_join(pr_obj: pr.PyRanges):
    """
    A helper function to calculate distances between regions and genes.
    """
    # Split per strand
    pr_obj_df = pr_obj.df
    distance_df = pd.DataFrame(
        [
            pr_obj_df.Start_b - pr_obj_df.Start,
            pr_obj_df.End_b - pr_obj_df.Start,
            pr_obj_df.Strand
        ],
        index=['start_dist', 'end_dist', 'strand']
    )
    distance_df = distance_df.transpose()
    distance_df.loc[:, 'min_distance'] = abs(distance_df.loc[:, ['start_dist', 'end_dist']].transpose()).min()
    distance_df.strand[distance_df.strand == '+'] = 1
    distance_df.strand[distance_df.strand == '-'] = -1
    distance_df.loc[:, 'location'] = 0
    distance_df.loc[(distance_df.start_dist > 0) & (distance_df.end_dist > 0), 'location'] = 1
    distance_df.loc[(distance_df.start_dist < 0) & (distance_df.end_dist < 0), 'location'] = -1
    distance_df.loc[:, 'location'] = distance_df.loc[:, 'location'] * distance_df.loc[:, 'strand']
    pr_obj.Distance = distance_df.loc[:, 'location'] * distance_df.loc[:, 'min_distance'].astype(np.int32)
    pr_obj = pr_obj[['Chromosome',
                     'Start',
                     'End',
                     'Strand',
                     'Name',
                     'Gene',
                     'Gene_width',
                     'Distance']]
    return pr_obj

def coord_to_region_names(coord):
    """
    PyRanges to region names
    """
    if isinstance(coord, pr.PyRanges):
        coord = coord.as_df()
        return list(coord['Chromosome'].astype(str) + ':' + coord['Start'].astype(str) + '-' + coord['End'].astype(str))

def region_names_to_coordinates(region_names):
    chrom = pd.DataFrame([i.split(':', 1)[0] for i in region_names if ':' in i])
    coor = [i.split(':', 1)[1] for i in region_names if ':' in i]
    start = pd.DataFrame([int(i.split('-', 1)[0]) for i in coor])
    end = pd.DataFrame([int(i.split('-', 1)[1]) for i in coor])
    regiondf = pd.concat([chrom, start, end], axis=1, sort=False)
    regiondf.index = [i for i in region_names if ':' in i]
    regiondf.columns = ['Chromosome', 'Start', 'End']
    return (regiondf)

def message_join_vector(v, sep = ', ', max_len = 4):
    uniq_v = list(set(v))
    if len(uniq_v) > max_len:
        msg = sep.join(uniq_v[0:max_len -1]) + ' ... ' + uniq_v[-1]
    else:
        msg = sep.join(uniq_v)
    return msg

def eRegulons_tbl_to_nx(df_eRegulons, TF_only = False, selected_nodes = None, directed = True):
    #create adjecency matrix: https://stackoverflow.com/questions/42806398/create-adjacency-matrix-for-two-columns-in-pandas-dataframe
    A = (pd.crosstab(df_eRegulons['TF'], df_eRegulons['gene']) != 0) * 1
    if selected_nodes is None:
        idx = A.columns.intersection(A.index) if TF_only else A.columns.union(A.index)
    else:
        idx = A.columns.intersection(selected_nodes)
    A = A.reindex(index = idx, columns=idx, fill_value=0)
    return nx.from_pandas_adjacency(A, create_using = nx.DiGraph) if directed else nx.from_pandas_adjacency(A)

def eRegulons_tbl_to_genesig(df_eRegulons, mode = 'target_genes'):
    sign = lambda x: '+' if x > 0 else '-' if x < 0 else None
    genesigs = []
    for id, data in df_eRegulons.groupby(['TF', 'regulation_TF2G', 'regulation_R2G']):
        name = id[0] + ' (' + sign(id[1]) + ' / ' + sign(id[2]) + ')'
        gene2weight = data[['gene','importance_TF2G']].drop_duplicates().to_numpy() if mode == 'target_genes' else data[['region', 'aggr_rank_score']].drop_duplicates().to_numpy()
        genesigs.append(Regulon(
            name = name,
            gene2weight = dict(gene2weight),
            transcription_factor = id[0],
            gene2occurrence = []))
    return genesigs

def annotate_regions(pr_regions,
                     species,
                     extend_tss = [10, 10], 
                     exon_fraction_overlap = 0.7):
    # Create logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('R2G')
    pr_regions.Name = coord_to_region_names(pr_regions)
    pr_regions.Start = pr_regions.Start.astype(np.int32)
    pr_regions.End = pr_regions.End.astype(np.int32)

    import pybiomart as pbm
    dataset_name = '{}_gene_ensembl'.format(species)
    server = pbm.Server(host = 'http://www.ensembl.org', use_cache = False)
    mart = server['ENSEMBL_MART_ENSEMBL']
    if dataset_name not in mart.list_datasets()['name'].to_numpy():
         raise Exception('{} could not be found as a dataset in biomart. Check species name or consider manually providing gene annotations!')
    else:
        log.info("Downloading gene annotation from biomart dataset: {}".format(dataset_name))
        dataset = mart[dataset_name]
        annot = dataset.query(attributes=['chromosome_name', 
                                          'start_position', 
                                          'end_position', 
                                          'strand', 
                                          'external_gene_name', 
                                          'transcription_start_site', 
                                          'transcript_biotype',
                                          'exon_chrom_start',
                                          'exon_chrom_end',
                                          'ensembl_transcript_id',
                                          '5_utr_start',
                                          '5_utr_end',
                                          '3_utr_start',
                                          '3_utr_end'])
        annot['Chromosome/scaffold name'] = 'chr' + annot['Chromosome/scaffold name'].astype(str)
        annot.columns=['Chromosome', 
                       'Gene_start', 
                       'Gene_end', 
                       'Strand', 
                       'Gene_name', 
                       'Transcription_Start_Site', 
                       'Transcript_type', 
                       'Exon_start', 
                       'Exon_end', 
                       'Transcript_id',
                       '5_utr_start',
                       '5_utr_end',
                       '3_utr_start',
                       '3_utr_end']
        annot = annot[annot.Transcript_type == 'protein_coding']
        annot.Strand[annot.Strand == 1] = '+'
        annot.Strand[annot.Strand == -1] = '-'
    
    #get promoter locations
    pd_promoters = annot.loc[:, ['Chromosome', 'Transcription_Start_Site', 'Strand', 'Transcript_id']]
    pd_promoters['Transcription_Start_Site'] = (
        pd_promoters.loc[:, 'Transcription_Start_Site']
    ).astype(np.int32)
    pd_promoters['End'] = (pd_promoters.loc[:, 'Transcription_Start_Site']).astype(np.int32)
    pd_promoters.columns = ['Chromosome', 'Start', 'Strand', 'Transcript_id', 'End']
    pd_promoters = pd_promoters.loc[:, ['Chromosome', 'Start', 'End', 'Strand', 'Transcript_id']]
    pr_promoters = pr.PyRanges(pd_promoters.drop_duplicates())
    log.info('Extending promoter annotation to {} bp upstream and {} downstream'.format( str(extend_tss[0]), str(extend_tss[1]) ))
    pr_promoters = extend_pyranges(pr_promoters, extend_tss[0], extend_tss[1])

    #get exon locations
    pd_exons = annot.loc[:, ['Chromosome', 'Exon_start', 'Exon_end', 'Strand', 'Transcript_id']]
    pd_exons.columns = ['Chromosome', 'Start', 'End', 'Strand', 'Transcript_id']
    pr_exons = pr.PyRanges(pd_exons.drop_duplicates())

    #get gene_start_end
    pd_genes = annot.loc[:, ['Chromosome', 'Gene_start', 'Gene_end', 'Strand', 'Gene_name']]
    pd_genes.columns = ['Chromosome', 'Start', 'End', 'Strand', 'Gene_name']
    pr_genes = pr.PyRanges(pd_genes.drop_duplicates())

    #get utr
    pd_5_UTR = annot.loc[:, ['Chromosome', '5_utr_start', '5_utr_end', 'Transcript_id']].dropna()
    pd_5_UTR.columns = ['Chromosome', 'Start', 'End', 'Transcript_id']
    pr_5_UTR = pr.PyRanges(pd_5_UTR)

    pd_3_UTR = annot.loc[:, ['Chromosome', '3_utr_start', '3_utr_end', 'Transcript_id']].dropna()
    pd_3_UTR.columns = ['Chromosome', 'Start', 'End', 'Transcript_id']
    pr_3_UTR = pr.PyRanges(pd_3_UTR)

    pr_regions_ = pr_regions.join(pr_genes, how = 'left', report_overlap = True)
    pr_regions_.Start = pr_regions_.Start.astype(np.int32)
    pr_regions_.End = pr_regions_.End.astype(np.int32)
    pr_intergenic_regions = pr_regions_.subset(lambda region: region.Overlap < 0).drop(['Start_b', 'End_b', 'Gene_name', 'Overlap', 'Strand'])
    pr_genic_regions = pr_regions_.subset(lambda region: region.Overlap > 0).drop(['Start_b', 'End_b', 'Overlap', 'Gene_name', 'Strand'])
    
    pr_promoter_regions = pr_regions.join(pr_promoters, report_overlap = True).drop(['Start_b', 'End_b', 'Strand', 'Transcript_id'])
    pr_5_utr_regions = pr_regions.join(pr_5_UTR, report_overlap = True).drop(['Start_b', 'End_b', 'Transcript_id'])
    pr_3_utr_regions = pr_regions.join(pr_3_UTR, report_overlap = True).drop(['Start_b', 'End_b', 'Transcript_id'])
    pr_exon_regions = pr_genic_regions.join(pr_exons, report_overlap = True).drop(['Start_b', 'End_b', 'Strand', 'Transcript_id'])
    pr_exon_regions.Length = pr_exon_regions.End - pr_exon_regions.Start
    pr_exon_regions.Fraction = pr_exon_regions.Overlap / pr_exon_regions.Length
    pr_exon_regions = pr_exon_regions.subset(lambda region: region.Fraction > exon_fraction_overlap)

    pr_regions.annotation = None
    df_regions = pr_regions.df
    df_regions.index = df_regions['Name']
    #order matters here
    df_regions.loc[pr_intergenic_regions.Name, 'annotation'] = 'Intergenic'
    df_regions.loc[pr_genic_regions.Name, 'annotation'] = 'Intron'
    df_regions.loc[pr_exon_regions.Name, 'annotation'] = 'Exon'
    df_regions.loc[pr_3_utr_regions.Name, 'annotation'] = "3' UTR"
    df_regions.loc[pr_5_utr_regions.Name, 'annotation'] = "5' UTR"
    df_regions.loc[pr_promoter_regions.Name, 'annotation'] = "Promoter"
    
    return pr.PyRanges(df_regions.drop_duplicates())