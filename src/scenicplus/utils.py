import pandas as pd
import pyranges as pr
import numpy as np

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

def message_join_vector(v, sep = ', ', max_len = 4):
    uniq_v = list(set(v))
    if len(uniq_v) > max_len:
        msg = sep.join(uniq_v[0:max_len -1]) + ' ... ' + uniq_v[-1]
    else:
        msg = sep.join(uniq_v)
    return msg
