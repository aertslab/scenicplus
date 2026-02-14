"""Link enhancers to genes based on co-occurence of chromatin accessbility of the enhancer and gene expression.

Both linear methods (spearman or pearson correlation) and non-linear methods (random forrest or gradient boosting) are used to link enhancers to genes.

The correlation methods are used to seperate regions which are infered to have a positive influence on gene expression (i.e. positive correlation) 
and regions which are infered to have a negative influence on gene expression (i.e. negative correlation).

"""


import logging
import os
import subprocess
import sys
from typing import List, Literal, Optional, Tuple, Set, Union
import joblib
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from tqdm import tqdm

from scenicplus.utils import ( Groupby, flatten_list,
                              region_names_to_coordinates)
import pathlib

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

def _score_regions_to_single_gene(
    acc: np.ndarray,
    exp: np.ndarray,
    gene_name: str,
    region_names: Set[str],
    regressor_type: Literal["RF", "ET", "GBM", "PR", "SR"],
    regressor_kwargs: dict,
    mask_expr_dropout: bool
    ) -> Optional[Tuple[str, pd.Series]]:
    """
    Calculates region to gene importances or region to gene correlations for a single gene

    Parameters
    ----------
    acc: np.ndarray
        Numpy array containing matrix of accessibility of regions in search space.
    exp: 
        Numpy array containing expression vector.
    gene_name: str
        Name of the gene.
    region_names: List[str]
        Names of the regions.
    regressor_type: Literal["RF", "ET", "GBM", "PR", "SR"]
        Regressor type to use, must be any of "RF", "ET", "GBM", "PR", "SR".
    regressor_kwargs: dict
        Keyword arguments to pass to the regressor function.
    mask_expr_dropout: bool
        Wether or not to mask expression dropouts.
    
    Returns
    -------
    feature_importance for regression methods and correlation_coef for correlation methods
    """
    if mask_expr_dropout:
        cell_non_zero = exp != 0
        exp = exp[cell_non_zero]
        acc = acc[:, cell_non_zero]
    # Check-up for genes with 1 region only, related to issue 2
    if acc.ndim == 1:
        acc = acc.reshape(-1, 1)
    if regressor_type in SKLEARN_REGRESSOR_FACTORY.keys():
        from arboreto import core as arboreto_core

        # fit model
        fitted_model = arboreto_core.fit_model(regressor_type=regressor_type,
                                               regressor_kwargs=regressor_kwargs,
                                               tf_matrix=acc,
                                               target_gene_expression=exp)
        # get importance scores for each feature
        feature_importance = arboreto_core.to_feature_importances(regressor_type=regressor_type,
                                                                  regressor_kwargs=regressor_kwargs,
                                                                  trained_regressor=fitted_model)
        return gene_name, pd.Series(feature_importance, index=region_names)

    elif regressor_type in SCIPY_CORRELATION_FACTORY.keys():
        # define correlation method
        correlator = SCIPY_CORRELATION_FACTORY[regressor_type]

        # do correlation and get correlation coef
        correlation_result = np.array([correlator(x, exp) for x in acc.T])
        correlation_coef = correlation_result[:, 0]

        return gene_name, pd.Series(correlation_coef, index=region_names)
    else:
        raise ValueError("Unsuported regression model")

def _get_acc_idx_per_gene(
        scplus_region_names: pd.Index,
        search_space: pd.DataFrame) -> Tuple[np.ndarray, List[List[str]]]:
    region_names = search_space["Name"].to_numpy()
    gene_names = search_space["Gene"].to_numpy()
    s = np.argsort(gene_names)
    region_names = region_names[s]
    gene_names = gene_names[s]
    region_names_to_idx = pd.DataFrame(
        index = scplus_region_names,
        data = {'idx': np.arange(len(scplus_region_names))})
    unique_gene_names, gene_idx = np.unique(gene_names, return_index = True)
    region_idx_per_gene = []
    for i in range(len(gene_idx)):
        if i < len(gene_idx) - 1:
            region_idx_per_gene.append(
                region_names_to_idx.loc[region_names[gene_idx[i]:gene_idx[i+1]], 'idx'].to_list())
        else:
            region_idx_per_gene.append(
                region_names_to_idx.loc[region_names[gene_idx[i]:], 'idx'].to_list())
    return unique_gene_names, region_idx_per_gene

def _score_regions_to_genes(
        df_exp_mtx: pd.DataFrame,
        df_acc_mtx: pd.DataFrame,
        search_space: pd.DataFrame,
        mask_expr_dropout: bool,
        regressor_type: Literal["RF", "ET", "GBM", "PR", "SR"],
        regressor_kwargs: dict,
        n_cpu: int,
        temp_dir: Union[None, pathlib.Path]) -> dict:
    """
    # TODO: Add doctstrings
    """
    if len(set(df_exp_mtx.columns)) != len(df_exp_mtx.columns):
        raise ValueError("Expression matrix contains duplicate gene names")
    if len(set(df_acc_mtx.columns)) != len(df_acc_mtx.columns):
        raise ValueError("Chromatin accessibility matrix contains duplicate gene names")
    if temp_dir is not None:
        if type(temp_dir) == str:
            temp_dir = pathlib.Path(temp_dir)
        if not temp_dir.exists():
            Warning(f"{temp_dir} does not exist, creating it.")
            os.makedirs(temp_dir)
    scplus_region_names = df_acc_mtx.columns
    scplus_gene_names = df_exp_mtx.columns
    search_space = search_space[search_space['Name'].isin(scplus_region_names)]
    search_space = search_space[search_space['Gene'].isin(scplus_gene_names)]
    # Get region indeces per gene
    gene_names, acc_idx = _get_acc_idx_per_gene(
        scplus_region_names = scplus_region_names, search_space = search_space)
    EXP = df_exp_mtx[gene_names].to_numpy()
    ACC = df_acc_mtx.to_numpy()
    regions_to_genes = dict(
        joblib.Parallel(
            n_jobs = n_cpu,
            temp_folder=temp_dir)(
                joblib.delayed(_score_regions_to_single_gene)(
                    acc = ACC[:, acc_idx[idx]],
                    exp = EXP[:, idx],
                    gene_name = gene_names[idx],
                    region_names = scplus_region_names[acc_idx[idx]],
                    regressor_type = regressor_type,
                    regressor_kwargs = regressor_kwargs, 
                    mask_expr_dropout = mask_expr_dropout
                )
                for idx in tqdm(
                    range(len(gene_names)),
                    total = len(gene_names),
                    desc=f'Running using {n_cpu} cores')
                ))
    return regions_to_genes

def calculate_regions_to_genes_relationships(
        df_exp_mtx: pd.DataFrame,
        df_acc_mtx: pd.DataFrame,
        search_space: pd.DataFrame,
        temp_dir: pathlib.Path,
        mask_expr_dropout: bool = False,
        importance_scoring_method: Literal["RF", "ET", "GBM"] = 'GBM',
        importance_scoring_kwargs: dict = GBM_KWARGS,
        correlation_scoring_method: Literal["PR", "SR"] = 'SR',
        n_cpu: int = 1,
        add_distance: bool = True):
    """
    # TODO: add docstrings
    """
    # Create logger
    level = logging.INFO
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger('R2G')
    # calulcate region to gene importance
    log.info(
        f'Calculating region to gene importances, using {importance_scoring_method} method')
    region_to_gene_importances = _score_regions_to_genes(
        df_exp_mtx=df_exp_mtx,
        df_acc_mtx=df_acc_mtx,
        search_space=search_space,
        mask_expr_dropout = mask_expr_dropout,
        regressor_type = importance_scoring_method,
        regressor_kwargs = importance_scoring_kwargs,
        n_cpu = n_cpu,
        temp_dir = temp_dir)

    # calculate region to gene correlation
    log.info(
        f'Calculating region to gene correlation, using {correlation_scoring_method} method')
    region_to_gene_correlation = _score_regions_to_genes(
        df_exp_mtx=df_exp_mtx,
        df_acc_mtx=df_acc_mtx,
        search_space=search_space,
        mask_expr_dropout = mask_expr_dropout,
        regressor_type = correlation_scoring_method,
        regressor_kwargs = importance_scoring_kwargs,
        n_cpu = n_cpu,
        temp_dir = temp_dir)

    # transform dictionaries to pandas dataframe
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
    if add_distance:
        search_space_rn = search_space.rename(
            {'Name': 'region', 'Gene': 'target'}, axis=1).copy()
        result_df = result_df.merge(search_space_rn, on=['region', 'target'])
        #result_df['Distance'] = result_df['Distance'].map(lambda x: x[0])
    log.info('Done!')
    return result_df


def export_to_UCSC_interact(scplus_obj,
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
    scplus_obj: SCENICPLUS
        An instance of class scenicplus_class.SCENICPLUS containing region to gene links in .uns.
    species: str
        Species corresponding to your datassets (e.g. hsapiens)
    outfile: str
        Path to output file 
    region_to_gene_key: str =' region_to_gene'
        Key in scplus_obj.uns.keys() under which to find region to gene links.
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
        key in scplus_obj.uns.keys() under which to find eRegulons.
    
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

    if region_to_gene_key not in scplus_obj.uns.keys():
        raise Exception(
            f'key {region_to_gene_key} not found in scplus_obj.uns, first calculate region to gene relationships using function: "calculate_regions_to_genes_relationships"')

    region_to_gene_df = scplus_obj.uns[region_to_gene_key].copy()

    if subset_for_eRegulons_regions:
        if eRegulons_key not in scplus_obj.uns.keys():
            raise ValueError(
                f'key {eRegulons_key} not found in scplus_obj.uns.keys()')
        eRegulon_regions = list(set(flatten_list(
            [ereg.target_regions for ereg in scplus_obj.uns[eRegulons_key]])))
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
    if not any(['chr' in c for c in scplus_obj.region_names]):
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