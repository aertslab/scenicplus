from ..scenicplus_class import SCENICPLUS
import attr
from typing import List
from collections import namedtuple
from itertools import chain
from tqdm import tqdm
from ..utils import cistarget_results_to_TF2R

flatten_list = lambda t: [item for sublist in t for item in sublist]

#HARDCODED VARIABLES
RHO_THRESHOLD = 0.03
TARGET_REGION_NAME = 'region'
TARGET_GENE_NAME = 'target'
IMPORTANCE_SCORE_NAME = 'importance'
CORRELATION_COEFFICIENT_NAME = 'rho'

REGIONS2GENES_HEADER = (TARGET_REGION_NAME, TARGET_GENE_NAME, IMPORTANCE_SCORE_NAME, CORRELATION_COEFFICIENT_NAME)

@attr.s(repr=False)
class eRegulon():
    """
    An eRegulon is a gene signature that defines the target regions and genes of a Transcription Factor (TF).
    """

    transcription_factor = attr.ib(type = str)
    regions2genes = attr.ib(type = List[namedtuple])
    #optional
    context = attr.ib(default = frozenset())
    in_leading_edge = attr.ib(type = List[bool], default = None)
    gsea_enrichment_score = attr.ib(type = float, default = None)
    gsea_pval = attr.ib(type = float, default = None)
    gsea_adj_pval = attr.ib(type = float, default = None)
    @regions2genes.validator
    def validate_regions2genes_header(self, attribute, value):
        if value is not None:
            if all([getattr(v, '_fields', None) == None for v in value]):
                Warning("{} genes2weights should be a list of named tuples".format(self.transcription_factor))
            if not all([v._fields == REGIONS2GENES_HEADER for v in value]):
                Warning("{} names of regions2genes should be: {}".format(self.transcription_factor, REGIONS2GENES_HEADER))
    
    @regions2genes.validator
    def validate_correlation_coef_same_sign(self, attribute, value):
        if value is not None:
            correlation_coefficients = [getattr(v, CORRELATION_COEFFICIENT_NAME) for v in value]
            if not (all([cc <= 0 for cc in correlation_coefficients]) or all([cc >= 0 for cc in correlation_coefficients])):
                Warning("{} correlation coefficients of regions to genes should all have the same sign".format(self.transcription_factor))
    
    @in_leading_edge.validator
    def validate_length(self, attribute, value):
        if value is not None:
            if not len(value) == self.n_target_genes:
                Warning(f"in_leading_edge ({len(value)}) should have the same length as the number of target genes ({self.n_target_genes})")

    @property
    def target_genes(self):
        """
        Return target genes of this eRegulon.
        """
        return list(set([ getattr(r2g, TARGET_GENE_NAME) for r2g in self.regions2genes ]))
    
    @property
    def target_regions(self):
        """
        Return target regions of this eRegulon.
        """
        return list(set([ getattr(r2g, TARGET_REGION_NAME) for r2g in self.regions2genes ]))
    
    @property
    def n_target_genes(self):
        """
        Return number of target genes.
        """
        return len(self.target_genes)
    
    @property
    def n_target_regions(self):
        """
        Return number of target regions.
        """
        return len(self.target_regions)

    def subset_leading_edge(self, inplace = True):
        if self.in_leading_edge is not None and self.gsea_enrichment_score is not None:
            regions2genes_subset = [
                r2g for r2g, in_le in  zip(self.regions2genes, self.in_leading_edge)
                if in_le]
            
            in_leading_edge_subset = [
                in_le for in_le in self.in_leading_edge
                if in_le
            ]
            if inplace:
                self.regions2genes = regions2genes_subset
                self.in_leading_edge = in_leading_edge_subset
            else:
                return eRegulon(
                    transcription_factor = self.transcription_factor,
                    context = self.context,
                    regions2genes = regions2genes_subset,
                    in_leading_edge = in_leading_edge_subset,
                    gsea_enrichment_score = self.gsea_enrichment_score,
                    gsea_pval = self.gsea_pval,
                    gsea_adj_pval = self.gsea_adj_pval)
        else:
            Warning('Leading edge not defined!')
    
    def __repr__(self) -> str:
        descr = f"eRegulon for TF {self.transcription_factor} in context {self.context}."
        descr += f"\n\tThis eRegulon has {self.n_target_regions} target regions and {self.n_target_genes} target genes."
        return descr

def quantile_thr(adjacencies, threshold, min_regions_per_gene,  context = frozenset()):
    c = frozenset(["{} quantile".format(threshold)]).union(context)
    df = adjacencies.groupby(by = TARGET_GENE_NAME).apply(
        lambda df_grp: df_grp.loc[df_grp[IMPORTANCE_SCORE_NAME] > df_grp[IMPORTANCE_SCORE_NAME].quantile(threshold)]
        if len(df_grp.loc[df_grp[IMPORTANCE_SCORE_NAME] > df_grp[IMPORTANCE_SCORE_NAME].quantile(threshold)]) >= min_regions_per_gene 
        else None)
    if len(df) > 0:
        yield c, df.droplevel(level = 0).reset_index(drop = True)

def top_targets(adjacencies, n, min_regions_per_gene, context = frozenset()):
    c = frozenset(["top {} gene targets".format(n)]).union(context)
    df = adjacencies.groupby(by = TARGET_GENE_NAME).apply(
        lambda df_grp: df_grp.nlargest(n, IMPORTANCE_SCORE_NAME)
        if len(df_grp.nlargest(n, IMPORTANCE_SCORE_NAME)) >= min_regions_per_gene
        else None)
    if len(df) > 0:
        yield c, df.droplevel(level = 0).reset_index(drop = True)

def top_regions(adjacencies, n, min_regions_per_gene, context = frozenset()):
    c = frozenset(["top {} region targets".format(n)]).union(context)
    df = adjacencies.groupby(by = TARGET_REGION_NAME).apply(
        lambda grp: grp.nlargest(n, IMPORTANCE_SCORE_NAME)).groupby(by = TARGET_GENE_NAME).apply(
            lambda grp: grp
            if len(grp) >= min_regions_per_gene
            else None)
    if len(df) > 0:
        yield c, df.droplevel(level = 0).reset_index(drop = True)

def create_emodules(SCENICPLUS_obj: SCENICPLUS,
                    region_to_gene_key = 'region_to_gene',
                    thresholds = (0.75, 0.90),
                    top_n_target_genes = (50, 100),
                    top_n_target_regions = (5, 10, 50),
                    min_regions_per_gene = 5,
                    rho_dichotomize=True,
                    keep_only_activating=False,
                    rho_threshold=RHO_THRESHOLD) -> List[eRegulon]:
    #check input
    if region_to_gene_key not in SCENICPLUS_obj.uns.keys():
        raise ValueError('Calculate region to gene relationships first.')
    
    def iter_thresholding(adj, context):
        yield from chain(
            chain.from_iterable(quantile_thr(adj, thr, min_regions_per_gene, context) for thr in thresholds),
            chain.from_iterable(top_targets(adj, n, min_regions_per_gene,  context) for n in top_n_target_genes),
            chain.from_iterable(top_regions(adj, n, min_regions_per_gene, context) for n in top_n_target_regions)
        )

    if rho_dichotomize:
        #split positive and negative correlation coefficients
        repressing_adj = SCENICPLUS_obj.uns[region_to_gene_key].loc[
            SCENICPLUS_obj.uns[region_to_gene_key][CORRELATION_COEFFICIENT_NAME] < -rho_threshold]
        activating_adj = SCENICPLUS_obj.uns[region_to_gene_key].loc[
            SCENICPLUS_obj.uns[region_to_gene_key][CORRELATION_COEFFICIENT_NAME] >  rho_threshold]
        r2g_iter = chain(
                iter_thresholding(repressing_adj, frozenset(['negative r2g'])),
                iter_thresholding(activating_adj, frozenset(['positive r2g'])),
            )
    else:
        #don't split
        if keep_only_activating:
            r2g_iter = iter_thresholding(SCENICPLUS_obj.uns[region_to_gene_key].loc[
                SCENICPLUS_obj.uns[region_to_gene_key][CORRELATION_COEFFICIENT_NAME] >  rho_threshold])
        else:
            r2g_iter = iter_thresholding(SCENICPLUS_obj.uns[region_to_gene_key])

    #merge all cistarget results
    ctx_results = flatten_list([[SCENICPLUS_obj.menr[x][y] for y in SCENICPLUS_obj.menr[x].keys()] for x in SCENICPLUS_obj.menr.keys()])
    tfs_to_regions_d = cistarget_results_to_TF2R(ctx_results)
    #iterate over all thresholdings and generate eRegulons
    eRegulons = []
    for context, r2g_df in r2g_iter:
        for transcription_factor in tfs_to_regions_d.keys():
            regions_enriched_for_TF_motif = tfs_to_regions_d[transcription_factor]
            r2g_df_enriched_for_TF_motif = r2g_df.loc[ [ region in regions_enriched_for_TF_motif for region in r2g_df[TARGET_REGION_NAME] ] ]
            if len(r2g_df_enriched_for_TF_motif) > 0:
                eRegulons.append(
                    eRegulon(
                        transcription_factor = transcription_factor,
                        regions2genes = list(r2g_df_enriched_for_TF_motif[list(REGIONS2GENES_HEADER)].itertuples(index = False, name = 'r2g')),
                        context = context))
    return eRegulons