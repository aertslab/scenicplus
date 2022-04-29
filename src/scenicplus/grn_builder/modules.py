"""eRegulon class stores the transcription factor together with its target regions and genes.

"""

import attr
from typing import List, Tuple
from collections import namedtuple
from itertools import chain
from tqdm import tqdm
import numpy as np
import ray

from ..scenicplus_class import SCENICPLUS
from ..utils import Groupby, coord_to_region_names, flatten_list

# HARDCODED VARIABLES
RHO_THRESHOLD = 0.03
TARGET_REGION_NAME = 'region'
TARGET_GENE_NAME = 'target'
IMPORTANCE_SCORE_NAME = 'importance'
CORRELATION_COEFFICIENT_NAME = 'rho'
SCORE_NAME_1 = "importance_x_rho"
SCORE_NAME_2 = "importance_x_abs_rho"

REGIONS2GENES_HEADER = (TARGET_REGION_NAME, TARGET_GENE_NAME, IMPORTANCE_SCORE_NAME,
                        CORRELATION_COEFFICIENT_NAME, SCORE_NAME_1, SCORE_NAME_2)


@attr.s(repr=False)
class eRegulon():
    """
    An eRegulon is a gene signature that defines the target regions and genes of a Transcription Factor (TF).

    Parameters
    ----------
    transcription_factor
        A string specifying the transcription factor of the eRegulon
    cistrome_name
        A string specifying the cistrome name
    is_extended
        A boolean specifying wether the cistromes comes from an extended annotation or not
    regions2genes
        A list of named tuples containing information on region-to-gene links 
        (region, gene, importance score, correlation coefficient)
    context
        The context in which this eRegulon was created
        (this can be different threshold, activating / repressing, ...).
        default: frozenset()
    in_leading_edge
        A list specifying which genes are in the leading edge of a gsea analysis.
        default: None
    gsea_enrichment_score
        A float containing the enrichment score of a gsea analysis.
        default: None
    gsea_pval
        A float containing the p value of a gsea analysis.
        default: None
    gsea_adj_pval
        A float containing an adjusted p value of a gsea analysis.
        default: None

    Attributes
    ----------
    target_genes
        returns a unique list of target genes.
    target_regions
        returns a unique list of target regions.
    n_target_genes
        returns an int containing the number of unique target genes.
    n_target_regions
        returns an int containing the number of unique target regions.

    See Also
    --------
    scenicplus.grn_builder.modules.create_emodules

    """

    transcription_factor = attr.ib(type=str)
    cistrome_name = attr.ib(type=str)
    is_extended = attr.ib(type=bool)
    regions2genes = attr.ib(type=List[namedtuple])
    # optional
    context = attr.ib(default=frozenset())
    in_leading_edge = attr.ib(type=List[bool], default=None)
    gsea_enrichment_score = attr.ib(type=float, default=None)
    gsea_pval = attr.ib(type=float, default=None)
    gsea_adj_pval = attr.ib(type=float, default=None)

    @regions2genes.validator
    def _validate_regions2genes_header(self, attribute, value):
        if value is not None:
            if all([getattr(v, '_fields', None) == None for v in value]):
                Warning("{} genes2weights should be a list of named tuples".format(
                    self.transcription_factor))
            if not all([v._fields == REGIONS2GENES_HEADER for v in value]):
                Warning("{} names of regions2genes should be: {}".format(
                    self.transcription_factor, REGIONS2GENES_HEADER))

    @regions2genes.validator
    def _validate_correlation_coef_same_sign(self, attribute, value):
        if value is not None:
            correlation_coefficients = [
                getattr(v, CORRELATION_COEFFICIENT_NAME) for v in value]
            if not (all([cc <= 0 for cc in correlation_coefficients]) or all([cc >= 0 for cc in correlation_coefficients])):
                Warning("{} correlation coefficients of regions to genes should all have the same sign".format(
                    self.transcription_factor))

    @in_leading_edge.validator
    def _validate_length(self, attribute, value):
        if value is not None:
            if not len(value) == self.n_target_genes:
                Warning(
                    f"in_leading_edge ({len(value)}) should have the same length as the number of target genes ({self.n_target_genes})")

    @property
    def target_genes(self):
        """
        Return target genes of this eRegulon.
        """
        return list(set([getattr(r2g, TARGET_GENE_NAME) for r2g in self.regions2genes]))

    @property
    def target_regions(self):
        """
        Return target regions of this eRegulon.
        """
        return list(set([getattr(r2g, TARGET_REGION_NAME) for r2g in self.regions2genes]))

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

    def subset_leading_edge(self, inplace=True):
        """
        Subset eReglon on leading edge.

        Parameters
        ---------
        inplace
            if set to True update eRegulon else return new eRegulon after subset.
        """
        if self.in_leading_edge is not None and self.gsea_enrichment_score is not None:
            regions2genes_subset = [
                r2g for r2g, in_le in zip(self.regions2genes, self.in_leading_edge)
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
                    transcription_factor=self.transcription_factor,
                    cistrome_name=self.cistrome_name,
                    is_extended=self.is_extended,
                    context=self.context,
                    regions2genes=regions2genes_subset,
                    in_leading_edge=in_leading_edge_subset,
                    gsea_enrichment_score=self.gsea_enrichment_score,
                    gsea_pval=self.gsea_pval,
                    gsea_adj_pval=self.gsea_adj_pval)
        else:
            Warning('Leading edge not defined!')

    def __repr__(self) -> str:
        descr = f"eRegulon for TF {self.transcription_factor} in context {self.context}."
        descr += f"\n\tThis eRegulon has {self.n_target_regions} target regions and {self.n_target_genes} target genes."
        return descr


def _quantile_thr(adjacencies,
                 grouped,
                 threshold,
                 min_regions_per_gene,
                 context=frozenset(),
                 order_regions_to_genes_by='importance',
                 ray_n_cpu=None):
    """
    A helper function to binarize region-to-gene links using based on quantiles of importance.

    Parameters
    ----------
    adjacencies
        A :class:`pandas.DataFrame` containing region-to-gene importance scores.
    grouped
        A :class:`~scenicplus.utils.Groupby` containing group information on the `adjacencies` grouped on target region.
    threshold
        float specifying the quantile on the importance score of region-to-gene links to retain.
    min_regions_per_gene
        An int specifying a lower limit on regions per gene (after binarization) to consider for further analysis.
    context
        A frozenset containing contexts to add to the binarized results.
        default: frozenset()
    ray_n_cpu
        An int specifying the number of parallel cores to use

    Yields
    -------
    context, :class:`pandas.DataFrame` with binarized region-to-gene links.
    """
    def _qt(x):
        # function to return threshold_quantile from a vector
        return np.quantile(x, threshold)

    def _gt(x):
        # function to check minimum regions requirement
        if sum(x) >= min_regions_per_gene:
            return x
        else:
            return np.repeat(False, len(x))

    c = frozenset(["{} quantile".format(threshold)]).union(context)
    # grouped = Groupby(adjacencies[TARGET_GENE_NAME].to_numpy()) #this could be moved out of the function
    importances = adjacencies[order_regions_to_genes_by].to_numpy()

    # get quantiles and threshold
    thresholds = grouped.apply(_qt, importances, True, ray_n_cpu=ray_n_cpu)
    passing = importances > thresholds

    if min_regions_per_gene > 0:
        # check min regions per gene
        passing = grouped.apply(
            _gt, passing, True, ray_n_cpu=ray_n_cpu).astype(bool)

    if sum(passing) > 0:
        df = adjacencies.loc[passing]
        df.index = df[TARGET_REGION_NAME]
        yield c, df


def _top_targets(adjacencies,
                grouped,
                n,
                min_regions_per_gene,
                context=frozenset(),
                order_regions_to_genes_by='importance',
                ray_n_cpu=None):
    """
    A helper function to binarize region-to-gene links using based on the top region-to-gene links per gene.

    Parameters
    ----------
    adjacencies
        A :class:`pandas.DataFrame` containing region-to-gene importance scores.
    grouped
        A :class:`~scenicplus.utils.Groupby` containing group information on the `adjacencies` grouped on target gene.
    n
        integer specifying the top n number of region-to-gene links to retain.
    min_regions_per_gene
        An int specifying a lower limit on regions per gene (after binarization) to consider for further analysis.
    context
        A frozenset containing contexts to add to the binarized results.
        default: frozenset()
    ray_n_cpu
        An int specifying the number of parallel cores to use

    Yields
    -------
    context, :class:`pandas.DataFrame` with binarized region-to-gene links.
    """
    def _top(x):
        # function to get top n entries.
        if len(x) >= n:
            return min(np.sort(x)[-n:])
        else:
            return min(x)

    def _gt(x):
        # function to check minimum regions requirement
        if sum(x) >= min_regions_per_gene:
            return x
        else:
            return np.repeat(False, len(x))

    c = frozenset(
        ["Top {} region-to-gene links per gene".format(n)]).union(context)
    # grouped = Groupby(adjacencies[TARGET_GENE_NAME].to_numpy()) #this could be moved out of the function
    importances = adjacencies[order_regions_to_genes_by].to_numpy()

    # get top n threshold
    thresholds = grouped.apply(_top, importances, True, ray_n_cpu=ray_n_cpu)
    passing = importances >= thresholds

    if min_regions_per_gene > 0:
        # check min regions per gene
        passing = grouped.apply(
            _gt, passing, True, ray_n_cpu=ray_n_cpu).astype(bool)

    if sum(passing) > 0:
        df = adjacencies.loc[passing]
        df.index = df[TARGET_REGION_NAME]
        yield c, df


def _top_regions(adjacencies,
                grouped,
                n,
                min_regions_per_gene,
                context=frozenset(),
                order_regions_to_genes_by='importance',
                ray_n_cpu=None):
    """
    A helper function to binarize region-to-gene links using based on the top region-to-gene links per region.

    Parameters
    ----------
    adjacencies
        A :class:`pandas.DataFrame` containing region-to-gene importance scores.
    grouped
        A :class:`~scenicplus.utils.Groupby` containing group information on the `adjacencies` grouped on target region.
    n
        integer specifying the top n number of region-to-gene links to retain.
    min_regions_per_gene
        An int specifying a lower limit on regions per gene (after binarization) to consider for further analysis.
    context
        A frozenset containing contexts to add to the binarized results.
        default: frozenset()
    ray_n_cpu
        An int specifying the number of parallel cores to use

    Yields
    -------
    context, :class:`pandas.DataFrame` with binarized region-to-gene links.
    """
    def _top(x):
        # function to get top n entries.
        if len(x) >= n:
            return min(np.sort(x)[-n:])
        else:
            return min(x)

    def _gt(x):
        # function to check minimum regions requirement
        if sum(x) >= min_regions_per_gene:
            return x
        else:
            return np.repeat(False, len(x))

    c = frozenset(
        ["Per region top {} region-to-gene links per gene".format(n)]).union(context)

    # grouped = Groupby(adjacencies[TARGET_REGION_NAME].to_numpy()) #this could be moved out of the function
    importances = adjacencies[order_regions_to_genes_by].to_numpy()

    # get top n threshold
    thresholds = grouped.apply(_top, importances, True, ray_n_cpu=ray_n_cpu)
    passing = importances >= thresholds

    df = adjacencies.loc[passing]

    if min_regions_per_gene > 0:
        # check minimum target gene requirement
        grouped = Groupby(df[TARGET_GENE_NAME].to_numpy())
        passing = grouped.apply(
            _gt, passing[passing], True, ray_n_cpu=ray_n_cpu).astype(bool)
        df = df.loc[passing]
        df.index = df[TARGET_REGION_NAME]

    if len(df) > 0:
        yield c, df


def _binarize_BASC(adjacencies,
                  grouped,
                  min_regions_per_gene,
                  context=frozenset(),
                  order_regions_to_genes_by='importance',
                  ray_n_cpu=None):
    """
    A helper function to binarize region-to-gene links using BASC
    For more information see: Hopfensitz M, et al. Multiscale binarization of gene expression data for reconstructing Boolean networks. 
                              IEEE/ACM Trans Comput Biol Bioinform. 2012;9(2):487-98.

    Parameters
    ----------
    adjacencies
        A :class:`pandas.DataFrame` containing region-to-gene importance scores.
    grouped
        A :class:`~scenicplus.utils.Groupby` containing group information on the `adjacencies` grouped on target gene.
    min_regions_per_gene
        An int specifying a lower limit on regions per gene (after binarization) to consider for further analysis.
    context
        A frozenset containing contexts to add to the binarized results.
        default: frozenset()
    ray_n_cpu
        An int specifying the number of parallel cores to use

    Yields
    -------
    context, :class:`pandas.DataFrame` with binarized region-to-gene links.
    """
    from ..BASCA import binarize

    def _binarize_basc(x):
        if len(x) > 2 and not all(x == x[0]):
            try:
                threshold, _ = binarize(x, calc_p=False)
                return threshold
            except:
                return 0
        else:
            # can only binarize when array is > 2
            return 0

    def _gt(x):
        # function to check minimum regions requirement
        if sum(x) >= min_regions_per_gene:
            return x
        else:
            return np.repeat(False, len(x))

    c = frozenset(["BASC binarized"]).union(context)

    importances = adjacencies[order_regions_to_genes_by].to_numpy()

    # get BASC thresholds
    thresholds = grouped.apply(
        _binarize_basc, importances, True, ray_n_cpu=ray_n_cpu)
    passing = importances > thresholds

    if min_regions_per_gene > 0:
        # check min regions per gene
        passing = grouped.apply(
            _gt, passing, True, ray_n_cpu=ray_n_cpu).astype(bool)

    if sum(passing) > 0:
        df = adjacencies.loc[passing]
        df.index = df[TARGET_REGION_NAME]
        yield c, df


def create_emodules(SCENICPLUS_obj: SCENICPLUS,
                    region_to_gene_key: str = 'region_to_gene',
                    cistromes_key: str = 'Unfiltered',
                    order_regions_to_genes_by: str = 'importance',
                    quantiles: tuple = (0.85, 0.90),
                    top_n_regionTogenes_per_gene: tuple = (5, 10, 15),
                    top_n_regionTogenes_per_region: tuple = (),
                    binarize_using_basc: bool = False,
                    min_regions_per_gene: int = 0,
                    rho_dichotomize: bool = True,
                    keep_only_activating: bool = False,
                    rho_threshold: float = RHO_THRESHOLD,
                    keep_extended_motif_annot: bool = False,
                    disable_tqdm=False,
                    ray_n_cpu=None,
                    **kwargs) -> Tuple[List[str], List[eRegulon]]:
    """
    A function to create e_modules of :class: ~scenicplus.grn_builder.modules.eRegulon. 
    This function will binarize region-to-gene links using various parameters and couples these links to a TF
    based on infromation in the motif enrichment (menr) field of the :attr:`SCENICPLUS_obj`.

    Parameters
    ----------
    SCENICPLUS_obj
        An instance of :class: `~scenicplus.scenicplus_class.SCENICPLUS`, containing motif enrichment data under the slot .menr.
    region_to_gene_key
        A key specifying under which to find region-to-gene links in the .uns slot of `SCENICPLUS_obj`. 
        Default: "region_to_gene"
    cistromes_key
        A key specifying which cistromes to use in .uns['Cistromes'] slot of `SCENICPLUS_obj`
    quantiles
        A tuple specifying the quantiles used to binarize region-to-gene links
        Default: (0.85, 0.90)
    top_n_regionTogenes_per_gene
        A tuple specifying the top n region-to-gene links to take PER GENE in order to binarize region-to-gene links.
        Default: (5, 10, 15)
    top_n_regionTogenes_per_region
        A tuple specifying the top n region-to-gene links to take PER REGION in order to binarize region-to-gene links.
        Default: ()
    binarize_using_basc:
        A boolean specifying wether or not to binarize region-to-gene links using BASC. Hopfensitz M, et al.
    min_regions_per_gene:
        An integer specifying a lower limit on regions per gene (after binarization) to consider for further analysis.
        Default: 0
    rho_dichotomize:
        A boolean specifying wether or not to split region-to-gene links based on postive/negative correlation coefficients.
        default: True
    keep_only_activating:
        A boolean specifying wether or not to only retain region-to-gene links with a positive correlation coefficient.
        default: False
    rho_threshold:
        A floating point number specifying from which absolute value to consider a correlation coefficient positive or negative.
        default: 0.03
    keep_extended_motif_annot
        A boolean specifying wether or not keep extended motif annotations for further analysis.
        default: False
    ray_n_cpu
        An integer specifying the number of parallel cores to use to binarize region to gene links
        default: None (i.e. run using single core)
    **kwargs
        Extra keyword arguments passed to ray.init function

    Returns
    -------
    A set of relevant TFs, A list of :class: `~scenicplus.grn_builder.modules.eRegulon.`

    References
    ----------
    Hopfensitz M, et al. Multiscale binarization of gene expression data for reconstructing Boolean networks. IEEE/ACM Trans Comput Biol Bioinform. 2012;9(2):487-98.
    """
    # check input
    if region_to_gene_key not in SCENICPLUS_obj.uns.keys():
        raise ValueError('Calculate region to gene relationships first.')

    if 'Cistromes' not in SCENICPLUS_obj.uns.keys():
        raise ValueError(
            'Cistromes are undefined, first define cistromes in the slot .uns["Cistromes"]')

    def iter_thresholding(adj, context):
        grouped_adj_by_gene = Groupby(adj[TARGET_GENE_NAME].to_numpy())
        grouped_adj_by_region = Groupby(adj[TARGET_REGION_NAME].to_numpy())
        yield from chain(
            chain.from_iterable(_quantile_thr(adjacencies=adj,
                                             grouped=grouped_adj_by_gene,
                                             threshold=thr,
                                             min_regions_per_gene=min_regions_per_gene,
                                             context=context,
                                             order_regions_to_genes_by=order_regions_to_genes_by) for thr in quantiles),

            chain.from_iterable(_top_targets(adjacencies=adj,
                                            grouped=grouped_adj_by_gene,
                                            n=n,
                                            min_regions_per_gene=min_regions_per_gene,
                                            context=context,
                                            order_regions_to_genes_by=order_regions_to_genes_by) for n in top_n_regionTogenes_per_gene),

            chain.from_iterable(_top_regions(adjacencies=adj,
                                            grouped=grouped_adj_by_region,
                                            n=n,
                                            min_regions_per_gene=min_regions_per_gene,
                                            context=context,
                                            order_regions_to_genes_by=order_regions_to_genes_by) for n in top_n_regionTogenes_per_region),
            _binarize_BASC(adjacencies=adj,
                          grouped=grouped_adj_by_gene,
                          min_regions_per_gene=min_regions_per_gene,
                          context=context,
                          order_regions_to_genes_by=order_regions_to_genes_by) if binarize_using_basc else []
        )

    if rho_dichotomize:
        # split positive and negative correlation coefficients
        repressing_adj = SCENICPLUS_obj.uns[region_to_gene_key].loc[
            SCENICPLUS_obj.uns[region_to_gene_key][CORRELATION_COEFFICIENT_NAME] < -rho_threshold]
        activating_adj = SCENICPLUS_obj.uns[region_to_gene_key].loc[
            SCENICPLUS_obj.uns[region_to_gene_key][CORRELATION_COEFFICIENT_NAME] > rho_threshold]
        r2g_iter = chain(
            iter_thresholding(repressing_adj, frozenset(['negative r2g'])),
            iter_thresholding(activating_adj, frozenset(['positive r2g'])),
        )
    else:
        # don't split
        if keep_only_activating:
            r2g_iter = iter_thresholding(SCENICPLUS_obj.uns[region_to_gene_key].loc[
                SCENICPLUS_obj.uns[region_to_gene_key][CORRELATION_COEFFICIENT_NAME] > rho_threshold])
        else:
            r2g_iter = iter_thresholding(
                SCENICPLUS_obj.uns[region_to_gene_key])

    # get cistromes
    cistrome_to_regions_d = SCENICPLUS_obj.uns['Cistromes'][cistromes_key]

    # iterate over all thresholdings and generate eRegulons
    n_params = sum([len(quantiles) if not type(quantiles) == float else 1,
                    len(top_n_regionTogenes_per_gene) if not type(
                        top_n_regionTogenes_per_gene) == int else 1,
                    len(top_n_regionTogenes_per_region) if not type(top_n_regionTogenes_per_region) == int else 1])
    total_iter = (2 * (n_params + (binarize_using_basc * 1))
                  ) if rho_dichotomize else (n_params + (binarize_using_basc * 1))
    relevant_tfs = []
    eRegulons = []

    if ray_n_cpu is not None:
        # Init ray here so it only has to be inited once, instead of with every call of grouped.apply.
        # This to prevent the extra overhead of calling ray.init, which is not insignifican.
        ray.init(num_cpus=ray_n_cpu, **kwargs)

    try:
        for context, r2g_df in tqdm(r2g_iter, total=total_iter, disable=disable_tqdm):
            for cistrome_name in tqdm(cistrome_to_regions_d.keys(),
                                      total=len(cistrome_to_regions_d.keys()),
                                      desc=f"\u001b[32;1mProcessing:\u001b[0m {', '.join(context)}",
                                      leave=False,
                                      disable=disable_tqdm):
                if (not keep_extended_motif_annot) and 'extended' in cistrome_name:
                    # skip
                    continue
                regions_enriched_for_TF_motif = coord_to_region_names(
                    cistrome_to_regions_d[cistrome_name])
                r2g_df_enriched_for_TF_motif = r2g_df.loc[list(set(
                    regions_enriched_for_TF_motif) & set(r2g_df.index))]
                if len(r2g_df_enriched_for_TF_motif) > 0:
                    transcription_factor = cistrome_name.split('_')[0]
                    relevant_tfs.append(transcription_factor)
                    eRegulons.append(
                        eRegulon(
                            transcription_factor=transcription_factor,
                            cistrome_name=cistrome_name,
                            is_extended=('extended' in cistrome_name),
                            regions2genes=list(r2g_df_enriched_for_TF_motif[list(
                                REGIONS2GENES_HEADER)].itertuples(index=False, name='r2g')),
                            context=context.union(frozenset(['Cistromes_'+cistromes_key]))))
    except Exception as e:
        print(e)
    finally:
        if ray_n_cpu is not None:
            ray.shutdown()
    return set(relevant_tfs), eRegulons


def _merge_single_TF(l_e_modules):
    """
    Helper function to merge list of :class:`eRegulon` coming from a single transcription factor

    Parameters
    ----------
    l_e_modules
        list of :class:`eRegulon` to merge

    Returns
    -------
    A :class: `eRegulon`
    """
    transcription_factor = set([em.transcription_factor for em in l_e_modules])
    if len(transcription_factor) > 1:
        raise ValueError('l_e_modules should only contain a single TF')
    else:
        transcription_factor = list(transcription_factor)[0]

    cistrome_name = set([em.cistrome_name for em in l_e_modules])
    if len(cistrome_name) > 1:
        cistrome_name = ', '.join(cistrome_name)
    else:
        cistrome_name = list(cistrome_name)[0]

    is_extended = set([em.is_extended for em in l_e_modules])
    if len(is_extended) > 1:
        raise ValueError(
            'l_e_modules should either contain eRegulons which are direct or extended not both.')
    else:
        is_extended = list(is_extended)[0]

    regions2genes_merged = list(
        set(flatten_list([em.regions2genes for em in np.array(l_e_modules)])))
    context = frozenset(flatten_list([em.context for em in l_e_modules]))
    return eRegulon(
        transcription_factor=transcription_factor,
        cistrome_name=cistrome_name,
        is_extended=is_extended,
        regions2genes=regions2genes_merged,
        context=context)


def _merge_across_TF(i):
    """
    Helper function to merge list of :class:`eRegulon` coming from multiple single transcription factor

    Parameters
    ----------
    l_e_modules
        list of :class:`eRegulon` to merge

    Yields
    -------
    A :class: `eRegulon`
    """
    for l_e_modules in i:
        l_e_modules = np.array(l_e_modules)
        TFs = [em.transcription_factor for em in l_e_modules]
        if len(TFs) > 0:
            grouped = Groupby(TFs)
            for idx in grouped.indices:
                yield _merge_single_TF(l_e_modules[idx])
        else:
            yield []


def _rho_dichotomize(i):
    for l_e_modules in i:
        TF2G_pos_R2G_pos_ems = [em for em in l_e_modules if (
            'positive tf2g' in em.context and 'positive r2g' in em.context)]
        TF2G_pos_R2G_neg_ems = [em for em in l_e_modules if (
            'positive tf2g' in em.context and 'negative r2g' in em.context)]
        TF2G_neg_R2G_pos_ems = [em for em in l_e_modules if (
            'negative tf2g' in em.context and 'positive r2g' in em.context)]
        TF2G_neg_R2G_neg_ems = [em for em in l_e_modules if (
            'negative tf2g' in em.context and 'negative r2g' in em.context)]

        iter_emodules = [
            TF2G_pos_R2G_pos_ems,
            TF2G_pos_R2G_neg_ems,
            TF2G_neg_R2G_pos_ems,
            TF2G_neg_R2G_neg_ems
        ]

        for e_modules in iter_emodules:
            yield e_modules


def _split_direct_indirect(l_e_modules):
    direct_ems = [em for em in l_e_modules if not em.is_extended]
    extended_ems = [em for em in l_e_modules if em.is_extended]

    for e_modules in [direct_ems, extended_ems]:
        yield e_modules


def merge_emodules(SCENICPLUS_obj: SCENICPLUS = None,
                   e_modules: list = None,
                   e_modules_key: str = 'eRegulons',
                   rho_dichotomize: bool = True,
                   key_to_add: str = 'eRegulons',
                   inplace: bool = True):
    """
    Function to merge list of :class:`eRegulon`

    Parameters
    ----------
    SCENICPLUS_obj
        An instance of :class: `~scenicplus.scenicplus_class.SCENICPLUS`, containing eRegulons in slot .uns.
        default: None
    e_modules
        A list of :class:`eRegulon`.
        default: None
    e_modules_key
        A string key specifying where to find a list of :class:`eRegulon` under .uns slot of`SCENICPLUS_obj`.
        default: "eRegulons"
    rho_dichotomize
        A boolean specifying wether or not to split region-to-gene links based on postive/negative correlation coefficients.
        default: True
    key_to_add
        A string key specifying where to store the list of merged :class:`eRegulon` in the .uns slot of `SCENICPLUS_obj`.
    inplace
        A boolean if set to True update the `SCENICPLUS_obj` otherwise return list of merged :class:`eRegulon`
    """
    # check input
    if SCENICPLUS_obj is not None:
        if e_modules_key not in SCENICPLUS_obj.uns.keys():
            raise ValueError(f'No e-modules found under key: {e_modules_key}.')

    e_modules = SCENICPLUS_obj.uns[e_modules_key] if SCENICPLUS_obj is not None else e_modules

    if rho_dichotomize:
        iter_merger = _merge_across_TF(
            _rho_dichotomize(_split_direct_indirect(e_modules)))
    else:
        iter_merger = _merge_across_TF(_split_direct_indirect(e_modules))

    if inplace:
        SCENICPLUS_obj.uns[key_to_add] = [eReg for eReg in iter_merger]
    else:
        return [eReg for eReg in iter_merger]
