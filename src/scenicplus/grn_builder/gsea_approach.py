import pandas as pd
import numpy as np
import logging
import sys
from tqdm import tqdm
import ray

from ..scenicplus_class import SCENICPLUS
from ..utils import p_adjust_bh
from .modules import create_emodules, eRegulon, merge_emodules, RHO_THRESHOLD, TARGET_GENE_NAME
from .gsea import run_gsea 

#create logger
level    = logging.INFO
format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
handlers = [logging.StreamHandler(stream=sys.stdout)]
logging.basicConfig(level = level, format = format, handlers = handlers)
log = logging.getLogger('GSEA')

def run_gsea_for_e_module(e_module, rnk, gsea_n_perm, context):
    """
    Helper function to run gsea for single e_module

    Parameters
    ----------
    e_module
        Instance of :class:`~scenicplus.grn_builder.modules.eRegulon`
    rnk
        Instance of :class:`pd.Series` containing ranked genes.
    gsea_n_perm
        Int specifying number of permutations for gsea p value calculation
    context
        Context of eRegulon
    
    Returns
    -------
    Instance of :class:`~scenicplus.grn_builder.modules.eRegulon`
    """
    gene_set = e_module.target_genes #is already made unique by the class
    TF = e_module.transcription_factor
    NES, pval, LE_genes = run_gsea(
        ranked_gene_list = rnk,
        gene_set = gene_set,
        n_perm = gsea_n_perm)
    return eRegulon(
            transcription_factor = TF,
            cistrome_name = e_module.cistrome_name,
            is_extended = e_module.is_extended,
            regions2genes = e_module.regions2genes,
            context = e_module.context.union(context),
            gsea_enrichment_score = NES,
            gsea_pval = pval,
            in_leading_edge = [getattr(r2g, TARGET_GENE_NAME) in LE_genes for r2g in e_module.regions2genes])

@ray.remote
def ray_run_gsea_for_e_module(e_module, rnk, gsea_n_perm, context):
    return run_gsea_for_e_module(e_module, rnk, gsea_n_perm, context)

def build_grn(SCENICPLUS_obj: SCENICPLUS,
             adj_key = 'TF2G_adj',
             cistromes_key = 'Unfiltered',
             region_to_gene_key = 'region_to_gene',
             order_regions_to_genes_by = 'importance',
             order_TFs_to_genes_by = 'importance',
             gsea_n_perm = 1000,
             quantiles = (0.85, 0.90),
             top_n_regionTogenes_per_gene = (5, 10, 15),
             top_n_regionTogenes_per_region = (),
             binarize_using_basc = False,
             min_regions_per_gene = 0,
             rho_dichotomize_tf2g = True,
             rho_dichotomize_r2g = True,
             rho_dichotomize_eregulon = True,
             keep_only_activating = False,
             rho_threshold = RHO_THRESHOLD,
             NES_thr = 0,
             adj_pval_thr = 1,
             min_target_genes = 5,
             inplace = True,
             key_added = 'eRegulons',
             ray_n_cpu = None,
             merge_eRegulons = True,
             keep_extended_motif_annot = False,
             disable_tqdm = False,
             **kwargs):
    """
    Build GRN using GSEA approach

    Parameters
    ---------
    SCENICPLUS_obj
        An instance of :class: `~scenicplus.scenicplus_class.SCENICPLUS`
    adj_key
        Key under which to find TF2G adjacencies
        default: "TF2G_adj"
    region_to_gene_key
        Key under which to find R2G adjacnecies
        default: "region_to_gene"
    gsea_n_perm
        Int specifying number of gsea permutations to run for p value calculation
        default: 1000
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
        A boolean specifying wether or not to binarize region-to-gene links using BASC.
        See: Hopfensitz M, et al. Multiscale binarization of gene expression data for reconstructing Boolean networks. 
             IEEE/ACM Trans Comput Biol Bioinform. 2012;9(2):487-98.
        Defailt: False
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
    NES_thr
        Float specifying threshold on gsea NES value
        defaut: 0
    adj_pval_thr
        Float specifying threshold on gsea adjusted p value
        default: 1
    min_target_genes
        Int specifying minumum number of target genes in leading edge
        default: 5
    inplace
        Boolean specifying wether to store results in :param:`SCENICPLUS_obj`
        default: True
    key_added
        Key specifying in under which key to store result in :param:`SCENICPLUS_obj`.uns
        default: "eRegulons"
    ray_n_cpu
        Int specifying number of cores to use
        default: None
    merge_eRegulons
        Boolean specifying wether to merge eRegulons form the same TF but different thresholding approaches
        default: True
    keep_extended_motif_annot
        A boolean specifying wether or not keep extended motif annotations for further analysis.
        default: False
    **kwargs
        Additional keyword arguments passed to `ray.init`
    
    Returns
    -------
    None if inplace is True else list of :class:`~scenicplus.grn_builder.modules.eRegulon` instances.
    """

    if not adj_key in SCENICPLUS_obj.uns.keys():
        raise ValueError(f'key {adj_key} not found in uns slot. Please first load TF2G adjacencies!')
    
    log.info('Thresholding region to gene relationships')
    relevant_tfs, e_modules = create_emodules(
        SCENICPLUS_obj = SCENICPLUS_obj,
        region_to_gene_key = region_to_gene_key,
        quantiles = quantiles,
        top_n_regionTogenes_per_gene = top_n_regionTogenes_per_gene,
        top_n_regionTogenes_per_region = top_n_regionTogenes_per_region,
        binarize_using_basc = binarize_using_basc,
        min_regions_per_gene = min_regions_per_gene,
        rho_dichotomize = rho_dichotomize_r2g,
        keep_only_activating = keep_only_activating,
        rho_threshold = rho_threshold,
        keep_extended_motif_annot = keep_extended_motif_annot,
        order_regions_to_genes_by = order_regions_to_genes_by,
        disable_tqdm=disable_tqdm,
        ray_n_cpu=ray_n_cpu,
        cistromes_key=cistromes_key,
        **kwargs)
    
    log.info('Subsetting TF2G adjacencies for TF with motif.')
    TF2G_adj_relevant = SCENICPLUS_obj.uns[adj_key].loc[[tf in relevant_tfs for tf in SCENICPLUS_obj.uns[adj_key]['TF']]]

    if ray_n_cpu is not None:
        ray.init(num_cpus = ray_n_cpu, **kwargs)
        jobs = []
    try:
        log.info(f'Running GSEA...')
        new_e_modules = []
        TF_to_TF_adj_d = {} #dict so adjacencies matrix is only subsetted once per TF (improves performance)
        tqdm_desc = 'initializing' if ray_n_cpu is not None else 'Running using single core'
        for e_module in tqdm(e_modules, total = len(e_modules), desc = tqdm_desc, disable=disable_tqdm):
            TF = e_module.transcription_factor
            if TF in TF_to_TF_adj_d.keys():
                TF2G_adj = TF_to_TF_adj_d[TF]
            else:
                TF2G_adj = TF2G_adj_relevant.loc[TF2G_adj_relevant['TF'] == TF]
                TF_to_TF_adj_d[TF] = TF2G_adj
            if rho_dichotomize_tf2g:
                TF2G_adj_activating = TF2G_adj.loc[TF2G_adj['rho'] > rho_threshold]
                TF2G_adj_repressing = TF2G_adj.loc[TF2G_adj['rho'] < -rho_threshold]

                TF2G_adj_activating.index = TF2G_adj_activating['target']
                TF2G_adj_repressing.index = TF2G_adj_repressing['target']

                if len(TF2G_adj_activating) > 0:
                    if ray_n_cpu is None:
                        new_e_modules.append(
                            run_gsea_for_e_module(
                                e_module, 
                                pd.Series(TF2G_adj_activating[order_TFs_to_genes_by]).sort_values(ascending = False),
                                gsea_n_perm,
                                frozenset(['positive tf2g'])))
                    else:
                        jobs.append(
                            ray_run_gsea_for_e_module.remote(
                                e_module, 
                                pd.Series(TF2G_adj_activating[order_TFs_to_genes_by]).sort_values(ascending = False),
                                gsea_n_perm,
                                frozenset(['positive tf2g'])))
                        
                if len(TF2G_adj_repressing) > 0:
                    if ray_n_cpu is None:
                        new_e_modules.append(
                            run_gsea_for_e_module(
                                e_module,
                                pd.Series(TF2G_adj_repressing[order_TFs_to_genes_by]).sort_values(ascending = False),
                                gsea_n_perm,
                                frozenset(['negative tf2g'])))
                    else:
                        jobs.append(
                            ray_run_gsea_for_e_module.remote(
                                e_module,
                                pd.Series(TF2G_adj_repressing[order_TFs_to_genes_by]).sort_values(ascending = False),
                                gsea_n_perm,
                                frozenset(['negative tf2g'])))
            else:
                if ray_n_cpu is None:
                    new_e_modules.append(
                        run_gsea_for_e_module(
                            e_module,
                            pd.Series(TF2G_adj[order_TFs_to_genes_by]).sort_values(ascending = False),
                            gsea_n_perm,
                            frozenset([''])))
                else:
                    jobs.append(
                        ray_run_gsea_for_e_module.remote(
                            e_module,
                            pd.Series(TF2G_adj[order_TFs_to_genes_by]).sort_values(ascending = False),
                            gsea_n_perm,
                            frozenset([''])))
        if ray_n_cpu is not None:
            def to_iterator(obj_ids):
                while obj_ids:
                    finished_ids, obj_ids = ray.wait(obj_ids)
                    for finished_id in finished_ids:
                        yield ray.get(finished_id)

            for e_module in tqdm(to_iterator(jobs), 
                                 total=len(jobs), 
                                 desc = f'Running using {ray_n_cpu} cores',
                                 smoothing = 0.1,
                                 disable=disable_tqdm):
                new_e_modules.append(e_module)

    except Exception as e:
        print(e)
    finally:
        if ray_n_cpu is not None:
            ray.shutdown()
    
    #filter out nans
    new_e_modules = [m for m in new_e_modules if not np.isnan(m.gsea_enrichment_score) and not np.isnan(m.gsea_pval)]

    log.info(f'Subsetting on adjusted pvalue: {adj_pval_thr}, minimal NES: {NES_thr} and minimal leading edge genes {min_target_genes}')
    #subset on adj_p_val
    adj_pval = p_adjust_bh([m.gsea_pval for m in new_e_modules])
    if any([np.isnan(p) for p in adj_pval]):
        Warning('Something went wrong with calculating adjusted p values, early returning!')
        return new_e_modules
    
    for module, adj_pval in zip(new_e_modules, adj_pval):
        module.gsea_adj_pval = adj_pval

    e_modules_to_return = []
    for module in new_e_modules:
        if module.gsea_adj_pval < adj_pval_thr and module.gsea_enrichment_score > NES_thr and sum(module.in_leading_edge) >= min_target_genes:
            e_modules_to_return.append(module.subset_leading_edge(inplace = False ))
    if merge_eRegulons:
        log.info('Merging eRegulons')
        e_modules_to_return = merge_emodules(e_modules = e_modules_to_return, inplace = False, rho_dichotomize = rho_dichotomize_eregulon)
    e_modules_to_return = [x for x in e_modules_to_return if not isinstance(x, list)]
    if inplace:
        log.info(f'Storing eRegulons in .uns[{key_added}].')
        SCENICPLUS_obj.uns[key_added] = e_modules_to_return
    else:
        return e_modules_to_return
