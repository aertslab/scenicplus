"""Generate enhancer drive GRNs (eGRS) using the GSEA approach.

Using this approach we will test if the gene set obtained from region-to-gene links, where the regions have a high score for a motif of a certain TF 
(region indicated in black in the diagram: r8, r11, r18, r20, r22, r24), 
are enriched in the top of the ranking based on the TF-to-gene links of the same TF (bottom right panel). 

Only genes, from the set, and the regions linked to these genes in the top of the ranking (i.e. leading edge) will be kept.

This aproach is done seperatly for positive and negative TF and region to gene links.

Generating following four combinations:

.. list-table:: Possible eRegulons
   :widths: 25 25 50
   :header-rows: 1

   * - TF-to-gene relationship
     - region-to-gene relationship
     - biological role
   * - positive (+)
     - positive (+)
     - TF opens chromatin and activates gene expression
   * - positive (+)
     - negative (-)
     - When the TF is expressed the target gene is also expressed but the regions linked to the gene are closed.
   * - negative (-)
     - positive (+)
     - When the TF is expressed the target gene is not expressed. When the target gene is expressed, regions linked to this gene are open. TF could be a chromatin closing repressor.
   * - negative (-)
     - negative (-)
     - When the TF is expressed the target gene is not expressed. When the target gene is expressed, regions linked to this gene are closed.


Left panel indicates the TF-to-gene links (blue) and region-to-gene links (yellow).
Witdh of arrows correspond the strength of the connections based on non-linear regression.

Top right panel indicates regions with a high score for a motif of a TF

Bottom right panel shows GSEA analysis, with on the left genes ranked by TF-to-gene connection strength and on the right the gene-set obtained from region-to-gene links.
In the diagram: g2, g6, and g10 are located in the top of the TF-to-gene ranking (i.e. leading edge), only these genes and the regions linked to these genes: r20, r18, r11, r8, r24 and r22 
will be kept. 

"""

import pandas as pd
import numpy as np
import logging
import sys
from tqdm import tqdm
import ray
import anndata
from typing import Dict, List

from scenicplus.scenicplus_class import SCENICPLUS
from scenicplus.utils import p_adjust_bh
from scenicplus.grn_builder.modules import (
    create_emodules, eRegulon, merge_emodules, RHO_THRESHOLD, TARGET_GENE_NAME)
from scenicplus.grn_builder.gsea import run_gsea

# create logger
level = logging.INFO
format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
handlers = [logging.StreamHandler(stream=sys.stdout)]
logging.basicConfig(level=level, format=format, handlers=handlers)
log = logging.getLogger('GSEA')

def _run_gsea_for_e_module(e_module, rnk, gsea_n_perm, context):
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
    gene_set = e_module.target_genes  # is already made unique by the class
    TF = e_module.transcription_factor
    try:
        NES, pval, LE_genes = run_gsea(
            ranked_gene_list=rnk,
            gene_set=gene_set,
            n_perm=gsea_n_perm)
    except:
        NES = np.nan
        pval = np.nan
        LE_genes = np.nan
    return eRegulon(
        transcription_factor=TF,
        cistrome_name=e_module.cistrome_name,
        is_extended=e_module.is_extended,
        regions2genes=e_module.regions2genes,
        context=e_module.context.union(context),
        gsea_enrichment_score=NES,
        gsea_pval=pval,
        in_leading_edge=[getattr(r2g, TARGET_GENE_NAME) in LE_genes for r2g in e_module.regions2genes])


@ray.remote
def _ray_run_gsea_for_e_module(e_module, rnk, gsea_n_perm, context):
    return _run_gsea_for_e_module(e_module, rnk, gsea_n_perm, context)


def build_grn(
        tf_to_gene: pd.DataFrame,
        region_to_gene: pd.DataFrame,
        cistromes: anndata.AnnData,
        is_extended: bool,
        order_regions_to_genes_by='importance',
        order_TFs_to_genes_by='importance',
        gsea_n_perm=1000,
        quantiles=(0.85, 0.90),
        top_n_regionTogenes_per_gene=(5, 10, 15),
        top_n_regionTogenes_per_region=(),
        binarize_using_basc=False,
        min_regions_per_gene=0,
        rho_dichotomize_tf2g=True,
        rho_dichotomize_r2g=True,
        rho_dichotomize_eregulon=True,
        keep_only_activating=False,
        rho_threshold=RHO_THRESHOLD,
        NES_thr=0,
        adj_pval_thr=1,
        min_target_genes=5,
        ray_n_cpu=None,
        merge_eRegulons=True,
        disable_tqdm=False,
        **kwargs) -> List[eRegulon]:
    log.info('Thresholding region to gene relationships')
    relevant_tfs, e_modules = create_emodules(
        region_to_gene=region_to_gene,
        cistromes=cistromes,
        is_extended=is_extended,
        order_regions_to_genes_by=order_regions_to_genes_by,
        quantiles=quantiles,
        top_n_regionTogenes_per_gene=top_n_regionTogenes_per_gene,
        top_n_regionTogenes_per_region=top_n_regionTogenes_per_region,
        binarize_using_basc=binarize_using_basc,
        min_regions_per_gene=min_regions_per_gene,
        rho_dichotomize=rho_dichotomize_r2g,
        keep_only_activating=keep_only_activating,
        rho_threshold=rho_threshold,
        disable_tqdm=disable_tqdm,
        ray_n_cpu=ray_n_cpu,
        **kwargs)
    log.info('Subsetting TF2G adjacencies for TF with motif.')
    TF2G_adj_relevant = tf_to_gene.loc[tf_to_gene['TF'].isin(relevant_tfs)]

    if ray_n_cpu is not None:
        ray.init(num_cpus=ray_n_cpu, **kwargs)
        jobs = []

    log.info('Running GSEA...')
    new_e_modules: List[eRegulon] = []
    # dict so adjacencies matrix is only subsetted once per TF (improves performance)
    TF_to_TF_adj_d:Dict[str, pd.DataFrame] = {}
    tqdm_desc = 'initializing' if ray_n_cpu is not None else 'Running using single core'
    for e_module in tqdm(e_modules, total=len(e_modules), desc=tqdm_desc, disable=disable_tqdm):
        TF = e_module.transcription_factor
        if TF in TF_to_TF_adj_d.keys():
            TF2G_adj = TF_to_TF_adj_d[TF]
        else:
            TF2G_adj = TF2G_adj_relevant.loc[TF2G_adj_relevant['TF'] == TF]
            TF2G_adj.index = TF2G_adj['target']
            TF_to_TF_adj_d[TF] = TF2G_adj
        if rho_dichotomize_tf2g:
            TF2G_adj_activating = TF2G_adj.loc[TF2G_adj['rho']
                                               > rho_threshold]
            TF2G_adj_repressing = TF2G_adj.loc[TF2G_adj['rho']
                                               < -rho_threshold]

            TF2G_adj_activating_ranking = pd.Series(TF2G_adj_activating[order_TFs_to_genes_by]).sort_values(ascending=False)
            TF2G_adj_repressing_ranking = pd.Series(TF2G_adj_repressing[order_TFs_to_genes_by]).sort_values(ascending=False)

            if len(TF2G_adj_activating_ranking) > 0:
                if ray_n_cpu is None:
                    new_e_modules.append(
                        _run_gsea_for_e_module(
                            e_module,
                            TF2G_adj_activating_ranking,
                            gsea_n_perm,
                            frozenset(['positive tf2g'])))
                else:
                    jobs.append(
                        _ray_run_gsea_for_e_module.remote(
                            e_module,
                            TF2G_adj_activating_ranking,
                            gsea_n_perm,
                            frozenset(['positive tf2g'])))

            if len(TF2G_adj_repressing_ranking) > 0:
                if ray_n_cpu is None:
                    new_e_modules.append(
                        _run_gsea_for_e_module(
                            e_module,
                            TF2G_adj_repressing_ranking,
                            gsea_n_perm,
                            frozenset(['negative tf2g'])))
                else:
                    jobs.append(
                        _ray_run_gsea_for_e_module.remote(
                            e_module,
                            TF2G_adj_repressing_ranking,
                            gsea_n_perm,
                            frozenset(['negative tf2g'])))
        else:
            TF2G_adj_ranking = pd.Series(TF2G_adj[order_TFs_to_genes_by]).sort_values(ascending=False)
            if len(TF2G_adj_ranking) > 0:
                if ray_n_cpu is None:
                    new_e_modules.append(
                        _run_gsea_for_e_module(
                            e_module,
                            TF2G_adj_ranking,
                            gsea_n_perm,
                            frozenset([''])))
                else:
                    jobs.append(
                        _ray_run_gsea_for_e_module.remote(
                            e_module,
                            TF2G_adj_ranking,
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
                             desc=f'Running using {ray_n_cpu} cores',
                             smoothing=0.1,
                             disable=disable_tqdm):
            new_e_modules.append(e_module)


    if ray_n_cpu is not None:
        ray.shutdown()

    # filter out nans
    new_e_modules = [m for m in new_e_modules if not np.isnan(
        m.gsea_enrichment_score) and not np.isnan(m.gsea_pval)]

    log.info(
        f'Subsetting on adjusted pvalue: {adj_pval_thr}, minimal NES: {NES_thr} and minimal leading edge genes {min_target_genes}')
    # subset on adj_p_val
    adj_pval = p_adjust_bh([m.gsea_pval for m in new_e_modules])
    if any([np.isnan(p) for p in adj_pval]):
        Warning(
            'Something went wrong with calculating adjusted p values, early returning!')
        return new_e_modules

    for module, adj_pval in zip(new_e_modules, adj_pval):
        module.gsea_adj_pval = adj_pval

    e_modules_to_return: List[eRegulon] = []
    for module in new_e_modules:
        if module.gsea_adj_pval < adj_pval_thr and module.gsea_enrichment_score > NES_thr:
            module_in_LE = module.subset_leading_edge(inplace=False)
            if module_in_LE.n_target_genes >= min_target_genes:
                e_modules_to_return.append(module_in_LE)
    if merge_eRegulons:
        log.info('Merging eRegulons')
        e_modules_to_return = merge_emodules(
            e_modules=e_modules_to_return, inplace=False, rho_dichotomize=rho_dichotomize_eregulon)
    e_modules_to_return = [
        x for x in e_modules_to_return if not isinstance(x, list)]
    return e_modules_to_return
