from ..scenicplus_class import SCENICPLUS
from .modules import create_emodules, eRegulon, TARGET_GENE_NAME, merge_emodules
from ..gsea import run_gsea #TODO:probably better to move this into the grn_builder directory.
from pyscenic.utils import add_correlation, COLUMN_NAME_CORRELATION
import pandas as pd
import numpy as np
import logging
from .modules import RHO_THRESHOLD
import sys
from tqdm import tqdm
from ..utils import p_adjust_bh
import ray

#create logger
level    = logging.INFO
format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
handlers = [logging.StreamHandler(stream=sys.stdout)]
logging.basicConfig(level = level, format = format, handlers = handlers)
log = logging.getLogger('GSEA')

def load_TF2G_adj_from_file(SCENICPLUS_obj: SCENICPLUS, 
                            f_adj: str, 
                            inplace = True, 
                            key_added = 'TF2G_adj', 
                            rho_threshold = RHO_THRESHOLD):
    log.info(f'Reading file: {f_adj}')
    df_TF_gene_adj = pd.read_csv(f_adj, sep = '\t')
    #only keep relevant entries
    idx_to_keep = np.logical_and( np.array([tf in SCENICPLUS_obj.gene_names for tf in df_TF_gene_adj['TF']]),
                                  np.array([gene in SCENICPLUS_obj.gene_names for gene in df_TF_gene_adj['target']]) )
    df_TF_gene_adj_subset = df_TF_gene_adj.loc[idx_to_keep]
    
    if not COLUMN_NAME_CORRELATION in df_TF_gene_adj_subset.columns:
        log.info(f'Adding correlation coefficients to adjacencies.')
        df_TF_gene_adj_subset = add_correlation(
            adjacencies = df_TF_gene_adj_subset,
            ex_mtx = SCENICPLUS_obj.to_df(layer = 'EXP'),
            rho_threshold = rho_threshold)
    
    if inplace:
        log.info(f'Storing adjacencies in .uns[{key_added}].')
        SCENICPLUS_obj.uns[key_added] = df_TF_gene_adj_subset
    else:
        return df_TF_gene_adj_subset

def run_gsea_for_e_module(e_module, rnk, gsea_n_perm, context):
    gene_set = e_module.target_genes #is already made unique by the class
    TF = e_module.transcription_factor
    NES, pval, LE_genes = run_gsea(
        ranked_gene_list = rnk,
        gene_set = gene_set,
        n_perm = gsea_n_perm)
    return eRegulon(
            transcription_factor = TF,
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
             region_to_gene_key = 'region_to_gene',
             gsea_n_perm = 1000,
             quantiles = (0.75, 0.90),
             top_n_regionTogenes_per_gene = (50, 100),
             top_n_regionTogenes_per_region = (),
             binarize_basc = False,
             min_regions_per_gene = 5,
             rho_dichotomize=True,
             keep_only_activating=False,
             rho_threshold=RHO_THRESHOLD,
             NES_thr = 0,
             adj_pval_thr = 0.05,
             min_target_genes = 5,
             inplace = True,
             key_added = 'eRegulons',
             ray_n_cpu = None,
             merge_eRegulons = True,
             keep_extended_motif_annot = False,
             **kwargs):

    if not adj_key in SCENICPLUS_obj.uns.keys():
        raise ValueError(f'key {adj_key} not found in uns slot. Please first load TF2G adjacencies!')
    
    log.info('Thresholding region to gene relationships')
    relevant_tfs, e_modules = create_emodules(
        SCENICPLUS_obj = SCENICPLUS_obj,
        region_to_gene_key = region_to_gene_key,
        thresholds = quantiles,
        top_n_target_genes = top_n_regionTogenes_per_gene,
        top_n_target_regions = top_n_regionTogenes_per_region,
        binarize_basc = binarize_basc,
        min_regions_per_gene = min_regions_per_gene,
        rho_dichotomize = rho_dichotomize,
        keep_only_activating = keep_only_activating,
        rho_threshold = rho_threshold,
        keep_extended_motif_annot = keep_extended_motif_annot)
    
    log.info('Subsetting TF2G adjacencies for TF with motif.')
    TF2G_adj_relevant = SCENICPLUS_obj.uns[adj_key].loc[[tf in relevant_tfs for tf in SCENICPLUS_obj.uns[adj_key]['TF']]]

    if ray_n_cpu is not None:
        ray.init(num_cpus = ray_n_cpu, **kwargs)
        jobs = []
    try:
        log.info(f'Running GSEA...')
        new_e_modules = []
        tqdm_desc = 'initializing' if ray_n_cpu is not None else 'Running using single core'
        for e_module in tqdm(e_modules, total = len(e_modules), desc = tqdm_desc):
            TF = e_module.transcription_factor
            TF2G_adj = TF2G_adj_relevant.loc[TF2G_adj_relevant['TF'] == TF]
            if rho_dichotomize:
                TF2G_adj_activating = TF2G_adj.loc[TF2G_adj['rho'] > rho_threshold]
                TF2G_adj_repressing = TF2G_adj.loc[TF2G_adj['rho'] < -rho_threshold]

                TF2G_adj_activating.index = TF2G_adj_activating['target']
                TF2G_adj_repressing.index = TF2G_adj_repressing['target']

                if len(TF2G_adj_activating) > 0:
                    if ray_n_cpu is None:
                        new_e_modules.append(
                            run_gsea_for_e_module(
                                e_module, 
                                pd.Series(TF2G_adj_activating['importance']).sort_values(ascending = False),
                                gsea_n_perm,
                                frozenset(['positive tf2g'])))
                    else:
                        jobs.append(
                            ray_run_gsea_for_e_module.remote(
                                e_module, 
                                pd.Series(TF2G_adj_activating['importance']).sort_values(ascending = False),
                                gsea_n_perm,
                                frozenset(['positive tf2g'])))
                        
                if len(TF2G_adj_repressing) > 0:
                    if ray_n_cpu is None:
                        new_e_modules.append(
                            run_gsea_for_e_module(
                                e_module,
                                pd.Series(TF2G_adj_repressing['importance']).sort_values(ascending = False),
                                gsea_n_perm,
                                frozenset(['negative tf2g'])))
                    else:
                        jobs.append(
                            ray_run_gsea_for_e_module.remote(
                                e_module,
                                pd.Series(TF2G_adj_repressing['importance']).sort_values(ascending = False),
                                gsea_n_perm,
                                frozenset(['negative tf2g'])))
            else:
                if ray_n_cpu is None:
                    new_e_modules.append(
                        run_gsea_for_e_module(
                            e_module,
                            pd.Series(TF2G_adj['importance']).sort_values(ascending = False),
                            gsea_n_perm,
                            frozenset([''])))
                else:
                    jobs.append(
                        ray_run_gsea_for_e_module.remote(
                            e_module,
                            pd.Series(TF2G_adj['importance']).sort_values(ascending = False),
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
                                 smoothing = 0.1):
                new_e_modules.append(e_module)

    except Exception as e:
        print(e)
    finally:
        if ray_n_cpu is not None:
            ray.shutdown()
    
    #filter out nans
    new_e_modules = [m for m in new_e_modules if not np.isnan(m.gsea_enrichment_score)]

    log.info(f'Subsetting on adjusted pvalue: {adj_pval_thr}, minimal NES: {NES_thr} and minimal leading edge genes {min_target_genes}')
    #subset on adj_p_val
    adj_pval = p_adjust_bh([m.gsea_pval for m in new_e_modules])
    for module, adj_pval in zip(new_e_modules, adj_pval):
        module.gsea_adj_pval = adj_pval
    
    e_modules_to_return = []
    for module in new_e_modules:
        if module.gsea_adj_pval < adj_pval_thr and module.gsea_enrichment_score > NES_thr and sum(module.in_leading_edge) >= min_target_genes:
            e_modules_to_return.append(module.subset_leading_edge(inplace = False ))
    if merge_eRegulons:
        log.info('Merging eRegulons')
        e_modules_to_return = merge_emodules(e_modules = e_modules_to_return, inplace = False)
    if inplace:
        log.info(f'Storing eRegulons in .uns[{key_added}].')
        SCENICPLUS_obj.uns[key_added] = e_modules_to_return
    else:
        return e_modules_to_return

