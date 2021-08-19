import pandas as pd
import numpy as np
import logging
import sys
from .gsea import run_gsea
from tqdm import tqdm
from pyscenic.utils import ACTIVATING_MODULE, REPRESSING_MODULE

def assemble_e_regulons(
    modules: list,
    df_cistromes: pd.DataFrame,
    df_region_to_gene: pd.DataFrame,
    n_perm: int = 5000,
    GSEA_NES_thr: float = 0.0,
    GSEA_PVal_thr: float = 0.01):
    # Create logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('ASM')

    # Link regions of citromes to genes
    log.info('Linking cistromes to genes.')
    df_cistrome_to_gene = df_cistromes.merge(df_region_to_gene, on = 'region')

    # Selected relevant modules
    module_TFs = [module.transcription_factor for module in modules]
    uniq_module_TFs = set(module_TFs)
    uniq_cistrome_TFs = set(df_cistrome_to_gene['TF'])
    cistrome_TFs_w_module = uniq_module_TFs & uniq_cistrome_TFs
    log.info('{} of the {} cistrome transcription factors has a transcription factor to gene module.'.format(
        len(cistrome_TFs_w_module), len(uniq_cistrome_TFs)))
    #convert list of modules to np.array to allow indexing using boolean array
    relevant_modules = list( np.array(modules)[ np.isin(module_TFs, list(cistrome_TFs_w_module)) ] ) 

    #sanity check
    assert len( set([module.transcription_factor for module in relevant_modules]) & cistrome_TFs_w_module) == len(cistrome_TFs_w_module)

    log.info('Performing GSEA for each module to test which genes linked to region in the associated TF cistrome are enriched in the top of the module.')
    
    results = []
    for module in tqdm(relevant_modules, total = len(relevant_modules)):
        TF  = module.transcription_factor
        rnk = pd.Series(module.gene2weight)
        rnk.loc[TF] = max(rnk) + 1 # put the TF itself in the first position of the ranking 
        rnk = rnk.sort_values(ascending = False)
        TF_cistrome = df_cistrome_to_gene.loc[ df_cistrome_to_gene['TF'] == TF ]
        regulation = 1 if ACTIVATING_MODULE in module.context else -1 if REPRESSING_MODULE in module.context else None
        assert regulation is not None
        for group in set( TF_cistrome['group'] ):
            TF_cistrome_group = TF_cistrome.loc[ TF_cistrome['group']  == group]
            gene_set = list( set( TF_cistrome_group['gene'] ) )
            NES, pval, LE_genes = run_gsea(
                ranked_gene_list = rnk.index.to_numpy(),
                gene_set = gene_set,
                n_perm = n_perm)
            results.append( (TF, group, NES, pval, LE_genes, regulation) )
    df_results = pd.DataFrame(results, columns = ['TF', 'group', 'gsea_NES', 'gsea_pval', 'LE_genes', 'regulation'])
    df_sign_results = df_results.loc[np.logical_and(df_results['gsea_pval'] <= GSEA_PVal_thr,  
                                                    df_results['gsea_NES']  >= GSEA_NES_thr)]
    df_sign_results = df_sign_results.explode('LE_genes')
    df_sign_results.rename({'LE_genes': 'gene'}, inplace = True, axis = 1)

    log.info('Subsetting for leading edge genes and generating final eRegulon dataframe.')
    eRegulons = df_sign_results.merge(df_cistrome_to_gene, on = ['TF', 'group', 'gene'], suffixes = ['_TF2G', '_R2G'])
    
    #regenerate adjecencies matrix from modules to be able to add tf to region importances
    adj = pd.DataFrame([(np.repeat(module.transcription_factor, len(module)), 
                         pd.Series(module.gene2weight).index, 
                         pd.Series(module.gene2weight).to_numpy()) for module in modules], 
                         columns = ['TF', 'gene', 'importance']).apply(pd.Series.explode)
    adj.drop_duplicates(inplace = True)

    eRegulons = eRegulons.merge(adj, on = ['TF', 'gene'], suffixes = ['_R2G', '_TF2G'])

    #multiple NES en pvals for single TF to gene, coming from several modules. Take the highest nes and lowest pval
    cols = list(eRegulons.columns)
    cols.remove('gsea_NES')
    cols.remove('gsea_pval')
    eRegulons = eRegulons.groupby(cols, as_index = False)['gsea_NES', 'gsea_pval'].agg({'gsea_NES': max, 'gsea_pval': min})
    log.info('Done!')
    return eRegulons
