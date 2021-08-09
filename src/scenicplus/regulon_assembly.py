import pandas as pd
import numpy as np
import logging
import sys
from .gsea import run_gsea
import ray

def assemble_e_regulons(
    df_region2gene: pd.DataFrame,
    df_tf2gene: pd.DataFrame,
    df_tf2region: pd.DataFrame,
    ray_n_cpu = None,
    sort_by = 'importance',
    n_perm = 1000,
    GSEA_NES_thr = 3,
    GSEA_pVal_thr = 0.05,
    **kwargs):
    """
    Assembles final eRegulons by linking regions in cistromes to genes but only keeping genes which are enriched in the top of the TF-to-gene adjecency ranking.
    
    Parameters
    ----------
    df_region2gene: pd.DataFrame
        A :class: `pd.DataFrame` containing region-to-gene links (should be already filtered). 
        Should minimally contain following columns: "region", "gene", "importance", "rho", "aggr_rank_score". Extra columns are allowed.
    df_tf2gene: pd.DataFrame
        A :class: `pd.DataFrame` containing tf-to-gene adjecencies (should be already filtered).
        Should minimally contain following columns: "TF", "gene", "importance", "rho", "regulation". Extra columns are allowed
    df_tf2region: pd.DataFrame
        A :class: `pd.DataFrame` containing tf-to-region links based on motif enrichment (should be already filtered).
        Should minimally contain following columns: "TF", "region", "max_nes". Extra columns are allowed
    ray_n_cpu: int
        Number of cores to use when performing GSEA. Default is None (i.e. single core).
    sort_by: str
        Key by which to sort the tf-to-gene adjecencies to generate a ranking for GSEA. Default is "importance".
        When the key is equal to "rho_abs", an extra column containing absolute correlation coefficient values is generated to create the ranking.
    n_perm: int
        Number of permutations to perform when empirically estimating the GSEA p value. Default is 1000.
    GSEA_NES_thr: int
        Mininum GSEA Normalized Enrichment Score (NES) value to retain. Default is 3.
    GSEA_pVal_thr: float
        Maximum GSEA p value to retain. Default is 0.05.
    kwargs
        Extra keyword arguments passed to ray.init()
    
    Returns
    ------
    pd.DataFrame
        A :class: `pd.DataFrame` containing the final eRegulon table.
    """

    # Create logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('ASM')

    #link cistrome regions to genes
    log.info('Linking cistromes to genes.')
    df_cistrome_to_gene = df_tf2region.merge(df_region2gene, on = 'region')

    #perform gsea by TF with cistrome genes as geneset and TF2Gene adj as ranking
    log.info('Performing GSEA to test which genes linked to region in TF cistrome are enriched in the top of the TF-to-gene adjacency ranking.')
    TFs = list(set(df_tf2gene['TF']) & set(df_cistrome_to_gene['TF']))

    if sort_by == 'abs_rho':
        df_tf2gene['abs_rho'] = abs(df_tf2gene['rho'])
    
    if ray_n_cpu is None:
        dict_GSEA_result = {}
        from tqdm import tqdm
        for TF in tqdm(TFs, total = len(TFs)):
            rnk = df_tf2gene.loc[df_tf2gene['TF'] == TF].sort_values(sort_by, ascending = False)
            gene_set = list(set(df_cistrome_to_gene.loc[df_cistrome_to_gene['TF'] == TF, 'gene']))
            NES, pval, LE_genes = run_gsea(
                ranked_gene_list = rnk['gene'].to_numpy(),
                gene_set = gene_set,
                n_perm = n_perm)
            dict_GSEA_result[TF] = (NES, pval, LE_genes)
    else:
        @ray.remote
        def _ray_run_gsea(ranked_gene_list, gene_set, weights = None, p = 0, n_perm = 1000, return_res = False):
            return run_gsea(ranked_gene_list, gene_set, weights, p, n_perm, return_res)
        ray.init(num_cpus = ray_n_cpu, **kwargs)
        try:
            jobs = []
            for TF in TFs:
                rnk = df_tf2gene.loc[df_tf2gene['TF'] == TF].sort_values(sort_by, ascending = False)
                gene_set = list(set(df_cistrome_to_gene.loc[df_cistrome_to_gene['TF'] == TF, 'gene']))
                jobs.append(
                    _ray_run_gsea.remote(
                        ranked_gene_list = rnk['gene'].to_numpy(),
                        gene_set = gene_set,
                        n_perm = n_perm
                    ))
            l_GSEA_result = ray.get(jobs)
        except Exception as e:
            print(e)
        finally:
            ray.shutdown()
        #transform list of GSEA results in dict
        dict_GSEA_result = {TF: gsea_result for TF, gsea_result in zip(TFs, l_GSEA_result)}

    #Transform dict dict_GSEA_result in dataframe df_GSEA
    df_GSEA = pd.DataFrame(dict_GSEA_result).T
    df_GSEA.columns = ['GSEA_NES', 'GSEA_pval', 'gene']
    df_GSEA['TF'] = df_GSEA.index
    df_GSEA.reset_index(inplace = True)
    df_GSEA.drop('index', axis = 1, inplace = True)
    df_GSEA = df_GSEA.explode('gene')

    #threshold on NES and pval
    df_GSEA = df_GSEA.loc[
        np.logical_and(
            df_GSEA['GSEA_NES'] > GSEA_NES_thr,
            df_GSEA['GSEA_pval'] < GSEA_pVal_thr)]

    #combine results
    log.info('Subsetting for leading edge genes and generating final eRegulon dataframe.')
    #subset cistrome to gene for leading edge genes
    df_cistrome_to_gene_enr = df_cistrome_to_gene.merge(df_GSEA, on = ['TF', 'gene'])
    df_final = df_cistrome_to_gene_enr.merge(df_tf2gene, on = ['TF', 'gene'], suffixes = ['_r2g', '_tf2g'])
    log.info('Done!')
    return df_final

