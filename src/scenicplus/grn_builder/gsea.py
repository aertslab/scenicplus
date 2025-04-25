from gseapy.algorithm import enrichment_score
from gseapy.algorithm import gsea_compute
import numpy as np
import pandas as pd

def run_gsea(ranked_gene_list: pd.Series,
             gene_set: list,
             n_perm: int = 1000,
             ascending: bool = False,
             name: str = 'gene_set',
             seed: int = 555):
    """
    Calculates gene set enrichment score (and estimates its significance) and leading edge for the gene set in the ranked gene list using gseapy prerank.

    Parameters
    ----------
    ranked_gene_list: pd.Series
        A :class: `pd.Series` containing the gene scores, with gene names as index. 
    gene_set: 
        A list-like object containing a set of genes which will be tested for enrichment in the top / bottom of the sorted list of genes 
    n_perm: int
        Number of permutations to use to estimate the empirical p value.
    return_res: bool
        Wether or not to return the GSEAplot object. This object can then be visalised by applying the method .add_axes()
    name: str
        Name of the gene set, used as plot title.

    Returns
    ------
    NES, pval, LE_genes, (res)
        Normalized enrichment score (NES), estimated p value (pval), the genes in the leading edge (LE_genes) (and the values needed to plot the random walk (res))
    """

    # run gsea prerank with default option
    gmt = {name: list(gene_set)}
    gsea_results, ind, rank_ES, gs = gsea_compute(data=ranked_gene_list, n=n_perm, gmt=gmt,
                                                  weighted_score_type=1, permutation_type='gene_set', method=None,
                                                  pheno_pos='Pos', pheno_neg='Neg', classes=None, ascending=ascending, seed = seed)

    # extract enrichment scores
    gseale = list(gsea_results)[0]
    RES = rank_ES[0]
    ind = ind[0]

    # extract Leading Edges information
    es = gseale[0]
    if es > 0:
        idx = RES.argmax()
        ldg_pos = list(filter(lambda x: x <= idx, ind))
    elif es < 0:
        idx = RES.argmin()
        ldg_pos = list(filter(lambda x: x >= idx, ind))
    else:
        ldg_pos = ind  # es == 0 ?

    # return results
    nes = gseale[1]
    pval = gseale[2]
    fdr = gseale[3]
    LE = list(map(str, ranked_gene_list.iloc[ldg_pos].index))
    return nes, pval, LE


def run_enrichr(ranked_gene_list: np.array,
                gene_set,
                weights: np.array = None,
                p: float = 0,
                n_perm=1000,
                return_res=False):
    """
    Calculates enrichment score (and estimates its significance) and leading edge for the gene set in the ranked gene list ussing gseapy enrichr.

    Parameters
    ----------
    ranked_gene_list: np.array
        A :class: `np.array` containing a sorted list of genes (highest score first). 
    gene_set: 
        A list-like object containing a set of genes which will be tested for enrichment in the top / bottem of the sorted list of genes 
    weights: np.array
        A :class: `np.array` containing weights associated with each gene in ranked_gene_list. Default is None (i.e. each gene is weight equally).
        See: Subramanian et al. 2005 PNAS for more details. 
    p: float
        A floating point value which will be used as exponent on the weights. Default is 0 (i.e. each gene is weight equally).
        See: Subramanian et al. 2005 PNAS for more details.
    n_perm: int
        Number of permutations to use to estimate the empirical p value.
    return_res: bool
        Wether or not to return the values to plot the random walk.

    Returns
    ------
    NES, pval, LE_genes, (res)
        Normalized enrichment score (NES), estimated p value (pval), the genes in the leading edge (LE_genes) (and the values needed to plot the random walk (res))
    """

    # Calculate enrichment score and permuted enrichments

    # es: enrichment score is the supremum of the random walk (brownian bridge)
    # esnull: is the enrichment score obtained by shuffeling the hit locations along the ranking, this is done n_perm times
    # res: are the values of the random walk (brownian bridge)
    # The random walk is calculated as described in: Subramanian et al. 2005 PNAS. Here the correlation vector is replace by the weight vector.
    # For more info see also: "Introduction to Statistical Methods for Analyzing Large Data Sets: Gene-Set Enrichment Analysis. NEIL R. CLARK, AVI MA’AYAN 2011"

    es, esnull, _, res = enrichment_score(
        gene_list=ranked_gene_list,
        correl_vector=weights,
        gene_set=gene_set,
        weighted_score_type=p,
        nperm=n_perm,
        rs=seed,
        single=False,
        scale=True)
    # estimate p value

    # see: North BV, Curtis D, Sham PC. A note on the calculation of empirical P values from Monte Carlo procedures. Am J Hum Genet. 2002;71(2):439-441. doi:10.1086/341527
    # Correct formula for obtaining an empirical p value is (r + 1) / (n + 1)
    #   with n the number of replicate samples that have been simulated under the null hypothesis
    #   and r the number of these replicates that produce a test statistic that's equal or more extreme compared to the observed test statistic.
    #   Estimating the p value like this also circumvents getting values equal to 0.

    # esnull is bimodally distributed, therefore this if statement
    pval = (sum(esnull >= es) + 1) / (sum(esnull >= 0) +
                                      1) if es >= 0 else (sum(esnull < es) + 1) / (sum(esnull < 0) + 1)

    # calculate NES, this code is modified from GSEApy
    esnull_mean_pos = (esnull[esnull >= 0]).mean()
    esnull_mean_neg = (esnull[esnull < 0]).mean()
    denom = esnull_mean_pos if es >= 0 else esnull_mean_neg
    NES = es / abs(denom)

    # get genes in leading edge
    rank_at_max = np.argmax(res)
    # + 1 because: "We define the leading-edge subset to be those genes in the gene set S that appear in the ranked list L at, or before,
    #              the point where the running sum reaches its maximum deviation from zero"
    # https://www.pnas.org/content/102/43/15545
    LE_genes = ranked_gene_list[0:rank_at_max + 1]

    if return_res:
        return NES, pval, LE_genes, res
    else:
        return NES, pval, LE_genes
