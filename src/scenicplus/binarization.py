# coding=utf-8
#
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn import mixture
from tqdm import tqdm


def compare_gaussian_mixture_models(
        X: np.array,
        gmm1: mixture.GaussianMixture,
        gmm2: mixture.GaussianMixture,
        min_diff_bic=10
        ) -> bool:
    """
    Compare BIC between 2 gaussian mixture models
    Checks whether difference between BICs is smaller than 'min_diff_bic'
    :param X: 2d np.array that contains values models were fit to
    :param gmm1: first gaussian mixture model
    :param gmm2: second gaussian mixture model
    :param min_diff_bic: minimum difference between BIC  of gmm1 vs BIC of gmm2 required
    :return: True if first model has smaller BIC otherwise False
    """
    
    # compare BICs
    is_smaller = False
    
    bic1 = gmm1.bic(X)
    bic2 = gmm2.bic(X)

    # check absolute difference
    if (bic2 - bic1) > min_diff_bic:
        is_smaller = True
    else:
        is_smaller = False

    return is_smaller



def binarize(
    auc_mtx: pd.DataFrame,
    seed=None,
) -> (pd.DataFrame, pd.Series):
    """
    "Binarize" the supplied AUC matrix, i.e. decide if for each cells in the matrix a regulon is active or not based
    on the bimodal distribution of the AUC values for that regulon.
    :param auc_mtx: The dataframe with the AUC values for all cells and regulons (n_cells x n_regulons).
    :param threshold_overides: A dictionary that maps name of regulons to manually set thresholds.
    :return: A "binarized" dataframe and a series containing the AUC threshold used for each regulon.
    """

    num_regulons = auc_mtx.shape[1]
    
    # fit gaussian models to derive thresholds
    thresholds = np.zeros(num_regulons)
    
    for i in tqdm(range(num_regulons)):
        regulon_name = auc_mtx.columns[i]
        X = auc_mtx[regulon_name].values
        X = X.reshape(-1, 1)

        # fit 1 gaussian
        gmm1 = mixture.GaussianMixture(n_components=1, covariance_type="full", random_state=seed).fit(X)
        # fit 2 gaussian
        gmm2 = mixture.GaussianMixture(n_components=2, covariance_type="full", random_state=seed).fit(X)

        # check bimodality
        if compare_gaussian_mixture_models(X, gmm1=gmm2, gmm2=gmm1):
             thresholds[i] = minimize_scalar(fun=stats.gaussian_kde(X.reshape(1,-1)),
                                              bounds=sorted(gmm2.means_), method="bounded").x[0]
        else:
             thresholds[i] = np.mean(X) + 2 * np.std(X)

    # compile pandas series
    thresholds = pd.Series(index=auc_mtx.columns, data=thresholds)

    return (auc_mtx > thresholds).astype(int), thresholds
