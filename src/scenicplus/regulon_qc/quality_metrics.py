from mudata import MuData
from scenicplus.scenicplus_mudata import ScenicPlusMuData
from typing import Union, Callable
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scenicplus.utils import p_adjust_bh

def generate_pseudobulks(
        scplus_mudata: Union[MuData, ScenicPlusMuData],
        variable: str,
        modality: str,
        nr_cells_to_sample: int,
        nr_pseudobulks_to_generate: int,
        seed: int,
        normalize_data: bool = False) -> pd.DataFrame:
    # Input validation
    if variable not in scplus_mudata.obs.columns:
        raise ValueError(f"variable: {variable} not found in scplus_mudata.obs.columns")
    if modality not in scplus_mudata.mod.keys():
        raise ValueError(f"modality: {modality} not found in scplus_mudata.mod.keys()")
    np.random.seed(seed)
    data_matrix = scplus_mudata[modality].to_df()
    if normalize_data:
        data_matrix = np.log1p(data_matrix.T / data_matrix.T.sum(0) * 10**6).T.sum(1)
    variable_to_cells = scplus_mudata.obs \
        .groupby(variable).apply(lambda x: list(x.index)).to_dict()
    variable_to_mean_data = {}
    for x in variable_to_cells.keys():
            cells = variable_to_cells[x]
            if nr_cells_to_sample > len(cells):
                print(f"Number of cells to sample is greater than the number of cells annotated to {variable}, sampling {len(cells)} cells instead.")
                num_to_sample = len(cells)
            else:
                num_to_sample = nr_cells_to_sample
            for i in range(nr_pseudobulks_to_generate):
                sampled_cells = np.random.choice(
                    a = cells,
                    size = num_to_sample,
                    replace = False)
                variable_to_mean_data[f"{x}_{i}"] = data_matrix.loc[sampled_cells].mean(0)
    return pd.DataFrame(variable_to_mean_data).T

def calculate_correlation(
    A: pd.DataFrame,
    B: pd.DataFrame,
    mapping_A_to_B: dict,
    corr_function: Callable = pearsonr) -> pd.DataFrame:
    # Input validation
    if not all(A.index == B.index):
        raise ValueError("Index of A and B should match")
    common_features = set([mapping_A_to_B[f] for f in A.columns]) & set(B.columns)
    if len(common_features) == 0:
        raise ValueError("No features are common between A and B")
    mapping_B_to_A = {mapping_A_to_B[k]: k for k in mapping_A_to_B.keys()}
    correlations = []
    for B_feature in common_features:
        A_feature = mapping_B_to_A[B_feature]
        rho, p = corr_function(A[A_feature], B[B_feature])
        correlations.append(
            (A_feature, B_feature, rho, p))
    df_correlations = pd.DataFrame(correlations,
        columns = ["A", "B", "rho", "pval"]).fillna(0)
    df_correlations["pval_adj"] = p_adjust_bh(df_correlations["pval"])
    return df_correlations