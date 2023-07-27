from mudata import MuData
from scenicplus.scenicplus_mudata import ScenicPlusMuData
from typing import Union
import numpy as np
import pandas as pd

def generate_pseudobulks(
        scplus_mudata: Union[MuData, ScenicPlusMuData],
        variable: str,
        modality: str,
        nr_cells_to_sample: int,
        nr_pseudobulks_to_generate: int,
        seed: int,
        normalize_data: bool = False):
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

#TODO: fix multiple uses of pandas concat (generates a lot of warning)
def TF_cistrome_correlation(scplus_obj: SCENICPLUS,
                            variable: str = None,
                            use_pseudobulk: bool = True,
                            auc_key: str = 'Cistromes_AUC',
                            signature_key: str = 'Unfiltered',
                            out_key: str = 'Unfiltered',
                            subset: List[str] = None):
    """
    Get correlation between gene expression and cistrome accessibility

    Parameters
    ---------
    scplus_obj: :class:`SCENICPLUS`
        A :class:`SCENICPLUS` object with pseudobulk matrices (`scplus_obj.uns['Pseudobulk']`)
    variable: str, optional
        Variable used to create the pseudobulks. Must be a key in `scplus_obj.uns['Pseudobulk']`.
        Only required if use_pseudobulk is False.
    use_pseudobulk: bool, optional
        Whether to use pseudobulk matrix or actual values.
    cistromes_key: str, optional
        Key to retrieve the pseudobulk matrices.  Cistrome accessibility will be retrieved from
        `scplus_obj.uns['Pseudobulk'][variable]['Cistromes_AUC'][cistromes_key]` and 
        gene expression from `scplus_obj.uns['Pseudobulk'][variable]['Expression']`.
    out_key : str, optional
        Ouput key. Correlations will be stored at `scplus_obj.uns['TF_cistrome_correlation'][variable][out_key]`.
    subset: List, optional
        Subset of cells to be used to calculate correlations. Default: None (All)
    """
    if use_pseudobulk:
        dgem_agg = scplus_obj.uns['Pseudobulk'][variable]['Expression']
        cistromes_auc_agg = scplus_obj.uns['Pseudobulk'][variable][auc_key][signature_key]
    else:
        dgem_agg = pd.DataFrame(
            scplus_obj.X_EXP, index=scplus_obj.cell_names, columns=scplus_obj.gene_names).copy().T
        cistromes_auc_agg = scplus_obj.uns[auc_key][signature_key].copy().T

    if subset is not None:
        cell_data = pd.DataFrame([x.rsplit('_', 1)[0] for x in cistromes_auc_agg.columns],
                                 index=cistromes_auc_agg.columns).iloc[:, 0]
        subset_cells = cell_data[cell_data.isin(subset)].index.tolist()
        cistromes_auc_agg = cistromes_auc_agg.loc[:, subset_cells]
        dgem_agg = dgem_agg.loc[:, subset_cells]
    corr_df = pd.DataFrame(columns=['TF', 'Cistrome', 'Rho', 'P-value'])
    for tf in cistromes_auc_agg.index:
        # Handle _extended
        tf_rna = tf.split('_')[0]
        if tf_rna in dgem_agg.index:
            cistromes_auc_tf = cistromes_auc_agg.loc[tf, :]
            tf_expr = dgem_agg.loc[tf_rna, :]
            # Exception in case TF is only expressed in 1 cell
            # TFs expressed in few cells could be filtered too
            try:
                corr_1, _1 = pearsonr(tf_expr, cistromes_auc_tf)
                x = {'TF': tf_rna,
                     'Cistrome': tf,
                     'Rho': corr_1,
                     'P-value': _1}
                corr_df = corr_df.append(pd.DataFrame(
                    data=x, index=[0]), ignore_index=True)
            except:
                continue
    corr_df = corr_df.dropna()
    corr_df['Adjusted_p-value'] = p_adjust_bh(corr_df['P-value'])

    if not 'TF_cistrome_correlation' in scplus_obj.uns.keys():
        scplus_obj.uns['TF_cistrome_correlation'] = {}
    if not out_key in scplus_obj.uns['TF_cistrome_correlation'].keys():
        scplus_obj.uns['TF_cistrome_correlation'][out_key] = {}
    scplus_obj.uns['TF_cistrome_correlation'][out_key] = corr_df

#TODO: fix multiple uses of pandas concat (generates a lot of warning)
def eregulon_correlation(scplus_obj: SCENICPLUS,
                         auc_key: str = 'eRegulon_AUC',
                         signature_key1: str = 'Gene_based',
                         signature_key2: str = 'Region_based',
                         nSignif: int = 3,
                         out_key: str = 'Unfiltered',
                         subset_cellids: List[str] = None
                        ):
        """
        Get correlation between gene-based and region-based eRegulon AUC

        Parameters
        ---------
        scplus_obj: :class:`SCENICPLUS`
            A :class:`SCENICPLUS` object
        auc_key: str, optional
            Name of the AUC matrix used to calculate the correlation, normally 'eRegulon_AUC'. 
            Must be a key in `scplus_obj.uns.keys()`.
        signature_key1: str, optional
            Variable used to calculate the correlation, normally 'Gene_based'. 
            Must be a key in `scplus_obj.uns[auc_key].keys()`.
        signature_key2: str, optional
            Variable used to calculate the correlation, normally 'Region_based'. 
            Must be a key in `scplus_obj.uns[auc_key].keys()`.
        nSignif: str, optional
            Number of digits to save.
        out_key : str, optional
            Ouput key. Correlations will be stored at `scplus_obj.uns['eRegulon_correlation'][out_key]`.
        subset_cellids: List, optional
            Subset of cells to be used to calculate correlations. Default: None (All)
        """

        gene_auc = scplus_obj.uns[auc_key][signature_key1].copy().T
        region_auc = scplus_obj.uns[auc_key][signature_key2].copy().T

        if subset_cellids is not None:
            cell_data = pd.DataFrame([x.rsplit('_', 1)[0] for x in gene_auc.columns],
                                     index=gene_auc.columns).iloc[:, 0]
            subset_cells = cell_data[cell_data.isin(subset_cellids)].index.tolist()
            gene_auc = gene_auc.loc[:, subset_cells]
            region_auc = region_auc.loc[:, subset_cells]

        # cistrome naming includes number of genes/regions, so need to create matching names
        # x.rsplit('_', 1) splits at first _ from the right
        gene_auc['id_short'] = gene_auc.index.map(lambda x: x.rsplit('_', 1)[0])
        gene_auc['id_full'] = gene_auc.index
        gene_auc = gene_auc.set_index('id_short')

        region_auc['id_short'] = region_auc.index.map(lambda x: x.rsplit('_', 1)[0])
        region_auc['id_full'] = region_auc.index
        region_auc = region_auc.set_index('id_short')

        corr_df = pd.DataFrame(columns=['id', signature_key1, signature_key2, 'Rho', 'P-value'])

        for tf in gene_auc.index:
            # All TFs should match, but just in case
            if tf in region_auc.index:
                # record orginal cistrome name for results
                signature1_id = gene_auc.loc[tf, 'id_full']
                signature2_id = region_auc.loc[tf, 'id_full']
                # Exception in case TF is only expressed in 1 cell
                # TFs expressed in few cells could be filtered too
                try:
                    corr_1, _1 = pearsonr(gene_auc.loc[tf, gene_auc.columns != 'id_full'],
                                          region_auc.loc[tf, gene_auc.columns != 'id_full'])
                    x = {'id': tf,
                         signature_key1: signature1_id,
                         signature_key2: signature2_id,
                         'Rho': round(corr_1,nSignif),
                         'P-value': _1}
                    corr_df = pd.concat([corr_df,
                                         pd.DataFrame(data=x, index=[0])],
                                        ignore_index=True)
                except:
                    continue
        corr_df = corr_df.dropna()
        corr_df['Adjusted_pValue'] = p_adjust_bh(corr_df['P-value'])
        corr_df['Abs_rho'] = abs(corr_df['Rho'])
        corr_df.sort_values('Abs_rho', ascending=False, inplace=True)

        if not 'eRegulon_correlation' in scplus_obj.uns.keys():
            scplus_obj.uns['eRegulon_correlation'] = {}
        if not out_key in scplus_obj.uns['eRegulon_correlation'].keys():
            scplus_obj.uns['eRegulon_correlation'][out_key] = {}
        scplus_obj.uns['eRegulon_correlation'][out_key] = corr_df