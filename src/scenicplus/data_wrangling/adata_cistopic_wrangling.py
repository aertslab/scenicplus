import logging
import random
import sys
from random import sample
from typing import Any, Callable, Dict, Union

import numpy as np
import pandas as pd
from mudata import AnnData, MuData
from pycisTopic.cistopic_class import CistopicObject
from pycisTopic.diff_features import impute_accessibility

from scenicplus.utils import Groupby


def process_multiome_data(
        GEX_anndata: AnnData,
        cisTopic_obj: CistopicObject,
        use_raw_for_GEX_anndata: bool = True,
        imputed_acc_kwargs: Union[Dict[str, Any], None] =None,
        bc_transform_func: Callable = lambda x: x) -> MuData:
    """
    Format multi-ome data for SCENIC+ analysis.

    Parameters
    ----------
    GEX_anndata : AnnData
        AnnData object containing gene expression data.
    cisTopic_obj : CistopicObject
        cisTopic object containing chromatin accessibility data.
    use_raw_for_GEX_anndata : bool, optional
        Whether to use raw data for gene expression data, by default True.
    imputed_acc_kwargs : Mapping[str, Any], optional
        Arguments for impute_accessibility function, by default {"scale_factor": 10**6}.
    bc_transform_func : Callable, optional
        Function to transform barcodes from scRNA-seq to scATAC-seq,
        by default lambda x: x.

    Returns
    -------
    MuData
        MuData object containing gene expression and chromatin accessibility data.

    Raises
    ------
    Exception
        If no cells are found which are present in both assays.

    """
    # Create logger
    level = logging.INFO
    format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger("Ingesting multiome data")

    if imputed_acc_kwargs is None:
        imputed_acc_kwargs = {"scale_factor": 10**6}

    # TODO: .raw is deprecated, use layers instead.
    GEX_anndata = GEX_anndata.raw.to_adata() if use_raw_for_GEX_anndata else GEX_anndata

    # Use bc_transform_func to map scRNA-seq barcodes to scATAC-seq barcodes
    GEX_cell_names = [
        bc_transform_func(bc) for bc in GEX_anndata.obs_names]
    GEX_anndata.obs_names = GEX_cell_names
    ACC_cell_names = list(cisTopic_obj.cell_names.copy())

    # get cells with high quality (HQ cells) chromatin accessbility
    # AND gene expression profile
    common_cells = list(set(GEX_cell_names) & set(ACC_cell_names))
    if len(common_cells) == 0:
        raise Exception(
            "No cells found which are present in both assays, check input and consider using `bc_transform_func`!")
    log.info(f"Found {len(common_cells)} multiome cells.")

    # Subset and impute accessibility
    imputed_acc_obj = impute_accessibility(
        cisTopic_obj, selected_cells=common_cells, **imputed_acc_kwargs)
    # Subset and get gene expression data
    X_EXP = GEX_anndata[common_cells].X.copy()

    assert all(
        rna_bc == atac_bc
        for rna_bc, atac_bc in zip(
        GEX_anndata[common_cells].to_df().index, imputed_acc_obj.cell_names)
    ), \
            "rna and atac bc don't match"

    # Get and subset metadata from anndata and cisTopic object
    GEX_gene_metadata = GEX_anndata.var.copy(deep=True)
    GEX_cell_metadata = GEX_anndata.obs.copy(deep=True).loc[
        common_cells]
    ACC_region_metadata = pd.DataFrame(index = imputed_acc_obj.feature_names)
    ACC_region_metadata = ACC_region_metadata.merge(
        cisTopic_obj.region_data, left_index = True, right_index = True, how = "left")
    ACC_region_metadata = ACC_region_metadata.loc[imputed_acc_obj.feature_names]
    ACC_cell_metadata = cisTopic_obj.cell_data.copy(deep=True).loc[
        common_cells]

    # get cell dimensionality reductions
    GEX_dr_cell = {
        key: pd.DataFrame(
            index = GEX_cell_names,
            data=GEX_anndata.obsm[key]).loc[common_cells].to_numpy()
        for key in GEX_anndata.obsm}
    ACC_dr_cell = {
        key: cisTopic_obj.projections["cell"][key].loc[common_cells].to_numpy()
        for key in cisTopic_obj.projections["cell"]}

    mudata = MuData(
        {
            "scRNA": AnnData(
                X=X_EXP, obs=GEX_cell_metadata.infer_objects(),
                var=GEX_gene_metadata.infer_objects(), obsm=GEX_dr_cell),
            "scATAC": AnnData(
                X=imputed_acc_obj.mtx.T, obs=ACC_cell_metadata.infer_objects(),
                var=ACC_region_metadata.infer_objects(), obsm=ACC_dr_cell)
        }
    )

    return mudata

def _generate_pseudocells_for_numpy(
        X: np.ndarray,
        grouper: Groupby,
        nr_cells: list,
        nr_pseudobulks: list,
        axis=0
    ) -> np.ndarray:
    """Helper function to generate pseudocells for numpy array."""
    if len(nr_cells) != len(grouper.indices):
        raise ValueError(
            f"Length of nr_cells ({len(nr_cells)}) should be the same as length of grouper.indices ({len(grouper.indices)})")
    if len(nr_pseudobulks) != len(grouper.indices):
        raise ValueError(
            f"Length of nr_cells ({len(nr_pseudobulks)}) should be the same as length of grouper.indices ({len(grouper.indices)})")
    if axis == 0:
        shape_pseudo = (sum(nr_pseudobulks), X.shape[1])
    elif axis == 1:
        shape_pseudo = (X.shape[0], sum(nr_pseudobulks))
    else:
        raise ValueError(f"axis should be either 0 or 1 not {axis}")
    X_pseudo = np.zeros(shape=shape_pseudo)
    current_index = 0
    for idx, n_pseudobulk, n_cell in zip(grouper.indices, nr_pseudobulks, nr_cells):
        for x in range(n_pseudobulk):
            random.seed(x)
            sample_idx = sample(list(idx), n_cell)
            if axis == 0:
                sample_X = X[sample_idx, :]
            elif axis == 1:
                sample_X = X[:, sample_idx]
            mean_sample_X = sample_X.mean(axis=axis)
            if axis == 0:
                X_pseudo[current_index, :] = mean_sample_X
            elif axis == 1:
                X_pseudo[:, current_index] = mean_sample_X
            current_index += 1  # store index in X_pseudo where mean should be placed
    return X_pseudo

def _generate_pseudocell_names(
        grouper: Groupby,
        nr_pseudobulks: list,
        sep="_"
    ) -> list:
    """Helper function to generate pseudocell names."""
    if len(nr_pseudobulks) != len(grouper.indices):
        raise ValueError(
            f"Length of nr_cells ({len(nr_pseudobulks)}) should be the same as length of grouper.indices ({len(grouper.indices)})")
    names = []
    for n_pseudobulk, name in zip(nr_pseudobulks, grouper.keys):
        names.extend([name + sep + str(x) for x in range(n_pseudobulk)])

    return names

def process_non_multiome_data(
    GEX_anndata: AnnData,
    cisTopic_obj: CistopicObject,
    key_to_group_by: str,
    use_raw_for_GEX_anndata: bool = True,
    imputed_acc_kwargs: Union[Dict[str, Any], None] = None,
    nr_metacells: Union[None, int, Dict[str, int]] = None,
    nr_cells_per_metacells: Union[int, Dict[str, int]] = 10,
    meta_cell_split: str = "_") -> MuData:
    """
    Prepare non-multi-ome data for SCENIC+ analysis.

    Parameters
    ----------
    GEX_anndata : AnnData
        AnnData object containing gene expression data.
    cisTopic_obj : CistopicObject
        cisTopic object containing chromatin accessibility data.
    key_to_group_by : str
        Key to group cells by.
    use_raw_for_GEX_anndata : bool, optional
        Whether to use raw data for gene expression data, by default True.
    imputed_acc_kwargs : Dict[str, Any], optional
        Arguments for impute_accessibility function, by default {"scale_factor": 10**6}.
    nr_metacells : Union[None, int, Dict[str, int]], optional
        Number of metacells to generate, by default None.
    nr_cells_per_metacells : Union[int, Dict[str, int]], optional
        Number of cells per metacell, by default 10.
    meta_cell_split : str, optional
        Separator for metacell names, by default "_".

    Returns
    -------
    MuData
        MuData object containing gene expression and chromatin accessibility data.

    Raises
    ------
    ValueError
        If key_to_group_by is not found in GEX_anndata.obs.columns
        or cisTopic_obj.cell_data.columns.

    """
    # Create logger
    level = logging.INFO
    format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger("Ingesting non-multiome data")

    if imputed_acc_kwargs is None:
        imputed_acc_kwargs = {"scale_factor": 10**6}

    # TODO: .raw is deprecated, use layers instead.
    GEX_anndata = GEX_anndata.raw.to_adata() if use_raw_for_GEX_anndata else GEX_anndata

    # PROCESS NON-MULTI-OME DATA
    if key_to_group_by not in GEX_anndata.obs.columns:
        raise ValueError(
            f"key {key_to_group_by} not found in GEX_anndata.obs.columns")
    if key_to_group_by not in cisTopic_obj.cell_data.columns:
        raise ValueError(
            f"key {key_to_group_by} not found in cisTopic_obj.cell_data.columns")

    imputed_acc_obj = impute_accessibility(cisTopic_obj, **imputed_acc_kwargs)
    # check which annotations are common and if necessary subset
    common_annotations = list(
        set(GEX_anndata.obs[key_to_group_by].to_numpy()) & \
        set(cisTopic_obj.cell_data[key_to_group_by]))
    GEX_cells_to_keep = GEX_anndata.obs_names[np.isin(
        GEX_anndata.obs[key_to_group_by], common_annotations)]
    ACC_cells_to_keep = np.array(imputed_acc_obj.cell_names)[np.isin(
        cisTopic_obj.cell_data[key_to_group_by], common_annotations)]
    log.info(
        f'Following annotations were found in both assays under key {key_to_group_by}:\n\t{", ".join(common_annotations)}.\nKeeping {len(GEX_cells_to_keep)} cells for RNA and {len(ACC_cells_to_keep)} for ATAC.')
    imputed_acc_obj.subset(cells=ACC_cells_to_keep, copy=False)
    cisTopic_obj = cisTopic_obj.subset(cells=ACC_cells_to_keep, copy=True)
    GEX_anndata = GEX_anndata[GEX_cells_to_keep]

    # generate metacells
    grouper_EXP = Groupby(GEX_anndata.obs[key_to_group_by].to_numpy())
    grouper_ACC = Groupby(
        cisTopic_obj.cell_data[key_to_group_by].to_numpy())

    assert all(
        grouper_EXP.keys == grouper_ACC.keys), \
            "grouper_EXP.keys should be the same as grouper_ACC.keys"
    # this assertion is here because below we use only one of them
    # for a step which affects both assays

    if isinstance(nr_metacells, int):
        l_nr_metacells = [nr_metacells for i in range(
            len(common_annotations))]
    elif isinstance(nr_metacells, dict):
        # it is a mapping
        # for this we need the assertion above
        l_nr_metacells = [nr_metacells[k] for k in grouper_EXP.keys]
    elif nr_metacells is None:
        # automatically set this parameters
        if isinstance(nr_cells_per_metacells, int):
            l_nr_metacells = []
            for k in grouper_EXP.keys:  # for this we need the assertion above
                nr_cells_wi_annotation = min(sum(GEX_anndata.obs[key_to_group_by].to_numpy() == k),
                                                sum(cisTopic_obj.cell_data[key_to_group_by].to_numpy() == k))
                # using this formula each cell can be included in a metacell
                # on average 2 times
                l_nr_metacells.append(
                    (round(nr_cells_wi_annotation / nr_cells_per_metacells)) * 2)
        elif isinstance(nr_cells_per_metacells, dict):
            l_nr_metacells = []
            for k in grouper_EXP.keys:  # for this we need the assertion above
                nr_cells_wi_annotation = min(sum(GEX_anndata.obs[key_to_group_by].to_numpy() == k),
                                                sum(cisTopic_obj.cell_data[key_to_group_by].to_numpy() == k))
                n = nr_cells_per_metacells[k]
                # using this formula each cell can be included in a metacell
                # on average 2 times
                l_nr_metacells.append(
                    (round(nr_cells_wi_annotation / n)) * 2)
        else:
            raise TypeError("Wrong type for nr_cells_per_metacells, should be int or Dict[str, int]")
        log.info(
            f'Automatically set `nr_metacells` to: {", ".join([f"{k}: {n}" for k, n in zip(grouper_EXP.keys, l_nr_metacells)])}')
    else:
        raise TypeError("Wrong type for nr_metacells, should be None, int or Dict[str, int]")

    if isinstance(nr_cells_per_metacells, int):
        l_nr_cells = [nr_cells_per_metacells for i in range(
            len(common_annotations))]
    elif nr_cells_per_metacells is dict:
        # it is a mapping
        # for this we need the assertion above
        l_nr_cells = [nr_cells_per_metacells[k] for k in grouper_EXP.keys]
    else:
        raise TypeError("Wrong type for nr_cells_per_metacells, should be int or Dict[str, int]")

    log.info("Generating pseudo multi-ome data")
    meta_X_ACC = _generate_pseudocells_for_numpy(X=imputed_acc_obj.mtx if isinstance(imputed_acc_obj.mtx, np.ndarray) \
                                                 else imputed_acc_obj.mtx.toarray(),
                                                grouper=grouper_ACC,
                                                nr_cells=l_nr_cells,
                                                nr_pseudobulks=l_nr_metacells,
                                                axis=1)
    meta_cell_names_ACC = _generate_pseudocell_names(grouper=grouper_ACC,
                                                    nr_pseudobulks=l_nr_metacells,
                                                    sep=meta_cell_split)
    meta_X_EXP = _generate_pseudocells_for_numpy(X=GEX_anndata.X,
                                                grouper=grouper_EXP,
                                                nr_cells=l_nr_cells,
                                                nr_pseudobulks=l_nr_metacells,
                                                axis=0)
    meta_cell_names_EXP = _generate_pseudocell_names(grouper=grouper_EXP,
                                                    nr_pseudobulks=l_nr_metacells,
                                                    sep=meta_cell_split)

    assert meta_cell_names_ACC == meta_cell_names_EXP

    # generate cell metadata
    metadata_cell = pd.DataFrame(index=meta_cell_names_ACC,
                                    data={key_to_group_by: [x.split(meta_cell_split)[0] for x in meta_cell_names_ACC]})
    ACC_region_metadata = pd.DataFrame(index = imputed_acc_obj.feature_names)
    ACC_region_metadata = ACC_region_metadata.merge(
        cisTopic_obj.region_data, left_index = True, right_index = True, how = "left")
    ACC_region_metadata_subset = ACC_region_metadata.loc[imputed_acc_obj.feature_names]

    mudata = MuData(
        {
            "scRNA": AnnData(
                X=meta_X_EXP, obs=metadata_cell,
                var=GEX_anndata.var.copy(deep=True)),
            "scATAC": AnnData(
                X=meta_X_ACC.T, obs=metadata_cell,
                var=ACC_region_metadata_subset)
        }
    )

    return mudata
