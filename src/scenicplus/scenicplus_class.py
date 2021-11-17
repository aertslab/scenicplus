import attr
from numpy.lib.function_base import iterable
import scipy.sparse as sparse
import pandas as pd
import numpy as np
from typing import Union, List, Mapping, Any, Callable
from pycisTopic.diff_features import CistopicImputedFeatures, impute_accessibility
from pycisTopic.cistopic_class import CistopicObject
from scanpy import AnnData
import warnings

#hardcoded variables
TOPIC_FACTOR_NAME = 'topic'

"""
Create a SCENIC (or SCENIC+, if allowed) class with:

Functions (Additional slots to fill in the object):
- Filtering lowly accessible regions/genes (OPTIONAL)
- Cistrome pruning
- Region2gene links
- eGRN (GSEA)

Exploratory functions (after running pipeline)
- Plot region2gene (with option to give custom bigwigs)
- Dot płot
- eGRN
- Embeddings (plot metadata, genes, regions,…)
- Combinations TF
- Cytoscape
- Once we have network, predict perturbation effect (boolean modelling?)

Feel free to comment/edit/add extra things :)!!
"""


def _check_dimmensions(instance, attribute, value):
    if attribute.name == 'X_ACC':
        if not value.shape[1] == instance.X_EXP.shape[0]:
            raise ValueError(
                "RNA and ATAC matrix should have the same number of cells."
                f" RNA has {instance.X_EXP.shape[0]} number of cells and ATAC has {value.shape[1]} number of cells.")
    if attribute.name == 'X_EXP':
        if not value.shape[0] == instance.X_ACC.shape[1]:
            raise ValueError(
                "RNA and ATAC matrix should have the same number of cells."
                f" RNA has {value.shape[0]} number of cells and ATAC has {instance.X_ACC.shape[1]} number of cells.")

@attr.s(repr=False)
class SCENICPLUS():
    """
    An object containing: gene expression, chromatin accessbility and motif enrichment data.

    :class:`~scenicplus_class.SCENICPLUS` stores the data matrices :attr:`X_ACC` (chromatin accessbility) and
    :attr:`X_EXP` (gene expression) together with region annotation :attr:`metadata_regions`, gene annotation :attr:`metadata_genes`,
    cell annotation :attr:`metadata_cell` and motif enrichment data :attr:`menr`.

    Parameters
    ----------
    X_ACC
        A #regions x #cells data matrix
    X_EXP
        A #cells x #genes data matrix
    metadata_regions
        A :class:`pandas.DataFrame` containing region metadata annotation of length #regions
    metadata_genes
        A :class:`pandas.DataFrame` containing gene metadata annotation of length #genes
    metadata_cell
        A :class:`pandas.DataFrame` containing cell metadata annotation of lenght #cells
    menr
        A Dict containing motif enrichment results for topics of differentially accessbile regions (DARs), generate by pycistarget.
        Should take the form {'region_set_name1': {region_set1: result, 'region_set2': result}, 
                              'region_set_name2': {'region_set1': result, 'region_set2': result},
                              'topic': {'topic1': result, 'topic2': result}} 
        region set names, which aren't topics, should be columns in the :attr:`metadata_cell` dataframe
    dr_cell
        A Dict containing dimmensional reduction coordinates of cells.
    dr_region
        A Dict containing dimmensional reduction coordinates of regions.
    """

    #mandatory attributes
    X_ACC            = attr.ib(type = Union[sparse.spmatrix, np.ndarray, pd.DataFrame],
                               validator = _check_dimmensions)
    X_EXP            = attr.ib(type = Union[sparse.spmatrix, np.ndarray, pd.DataFrame],
                               validator = _check_dimmensions)
    metadata_regions = attr.ib(type = pd.DataFrame)
    metadata_genes   = attr.ib(type = pd.DataFrame)
    metadata_cell    = attr.ib(type = pd.DataFrame)
    menr             = attr.ib(type = Mapping[str, Mapping[str, Any]])
    
    #optional attributes
    dr_cell         = attr.ib(type = Mapping[str, iterable], default = None)
    dr_region       = attr.ib(type = Mapping[str, iterable], default = None)
    # unstructured attributes like: dimensional reduction, region to gene, eregulons, ...

    #validation
    @metadata_regions.validator
    def check_n_regions(self, attribute, value):
        if not len(value) == self.n_regions:
            raise ValueError(
                "`metadata_regions` must have the same number of annotations as rows in `X_ACC`"
                f" ({self.n_regions}), but has {len(value)} rows.")
    
    @metadata_genes.validator
    def check_n_genes(self, attribute, value):
        if not len(value) == self.n_genes:
            raise ValueError(
                "`metadata_genes` must have the same number of annotations as columns in `X_EXP`"
                f" ({self.n_genes}), but has {len(value)} rows.")
    
    @metadata_cell.validator
    def check_n_cells(self, attribute, value):
        if not len(value) == self.n_cells:
            raise ValueError(
                "`metadata_cell` must have the same number of cells as rows in `X_EXP` and columns `X_ACC`"
                f" ({self.n_cells}), but has {len(value)} rows.")

    @menr.validator
    def check_keys(self, attribute, value):
        #check wether all keys, except topic, of the motif enrichment dictionary are in the columns of the metadata_cell dataframe
        if not ( set(value.keys()) - set([TOPIC_FACTOR_NAME]))<= set(self.metadata_cell.columns):
            not_found = set(value.keys()) - set(self.metadata_cell.columns)
            raise ValueError(
                "All keys in `menr` (except {TOPIC_FACTOR_NAME}) should be factors in `metadata_cell`"
                f" the keys: {not_found} are not found in `metadata_cell`")
    
    @menr.validator
    def check_levels(self, attribute, value):
        #check wether for each key, except topic, of the motif enrichment dictionary its keys are levels in the metadata_cell dataframe
        if not all([set(value[s].keys()) <= set(self.metadata_cell[s]) for s in set(value.keys()) - set([TOPIC_FACTOR_NAME])]):
            raise ValueError(
                f"For each key (except {TOPIC_FACTOR_NAME}) of `menr` its keys should be levels in `metadata_cell` under the same key.")

    @dr_cell.validator
    def check_cell_dimmensions(self, attribute, value):
        if value is not None:
            if not all([value[k].shape[0] == self.n_cells for k in value.keys()]):
                raise ValueError(
                    f"Cell dimmensional reductions `dr_cell` should have :attr:`n_cells`{self.n_cells} entries along its 0th dimmension."
                )
    @dr_region.validator
    def check_region_dimmensions(self, attribute, value):
        if value is not None:
            if not all([value[k].shape[0] == self.n_regions for k in value.keys()]):
                raise ValueError(
                    f"Region dimmensional reductions `dr_region` should have :attr:`n_regions`{self.n_regions} entries along its 0th dimmension."
                )
    #properties:
    @property
    def n_cells(self):
        # we already checked that both RNA and ATAC have the same number of cells so we can return the number of cells from RNA
        return self.X_EXP.shape[0]

    @property
    def n_genes(self):
        return self.X_EXP.shape[1]
    
    @property
    def n_regions(self):
        return self.X_ACC.shape[0]
    
    @property
    def cell_names(self):
        return self.metadata_cell.index
    
    @property
    def gene_names(self):
        return self.metadata_genes.index
    
    @property
    def region_names(self):
        return self.metadata_regions.index
    
    def to_df(self, layer) -> pd.DataFrame:
        """
        Generate a :class:`~pandas.DataFrame`.

        The data matrix :attr:`X_EXP` or :attr:`X_ACC` is returned as
        :class:`~pandas.DataFrame`, with :attr:`cell_names` as index,
        and :attr:`gene_names` or :attr:`region_names` as columns.

        Parameters
        ----------
        layer: str
            ACC to return accessibility data and EXP to return expression data.
        """

        if not layer in ['ACC', 'EXP']:
            raise ValueError(
                f"`layer` should be either `ACC` or `EXP`, not {layer}.")
            
        if layer == 'ACC':
            X = self.X_ACC
            idx = self.region_names
            cols = self.cell_names
        elif layer == 'EXP':
            X = self.X_EXP
            cols = self.gene_names
            idx = self.cell_names
        
        if sparse.issparse(X):
            X = X.toarray()
        
        return pd.DataFrame(X, index = idx, columns = cols)
    
    #The three functions below can probably be combined in a single function.
    def add_cell_data(self, cell_data: pd.DataFrame):
        if not set(self.cell_names) <= set(cell_data.index):
            Warning("`cell_data` does not contain metadata for all cells in :attr:`cell_names`. This wil result in NaNs.")
        
        columns_to_overwrite = list(set(self.metadata_cell.columns) & set(cell_data.columns))

        if len(columns_to_overwrite) > 0:
            Warning(f"Columns: {str(columns_to_overwrite)[1:-1]} will be overwritten.")
            self.metadata_cell.drop(columns_to_overwrite, axis = 1, inplace = True)

        common_cells = list(set(self.cell_names) & set(cell_data.index))

        self.metadata_cell = pd.concat([self.metadata_cell, cell_data.loc[common_cells]], axis = 1)

    def add_region_data(self, region_data: pd.DataFrame):
        if not set(self.region_names) <= set(region_data.index):
            Warning("`region_data` does not contain metadata for all regions in :attr:`region_names`. This wil result in NaNs.")
        
        columns_to_overwrite = list(set(self.metadata_regions.columns) & set(region_data.columns))

        if len(columns_to_overwrite) > 0:
            Warning(f"Columns: {str(columns_to_overwrite)[1:-1]} will be overwritten.")
            self.metadata_regions.drop(columns_to_overwrite, axis = 1, inplace = True)
        
        common_regions = list(set(self.region_names) & set(region_data.index))

        self.metadata_regions = pd.concat([self.metadata_regions, region_data.loc[common_regions]], axis = 1)

    def add_gene_data(self, gene_data: pd.DataFrame):
        if not set(self.gene_names) <= set(gene_data.index):
            Warning("`gene_data` does not contain metadata for all genes in :attr:`gene_names`. This wil result in NaNs.")
        
        columns_to_overwrite = list(set(self.metadata_genes.columns) & set(gene_data.columns))

        if len(columns_to_overwrite) > 0:
            Warning(f"Columns: {str(columns_to_overwrite)[1:-1]} will be overwritten.")
            self.metadata_genes.drop(columns_to_overwrite, axis = 1, inplace = True)
        
        common_genes = list(set(self.gene_names) & set(gene_data.index))

        self.metadata_genes = pd.concat([self.metadata_genes, gene_data.loc[common_genes]], axis = 1)
        
    
    def __repr__(self) -> str:
        #inspired by AnnData
        descr = f"SCENIC+ object with n_cells x n_genes = {self.n_cells} x {self.n_genes} and n_cells x n_regions = {self.n_cells} x {self.n_regions}"
        for attr in [
            "metadata_regions",
            "metadata_genes",
            "metadata_cell",
            "menr",
            "dr_cell",
            "dr_region"
        ]:
            try:
                keys = getattr(self, attr).keys()
                if len(keys) > 0:
                    descr += f"\n\t{attr}:{str(list(keys))[1:-1]}"
            except:
                continue
        return descr
    

def create_SCENICPLUS_object(
    GEX_anndata: AnnData,
    cisTopic_obj: CistopicObject,
    menr: Mapping[str, Mapping[str, Any]],
    imputed_acc_obj: CistopicImputedFeatures = None,
    imputed_acc_kwargs: Mapping[str, Any] = {'scale_factor': 10**6},
    cell_metadata: pd.DataFrame = None,
    region_metadata: pd.DataFrame = None,
    gene_metadata: pd.DataFrame = None,
    bc_transform_func: Callable = lambda x: x.replace('-1___', '-1-').rsplit('__', 1)[0],
    ACC_prefix: str = 'ACC_',
    GEX_prefix: str = 'GEX_'
) -> SCENICPLUS:
    GEX_cell_metadata = GEX_anndata.obs.copy(deep = True)
    GEX_gene_metadata = GEX_anndata.var.copy(deep = True)
    GEX_cell_names = GEX_anndata.obs_names.copy(deep = True)
    if bc_transform_func is not None:
        #transform GEX barcodes to ACC barcodes
        GEX_cell_names = [bc_transform_func(bc) for bc in GEX_anndata.obs_names]
        GEX_cell_metadata.index = GEX_cell_names
    GEX_dr_cell = {k: pd.DataFrame(GEX_anndata.obsm[k].copy(), index = GEX_cell_names) for k in GEX_anndata.obsm.keys()}

    ACC_cell_names = list(cisTopic_obj.cell_names.copy())
    ACC_cell_metadata = cisTopic_obj.cell_data.copy(deep = True)
    ACC_region_metadata = cisTopic_obj.region_data.copy(deep = True)
    ACC_dr_cell = cisTopic_obj.projections['cell'].copy()
    ACC_dr_region = cisTopic_obj.projections['region'].copy()

    #get cells with high quality (HQ cells) chromatin accessbility AND gene expression profile
    common_cells = list( set(GEX_cell_names) & set(ACC_cell_names) )

    if len(common_cells) == 0:
        raise ValueError("No cells found which are present in both assays, check input and consider using `bc_transform_func`!")
    
    #impute accessbility if not given as parameter and subset for HQ cells
    if imputed_acc_obj is None:
        imputed_acc_obj = impute_accessibility(cisTopic_obj, selected_cells = common_cells, **imputed_acc_kwargs)
    else:
        imputed_acc_obj.subset(cells = common_cells)

    #subset gene expression data and metadata
    ACC_region_metadata_subset = ACC_region_metadata.loc[imputed_acc_obj.feature_names]

    GEX_cell_metadata_subset = GEX_cell_metadata.loc[common_cells]
    GEX_dr_cell_subset = {k: GEX_dr_cell[k].loc[common_cells] for k in GEX_dr_cell.keys()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_EXP_subset = GEX_anndata[[bc in common_cells for bc in GEX_cell_names], :].X.copy()

    ACC_cell_metadata_subset = ACC_cell_metadata.loc[common_cells]
    ACC_dr_cell_subset = {k: ACC_dr_cell[k].loc[common_cells] for k in ACC_dr_cell.keys()}
    X_ACC_subset = imputed_acc_obj.mtx

    #add prefixes
    if GEX_prefix is not None:
        GEX_cell_metadata_subset.columns = [GEX_prefix + colname for colname in GEX_cell_metadata_subset.columns]
        GEX_dr_cell_subset = {GEX_prefix + k: GEX_dr_cell_subset[k] for k in GEX_dr_cell_subset.keys()}
    
    if ACC_prefix is not None:
        ACC_cell_metadata_subset.columns = [ACC_prefix + colname for colname in ACC_cell_metadata_subset.columns]
        ACC_dr_cell_subset = {ACC_prefix + k: ACC_dr_cell_subset[k] for k in ACC_dr_cell_subset.keys()}
    
    #concatenate cell metadata and cell dimmensional reductions
    ACC_GEX_cell_metadata = pd.concat([GEX_cell_metadata_subset, ACC_cell_metadata_subset], axis = 1)
    dr_cell = {**GEX_dr_cell_subset, **ACC_dr_cell_subset}

    SCENICPLUS_obj = SCENICPLUS(
        X_ACC = X_ACC_subset,
        X_EXP = X_EXP_subset,
        metadata_regions = ACC_region_metadata_subset,
        metadata_genes = GEX_gene_metadata,
        metadata_cell = ACC_GEX_cell_metadata,
        menr = menr,
        dr_cell = dr_cell,
        dr_region = ACC_dr_region if len(ACC_dr_region.keys()) > 0 else None)
    
    if region_metadata is not None:
        SCENICPLUS_obj.add_region_data(region_metadata)
    
    if gene_metadata is not None:
        SCENICPLUS_obj.add_gene_data(gene_metadata)
    
    if cell_metadata is not None:
        SCENICPLUS_obj.add_cell_data(cell_metadata)
    
    return SCENICPLUS_obj