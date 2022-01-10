import attr
from numpy.lib.function_base import iterable
import scipy.sparse as sparse
import pandas as pd
import numpy as np
from typing import Mapping, Any, Callable, Union
from pycisTopic.diff_features import CistopicImputedFeatures, impute_accessibility, normalize_scores
from pycisTopic.cistopic_class import CistopicObject
from scanpy import AnnData
import warnings
import logging
import sys

from scenicplus.utils import Groupby

#hardcoded variables
TOPIC_FACTOR_NAME = 'topic'

"""
Create a SCENIC (or SCENIC+, if allowed) class with:

Functions (Additional slots to fill in the object):
- Cistrome pruning

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
    
    Properties
    ----------
    n_cells
        Returns number of cells.
    n_genes
        Returns number of genes.
    n_regions
        Returns number of regions.
    cell_names
        Returns cell names
    gene_names
        Returns gene names
    region_names
        Returns region names
    to_df
        Returns a :class:`~pd.DataFame` containing gene expression or region accessbility data

    Functions
    --------
    add_cell_data
        Add cell metadata
    add_region_data
        Add region metadata
    add_gene_data
        Add gene metadata
    subset 
        Subset object
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

    # unstructured attributes like: region to gene, eregulons, ...
    uns = attr.ib(type = Mapping[str, Any], default = {})
    

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
        """
        Add cell metadata
        
        Parameters
        ----------
        cell_data
            A :class:`~pd.DataFrame` containing cell metdata indexed with cell barcodes.
        """
        if not set(self.cell_names) <= set(cell_data.index):
            Warning("`cell_data` does not contain metadata for all cells in :attr:`cell_names`. This wil result in NaNs.")
        
        columns_to_overwrite = list(set(self.metadata_cell.columns) & set(cell_data.columns))

        if len(columns_to_overwrite) > 0:
            Warning(f"Columns: {str(columns_to_overwrite)[1:-1]} will be overwritten.")
            self.metadata_cell.drop(columns_to_overwrite, axis = 1, inplace = True)

        common_cells = list(set(self.cell_names) & set(cell_data.index))

        self.metadata_cell = pd.concat([self.metadata_cell, cell_data.loc[common_cells]], axis = 1)

    def add_region_data(self, region_data: pd.DataFrame):
        """
        Add region metadata

        Parameters
        ----------
        region_data
            A :class:`~pd.DataFrame` containing region metadata indexed with region names.
        """
        if not set(self.region_names) <= set(region_data.index):
            Warning("`region_data` does not contain metadata for all regions in :attr:`region_names`. This wil result in NaNs.")
        
        columns_to_overwrite = list(set(self.metadata_regions.columns) & set(region_data.columns))

        if len(columns_to_overwrite) > 0:
            Warning(f"Columns: {str(columns_to_overwrite)[1:-1]} will be overwritten.")
            self.metadata_regions.drop(columns_to_overwrite, axis = 1, inplace = True)
        
        common_regions = list(set(self.region_names) & set(region_data.index))

        self.metadata_regions = pd.concat([self.metadata_regions, region_data.loc[common_regions]], axis = 1)

    def add_gene_data(self, gene_data: pd.DataFrame):
        """
        Add gene metadata

        Parameters
        ---------
        gene_data
            A :class:`~pd.DataFrame` containing gene metadata indexed with gene names.
        """
        if not set(self.gene_names) <= set(gene_data.index):
            Warning("`gene_data` does not contain metadata for all genes in :attr:`gene_names`. This wil result in NaNs.")
        
        columns_to_overwrite = list(set(self.metadata_genes.columns) & set(gene_data.columns))

        if len(columns_to_overwrite) > 0:
            Warning(f"Columns: {str(columns_to_overwrite)[1:-1]} will be overwritten.")
            self.metadata_genes.drop(columns_to_overwrite, axis = 1, inplace = True)
        
        common_genes = list(set(self.gene_names) & set(gene_data.index))

        self.metadata_genes = pd.concat([self.metadata_genes, gene_data.loc[common_genes]], axis = 1)
        
    def subset(self, cells = None, regions = None, genes = None, return_copy = False):
        """
        Subset object
        
        Parameters
        ----------
        cells
            A list of cells to keep
            default: None
        regions
            A list of regions to keep
            default: None
        genes
            A list of genes to keep
            default:None
        return_copy
            A boolean specifying wether to update the object (False) or return a copy (True)
        """
        def _subset(X, row_idx, col_idx):
            if type(X) == pd.core.frame.DataFrame:
                return X.iloc[row_idx, col_idx].copy()
            else:
                return X[row_idx, :][:, col_idx].copy()

        if cells is not None:
            #keep subset of cells
            cell_idx_to_keep = [self.cell_names.get_loc(cell) for cell in cells]
        else:
            #keep all cells
            cell_idx_to_keep = [self.cell_names.get_loc(cell) for cell in self.cell_names]

        if regions is not None:
            #keep subset of regions
            region_idx_to_keep = [self.region_names.get_loc(region) for region in regions]
        else:
            #keep all regions
            region_idx_to_keep = [self.region_names.get_loc(region) for region in self.region_names]

        if genes is not None:
            #keep subset of genes
            gene_idx_to_keep = [self.gene_names.get_loc(gene) for gene in genes]
        else:
            #keep all genes
            gene_idx_to_keep = [self.gene_names.get_loc(gene) for gene in self.gene_names]

        #subset gene expression and chromatin accessibility
        X_EXP_subset = _subset(self.X_EXP, cell_idx_to_keep, gene_idx_to_keep)
        X_ACC_subset = _subset(self.X_ACC, region_idx_to_keep, cell_idx_to_keep)

        #subset dimmensional reductions
        if self.dr_cell is not None:
            dr_cell_subset = {key: _subset(
                self.dr_cell[key], 
                cell_idx_to_keep, 
                [ i for i in range(self.dr_cell[key].shape[1]) ]) for key in self.dr_cell.keys()}
        else:
            dr_cell_subset = None
        if self.dr_region is not None:
            dr_region_subset = {key: _subset(
                self.dr_region[key],
                region_idx_to_keep,
                [ i for i in range(self.dr_region[key].shape[1]) ]) for key in self.dr_region.keys()}
        else:
            dr_region_subset = None
        
        #subset metadata
        metadata_cell_subset = self.metadata_cell.iloc[cell_idx_to_keep, :]
        metadata_gene_subset = self.metadata_genes.iloc[gene_idx_to_keep, :]
        metadata_region_subset = self.metadata_regions.iloc[region_idx_to_keep, :]

        if return_copy:
            return SCENICPLUS(
                X_ACC = X_ACC_subset,
                X_EXP = X_EXP_subset,
                metadata_regions = metadata_region_subset,
                metadata_genes = metadata_gene_subset,
                metadata_cell = metadata_cell_subset,
                menr = self.menr,
                dr_cell = dr_cell_subset,
                dr_region = dr_region_subset)
        else:
            self.X_ACC = X_ACC_subset
            self.X_EXP = X_EXP_subset
            self.metadata_regions = metadata_region_subset
            self.metadata_genes = metadata_gene_subset
            self.metadata_cell = metadata_cell_subset
            self.menr = self.menr
            self.dr_cell = dr_cell_subset
            self.dr_region = dr_region_subset

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
    multi_ome_mode: bool = True,
    nr_metacells: Union[int, Mapping[str, int]] = None,
    nr_cells_per_metacells: Union[int, Mapping[str, int]] = 10,
    meta_cell_split: str = '_',
    key_to_group_by: str = None,
    imputed_acc_obj: CistopicImputedFeatures = None,
    imputed_acc_kwargs: Mapping[str, Any] = {'scale_factor': 10**6},
    normalize_imputed_acc: bool = False,
    normalize_imputed_acc_kwargs: Mapping[str, Any] = {'scale_factor': 10 ** 4},
    cell_metadata: pd.DataFrame = None,
    region_metadata: pd.DataFrame = None,
    gene_metadata: pd.DataFrame = None,
    bc_transform_func: Callable = None,
    ACC_prefix: str = 'ACC_',
    GEX_prefix: str = 'GEX_') -> SCENICPLUS:
    """
    Function to create instances of :class:`SCENICPLUS`

    Parameters
    ----------
    GEX_anndata
        An instance of :class:`~sc.AnnData` containing gene expression data and metadata.
    cisTopic_obj
        An instance of :class:`pycisTopic.cistopic_class.CistopicObject` containing chromatin accessibility data and metadata.
    menr
        A dict mapping annotations to motif enrichment results
    multi_ome_mode
        A boolean specifying wether data is multi-ome (i.e. combined scATAC-seq and scRNA-seq from the same cell) or not
        default: True
    nr_metacells
        For non multi_ome_mode, use this number of meta cells to link scRNA-seq and scATAC-seq
        If this is a single integer the same number of metacells will be used for all annotations.
        This can also be a mapping between an annotation and the number of metacells per annotation.
        default: None
    nr_cells_per_metacells
        For non multi_ome_mode, use this number of cells per metacell to link scRNA-seq and scATAC-seq.
        If this is a single integer the same number of cells will be used for all annotations.
        This can also be a mapping between an annotation and the number of cells per metacell per annotation.
        default: 10
    meta_cell_split
        Character which is used as seperator in metacell names
        default: '_'
    key_to_group_by
        For non multi_ome_mode, use this cell metadata key to generate metacells from scRNA-seq and scATAC-seq. 
        Key should be common in scRNA-seq and scATAC-seq side
        default: None
    imputed_acc_obj
        An instance of :class:`~pycisTopic.diff_features.CistopicImputedFeatures` containing imputed chromatin accessibility.
        default: None
    imputed_acc_kwargs
        Dict with keyword arguments for imputed chromatin accessibility.
        default: {'scale_factor': 10**6}
    normalize_imputed_acc
        A boolean specifying wether or not to normalize imputed chromatin accessbility.
        default: False
    normalize_imputed_acc_kwargs
        Dict with keyword arguments for normalizing imputed accessibility.
        default: {'scale_factor': 10 ** 4}
    cell_metadata
        An instance of :class:`~pd.DataFrame` containing extra cell metadata
        default: None
    region_metadata
        An instance of :class:`~pd.DataFrame` containing extra region metadata
        default: None
    gene_metadata
        An instance of :class:`~pd.DataFrame` containing extra gene metadata
        default: None
    bc_transform_func
        A function used to transform gene expression barcode layout to chromatin accessbility layout.
        default: None
    ACC_prefix
        String prefix to add to cell metadata coming from :param:`cisTopic_obj`
        default: "ACC_"
    GEX_prefix
        String prefix to add to cell metadata coming from :param:`GEX_anndata`
        default: "GEX_"
    """
    # Create logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('create scenicplus object')

    GEX_gene_metadata = GEX_anndata.var.copy(deep = True)
    ACC_region_metadata = cisTopic_obj.region_data.copy(deep = True)
    if multi_ome_mode:
        #PROCESS DATA LIKE IT IS MULTI-OME
        GEX_cell_metadata = GEX_anndata.obs.copy(deep = True)
        GEX_cell_names = GEX_anndata.obs_names.copy(deep = True)
        if bc_transform_func is not None:
            #transform GEX barcodes to ACC barcodes
            GEX_cell_names = [bc_transform_func(bc) for bc in GEX_anndata.obs_names]
            GEX_cell_metadata.index = GEX_cell_names
        else:
            GEX_cell_names = list(GEX_cell_names)

        GEX_dr_cell = {k: pd.DataFrame(GEX_anndata.obsm[k].copy(), index = GEX_cell_names) for k in GEX_anndata.obsm.keys()}

        ACC_cell_names = list(cisTopic_obj.cell_names.copy())
        ACC_cell_metadata = cisTopic_obj.cell_data.copy(deep = True)
        
        if 'cell' in cisTopic_obj.projections.keys():
            ACC_dr_cell = cisTopic_obj.projections['cell'].copy()
        else:
            ACC_dr_cell = {}
        if 'region' in cisTopic_obj.projections.keys():
            ACC_dr_region = cisTopic_obj.projections['region'].copy()
        else:
            ACC_dr_region = {}

        #get cells with high quality (HQ cells) chromatin accessbility AND gene expression profile
        common_cells = list( set(GEX_cell_names) & set(ACC_cell_names) )

        if len(common_cells) == 0:
            raise Exception("No cells found which are present in both assays, check input and consider using `bc_transform_func`!")
        
        #impute accessbility if not given as parameter and subset for HQ cells
        if imputed_acc_obj is None:
            imputed_acc_obj = impute_accessibility(cisTopic_obj, selected_cells = common_cells, **imputed_acc_kwargs)
        else:
            imputed_acc_obj.subset(cells = common_cells)
        
        if normalize_imputed_acc:
            imputed_acc_obj = normalize_scores(imputed_acc_obj, **normalize_imputed_acc_kwargs)

        #subset gene expression data and metadata
        ACC_region_metadata_subset = ACC_region_metadata.loc[imputed_acc_obj.feature_names]

        GEX_cell_metadata_subset = GEX_cell_metadata.loc[common_cells]
        GEX_dr_cell_subset = {k: GEX_dr_cell[k].loc[common_cells] for k in GEX_dr_cell.keys()}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_EXP_subset = GEX_anndata[[GEX_cell_names.index(bc) for bc in common_cells], :].X.copy()

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
            dr_cell = dr_cell if len(dr_cell.keys()) > 0 else None,
            dr_region = ACC_dr_region if len(ACC_dr_region.keys()) > 0 else None)

    else:
        from .utils import generate_pseudocells_for_numpy, generate_pseudocell_names
        #PROCESS NON-MULTI-OME DATA
        if key_to_group_by not in GEX_anndata.obs.columns:
            raise ValueError(f'key {key_to_group_by} not found in GEX_anndata.obs.columns')
        if key_to_group_by not in cisTopic_obj.cell_data.columns:
            raise ValueError(f'key {key_to_group_by} not found in cisTopic_obj.cell_data.columns')

        #if imputed accessibility is not provided compute it
        if imputed_acc_obj is None:
            imputed_acc_obj = impute_accessibility(cisTopic_obj, **imputed_acc_kwargs)
        if normalize_imputed_acc:
            imputed_acc_obj = normalize_scores(imputed_acc_obj, **normalize_imputed_acc_kwargs)

        #check which annotations are common and if necessary subset
        common_annotations = list(set(GEX_anndata.obs[key_to_group_by].to_numpy()) & set(cisTopic_obj.cell_data[key_to_group_by]))
        GEX_cells_to_keep = GEX_anndata.obs_names[np.isin(GEX_anndata.obs[key_to_group_by], common_annotations)]
        ACC_cells_to_keep = np.array(imputed_acc_obj.cell_names)[np.isin(cisTopic_obj.cell_data[key_to_group_by], common_annotations)]
        log.info(f'Following annotations were found in both assays under key {key_to_group_by}:\n\t{", ".join(common_annotations)}.\nKeeping {len(GEX_cells_to_keep)} cells for RNA and {len(ACC_cells_to_keep)} for ATAC.')
        imputed_acc_obj.subset(cells = ACC_cells_to_keep, copy = False)
        cisTopic_obj.subset(cells = ACC_cells_to_keep, copy = False)
        GEX_anndata = GEX_anndata[GEX_cells_to_keep]

        #generate metacells
        grouper_EXP = Groupby(GEX_anndata.obs[key_to_group_by].to_numpy())
        grouper_ACC = Groupby(cisTopic_obj.cell_data[key_to_group_by].to_numpy())

        assert all(grouper_EXP.keys == grouper_ACC.keys), 'grouper_EXP.keys should be the same as grouper_ACC.keys'
        #this assertion is here because below we use only one of them for a step which affects both assays

        if type(nr_metacells) is int:
            l_nr_metacells = [nr_metacells for i in range(len(common_annotations))]
        elif nr_metacells is not None:
            #it is a mapping
            l_nr_metacells = [nr_metacells[k] for k in grouper_EXP.keys] #for this we need the assertion above
        elif nr_metacells is None:
            #automatically set this parameters
            if type(nr_cells_per_metacells) is int:
                l_nr_metacells = []
                for k in grouper_EXP.keys: #for this we need the assertion above
                    nr_cells_wi_annotation = min(sum(GEX_anndata.obs[key_to_group_by].to_numpy() == k),
                                                 sum(cisTopic_obj.cell_data[key_to_group_by].to_numpy() == k))
                    #using this formula each cell can be included in a metacell on average 2 times
                    l_nr_metacells.append( (round(nr_cells_wi_annotation / nr_cells_per_metacells)) * 2 )
            elif nr_cells_per_metacells is not None:
                l_nr_metacells = []
                for k in grouper_EXP.keys: #for this we need the assertion above
                    nr_cells_wi_annotation = min(sum(GEX_anndata.obs[key_to_group_by].to_numpy() == k),
                                                 sum(cisTopic_obj.cell_data[key_to_group_by].to_numpy() == k))
                    n = nr_cells_per_metacells[k]
                    #using this formula each cell can be included in a metacell on average 2 times
                    l_nr_metacells.append( (round(nr_cells_wi_annotation / n)) * 2 )
            log.info(f'Automatically set `nr_metacells` to: {", ".join([f"{k}: {n}" for k, n in zip(grouper_EXP.keys, l_nr_metacells)])}')
        
        if type(nr_cells_per_metacells) is int:
            l_nr_cells = [nr_cells_per_metacells for i in range(len(common_annotations))]
        elif nr_cells_per_metacells is not None:
            #it is a mapping
            l_nr_cells = [nr_cells_per_metacells[k] for k in grouper_EXP.keys] #for this we need the assertion above
        
        log.info('Generating pseudo multi-ome data')
        meta_X_ACC = generate_pseudocells_for_numpy(X = imputed_acc_obj.mtx if isinstance(imputed_acc_obj.mtx, np.ndarray) else imputed_acc_obj.mtx.toarray(),
                                                    grouper = grouper_ACC,
                                                    nr_cells = l_nr_cells,
                                                    nr_pseudobulks = l_nr_metacells,
                                                    axis = 1)
        meta_cell_names_ACC = generate_pseudocell_names(grouper = grouper_ACC,
                                                        nr_pseudobulks = l_nr_metacells,
                                                        sep = meta_cell_split)
        meta_X_EXP = generate_pseudocells_for_numpy(X = GEX_anndata.X,
                                                    grouper = grouper_EXP,
                                                    nr_cells = l_nr_cells,
                                                    nr_pseudobulks = l_nr_metacells,
                                                    axis = 0)
        meta_cell_names_EXP = generate_pseudocell_names(grouper = grouper_EXP,
                                                        nr_pseudobulks = l_nr_metacells,
                                                        sep = meta_cell_split)

        assert meta_cell_names_ACC == meta_cell_names_EXP

        #generate cell metadata
        metadata_cell = pd.DataFrame(index = meta_cell_names_ACC, 
                                     data = {key_to_group_by: [x.split(meta_cell_split)[0] for x in meta_cell_names_ACC]})
        
        #create the object
        ACC_region_metadata_subset = ACC_region_metadata.loc[imputed_acc_obj.feature_names]
        SCENICPLUS_obj = SCENICPLUS(
            X_ACC = meta_X_ACC,
            X_EXP = meta_X_EXP,
            metadata_regions = ACC_region_metadata_subset,
            metadata_genes = GEX_gene_metadata,
            metadata_cell = metadata_cell,
            menr = menr)


    if region_metadata is not None:
        SCENICPLUS_obj.add_region_data(region_metadata)
    
    if gene_metadata is not None:
        SCENICPLUS_obj.add_gene_data(gene_metadata)
    
    if cell_metadata is not None:
        SCENICPLUS_obj.add_cell_data(cell_metadata)
    
    return SCENICPLUS_obj

