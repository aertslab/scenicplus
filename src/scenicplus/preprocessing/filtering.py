from ..scenicplus_class import SCENICPLUS
import numpy  as np
import logging
import sys

level    = logging.INFO
format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
handlers = [logging.StreamHandler(stream=sys.stdout)]
logging.basicConfig(level = level, format = format, handlers = handlers)
log = logging.getLogger('Preprocessing')

def filter_genes(SCENICPLUS_obj: SCENICPLUS,
                 min_pct: int = 0,
                 max_pct: int = 100,
                 return_copy = False) -> SCENICPLUS:
    percent_of_cell_gene_expressed = np.array((SCENICPLUS_obj.X_EXP > 0).sum(0) / SCENICPLUS_obj.n_cells).flatten()
    genes_to_keep = SCENICPLUS_obj.gene_names[
        np.logical_and(percent_of_cell_gene_expressed > (min_pct / 100),
                       percent_of_cell_gene_expressed < (max_pct / 100))]
    log.info(f'Going from {SCENICPLUS_obj.n_genes} genes to {len(genes_to_keep)} genes.')
    if return_copy:
        return SCENICPLUS_obj.subset(genes = genes_to_keep, return_copy = return_copy)
    else:
        SCENICPLUS_obj.subset(genes = genes_to_keep, return_copy = return_copy)

def filter_regions(SCENICPLUS_obj: SCENICPLUS,
                   min_pct: int = 0,
                   max_pct: int = 100,
                   return_copy = False) -> SCENICPLUS:
    percent_of_cells_region_accessible = np.array((SCENICPLUS_obj.X_ACC > 0).sum(1) / SCENICPLUS_obj.n_cells).flatten()
    regions_to_keep = SCENICPLUS_obj.region_names[
        np.logical_and(percent_of_cells_region_accessible > (min_pct / 100),
                       percent_of_cells_region_accessible < (max_pct / 100))]
    log.info(f'Going from {SCENICPLUS_obj.n_regions} regions to {len(regions_to_keep)} regions.')
    if return_copy:
        return SCENICPLUS_obj.subset(regions = regions_to_keep, return_copy = return_copy)
    else:
        SCENICPLUS_obj.subset(regions = regions_to_keep, return_copy = return_copy)