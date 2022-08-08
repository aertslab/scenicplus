"""Filter outlier genes and regions.

"""

from ..scenicplus_class import SCENICPLUS
from ..eregulon_enrichment import get_eRegulons_as_signatures
import numpy as np
import logging
import sys

level = logging.INFO
format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
handlers = [logging.StreamHandler(stream=sys.stdout)]
logging.basicConfig(level=level, format=format, handlers=handlers)
log = logging.getLogger('Preprocessing')


def filter_genes(SCENICPLUS_obj: SCENICPLUS,
                 min_pct: int = 0,
                 max_pct: int = 100,
                 return_copy=False) -> SCENICPLUS:
    """
    Filter scenciplus object genes

    Parameters
    ----------
    SCENICPLUS_obj
        An instance of :class: `~scenicplus.scenicplus_class.SCENICPLUS`.
    min_pct
        only keep genes which are expressed in at least `min_pct` of cells.
        default: 0
    max_pct
        only keep genes which are expressed in maximal `max_pct` of cells.
        default: 100
    return_copy
        If set to True a new SCENICPLUS object will be generated containing filtered data.
        default: False
    """
    percent_of_cell_gene_expressed = np.array(
        (SCENICPLUS_obj.X_EXP > 0).sum(0) / SCENICPLUS_obj.n_cells).flatten()
    genes_to_keep = SCENICPLUS_obj.gene_names[
        np.logical_and(percent_of_cell_gene_expressed > (min_pct / 100),
                       percent_of_cell_gene_expressed < (max_pct / 100))]
    log.info(
        f'Going from {SCENICPLUS_obj.n_genes} genes to {len(genes_to_keep)} genes.')
    if return_copy:
        return SCENICPLUS_obj.subset(genes=genes_to_keep, return_copy=return_copy)
    else:
        SCENICPLUS_obj.subset(genes=genes_to_keep, return_copy=return_copy)


def filter_regions(SCENICPLUS_obj: SCENICPLUS,
                   min_pct: int = 0,
                   max_pct: int = 100,
                   return_copy=False) -> SCENICPLUS:
    """
    Filter scenciplus object regions

    Parameters
    ----------
    SCENICPLUS_obj
        An instance of :class: `~scenicplus.scenicplus_class.SCENICPLUS`.
    min_pct
        only keep regions which are accessible in at least `min_pct` of cells.
        default: 0
    max_pct
        only keep regions which are accessible in maximal `max_pct` of cells.
        default: 100
    return_copy
        If set to True a new SCENICPLUS object will be generated containing filtered data.
        default: False
    """
    percent_of_cells_region_accessible = np.array(
        (SCENICPLUS_obj.X_ACC > 0).sum(1) / SCENICPLUS_obj.n_cells).flatten()
    regions_to_keep = SCENICPLUS_obj.region_names[
        np.logical_and(percent_of_cells_region_accessible > (min_pct / 100),
                       percent_of_cells_region_accessible < (max_pct / 100))]
    log.info(
        f'Going from {SCENICPLUS_obj.n_regions} regions to {len(regions_to_keep)} regions.')
    if return_copy:
        return SCENICPLUS_obj.subset(regions=regions_to_keep, return_copy=return_copy)
    else:
        SCENICPLUS_obj.subset(regions=regions_to_keep, return_copy=return_copy)

def simplify_eregulon(scplus_obj, eRegulon_signatures_key):
    md = scplus_obj.uns[eRegulon_signatures_key]['Gene_based']
    names = list(set([x.split('_(')[0][:len(x.split('_(')[0]) - 2] for x in md.keys()]))
    scplus_obj.uns[eRegulon_signatures_key]['Gene_based'] = {x:list(set(sum([value for key, value in md.items() if key.startswith(x)], []))) for x in names}
    scplus_obj.uns[eRegulon_signatures_key]['Gene_based'] = {x+'_('+str(len(scplus_obj.uns[eRegulon_signatures_key]['Gene_based'][x]))+'g)': scplus_obj.uns[eRegulon_signatures_key]['Gene_based'][x] for x in scplus_obj.uns[eRegulon_signatures_key]['Gene_based'].keys()}
    md = scplus_obj.uns[eRegulon_signatures_key]['Region_based']
    names = list(set([x.split('_(')[0][:len(x.split('_(')[0]) - 2] for x in md.keys()]))
    scplus_obj.uns[eRegulon_signatures_key]['Region_based'] = {x:list(set(sum([value for key, value in md.items() if key.startswith(x)], []))) for x in names}
    scplus_obj.uns[eRegulon_signatures_key]['Region_based'] = {x+'_('+str(len(scplus_obj.uns[eRegulon_signatures_key]['Region_based'][x]))+'r)': scplus_obj.uns[eRegulon_signatures_key]['Region_based'][x] for x in scplus_obj.uns[eRegulon_signatures_key]['Region_based'].keys()}


def remove_second_sign(x):
        if 'extended' not in x:
                TF, first, second, n = x.split('_')
                return f'{TF}_{first}_{n}'
        else:
                TF, extended, first, second, n = x.split('_')
                return f'{TF}_{extended}_{first}_{n}'

def apply_std_filtering_to_eRegulons(scplus_obj):
    ## only keep positive R2G
    print("Only keeping positive R2G")
    scplus_obj.uns['eRegulon_metadata_filtered'] = scplus_obj.uns['eRegulon_metadata'].query('R2G_rho > 0')
    ## only keep extended if no direct
    print("Only keep extended if not direct")
    scplus_obj.uns['eRegulon_metadata_filtered']['Consensus_name'] = scplus_obj.uns['eRegulon_metadata_filtered'].apply(lambda x: f"{x.TF}_{'+' if x.TF2G_rho > 0 else '-'}_{'+' if x.R2G_rho > 0 else '-'}", axis = 1)
    eRegulons_direct = set(
            scplus_obj.uns['eRegulon_metadata_filtered'].loc[
                    scplus_obj.uns['eRegulon_metadata_filtered']['is_extended'] == "False",
                    'Consensus_name'
            ])
    eRegulons_extended = set(
            scplus_obj.uns['eRegulon_metadata_filtered'].loc[
                    scplus_obj.uns['eRegulon_metadata_filtered']['is_extended'] == "True",
                    'Consensus_name'
            ])
    extended_not_direct = list(eRegulons_extended - eRegulons_direct)
    scplus_obj.uns['eRegulon_metadata_filtered'] = scplus_obj.uns['eRegulon_metadata_filtered'].loc[
            np.logical_or(
                    np.logical_and(
                            scplus_obj.uns['eRegulon_metadata_filtered']['is_extended'] == "True",
                            np.isin(scplus_obj.uns['eRegulon_metadata_filtered']['Consensus_name'], extended_not_direct)
                    ),
                    scplus_obj.uns['eRegulon_metadata_filtered']['is_extended'] == "False")]
    
    print("Getting signatures...")
    get_eRegulons_as_signatures(scplus_obj,
                                eRegulon_metadata_key='eRegulon_metadata_filtered',
                                key_added='eRegulon_signatures_filtered')
    print("Simplifying eRegulons ...")
    simplify_eregulon(scplus_obj, 'eRegulon_signatures_filtered')
    scplus_obj.uns['eRegulon_metadata_filtered']['Gene_signature_name'] = [remove_second_sign(x) for x in scplus_obj.uns['eRegulon_metadata_filtered']['Gene_signature_name']]
    scplus_obj.uns['eRegulon_metadata_filtered']['Region_signature_name'] = [remove_second_sign(x) for x in scplus_obj.uns['eRegulon_metadata_filtered']['Region_signature_name']]