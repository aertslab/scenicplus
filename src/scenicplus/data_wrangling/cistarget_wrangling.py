"""Wrangle output from pycistarget into a format compatible with SCENIC+

"""

from pycistarget.motif_enrichment_dem import DEM
from pycistarget.motif_enrichment_cistarget import cisTarget
import pandas as pd
import numpy as np
import anndata
from scipy import sparse
from pycistarget.utils import get_TF_list, get_motifs_per_TF
from typing import Set, Dict, List, Iterable, Union, Tuple
from dataclasses import dataclass


@dataclass
class Cistrome:
    """
    Dataclass for intermediate use
    """
    tf_name: str
    motifs: Set[str]
    target_regions: Set[str]
    extended: bool


def _signatures_to_iter(
        menr: Dict[str, Union[DEM, Dict[str, cisTarget]]]):
    for x in menr.keys():
        if isinstance(menr[x], DEM):
            for y in menr[x].motif_enrichment.keys():
                yield menr[x].motif_enrichment[y], menr[x].motif_hits["Region_set"][y]
        elif isinstance(menr[x], dict):
            for y in menr[x].keys():
                if not isinstance(menr[x][y], cisTarget):
                    raise ValueError(f'Only motif enrichment results from pycistarget or DEM are allowed, not {type(menr[x][y])}')
                yield menr[x][y].motif_enrichment, menr[x][y].motif_hits["Region_set"]
        else:
            raise ValueError(f'Only motif enrichment results from pycistarget or DEM are allowed, not {type(menr[x])}')

def _get_cistromes(
        motif_enrichment_table: pd.DataFrame,
        motif_hits: Dict[str, str],
        scplus_regions: Set[str],
        direct_annotation: List[str],
        extended_annotation: List[str]) -> List[Cistrome]:
    """
    Helper function to get region TF target regions based on motif hits

    Parameters
    ----------
        motif_enrichment_table: 
            Pandas DataFrame containing motif enrichment data
        motif_hits: 
            dict of motif hits (mapping motifs to regions)
        scplus_regions:
            set of regions in the scplus_obj
        direct_annotation: 
            list of annotations to use as 'direct'
        extended_annotation: 
            list of annotations to use as 'extended'
            
    Returns
    -------
        List of cistromes
    """
    tfs_direct = get_TF_list(
        motif_enrichment_table = motif_enrichment_table,
        annotation = direct_annotation)
    tfs_extended = get_TF_list(
        motif_enrichment_table = motif_enrichment_table,
        annotation = extended_annotation)
    cistromes = []
    for tf_name in tfs_direct:
        motifs_annotated_to_tf = get_motifs_per_TF(
            motif_enrichment_table = motif_enrichment_table,
            tf = tf_name,
            motif_column = "Index",
            annotation = direct_annotation)
        target_regions_motif_direct: Set[str] = set()
        for motif in motifs_annotated_to_tf:
            if motif in motif_hits.keys():
                target_regions_motif_direct.update(motif_hits[motif])
            else:
                raise ValueError(f"Motif enrichment table and motif hits don't match for the TF: {tf_name}")
        cistromes.append(
            Cistrome(
                tf_name = tf_name,
                motifs = set(motifs_annotated_to_tf),
                target_regions = target_regions_motif_direct & scplus_regions,
                extended = False))
    for tf_name in tfs_extended:
        motifs_annotated_to_tf = get_motifs_per_TF(
            motif_enrichment_table = motif_enrichment_table,
            tf = tf_name,
            motif_column = "Index",
            annotation = extended_annotation)
        target_regions_motif_extended: Set[str] = set()
        for motif in motifs_annotated_to_tf:
            if motif in motif_hits.keys():
                target_regions_motif_extended.update(motif_hits[motif])
            else:
                raise ValueError(f"Motif enrichment table and motif hits don't match for the TF: {tf_name}")
        cistromes.append(
            Cistrome(
                tf_name = tf_name,
                motifs = set(motifs_annotated_to_tf),
                target_regions = target_regions_motif_extended & scplus_regions,
                extended = True))
    return cistromes

def _merge_cistromes(cistromes: List[Cistrome]) -> Iterable[Cistrome]:
    a_cistromes = np.array(cistromes, dtype = 'object')
    tf_names = np.array([cistrome.tf_name for cistrome in a_cistromes])
    tf_names_sorted_idx = np.argsort(tf_names)
    a_cistromes = a_cistromes[tf_names_sorted_idx]
    tf_names = tf_names[tf_names_sorted_idx]
    u_tf_names, idx_tf_names = np.unique(tf_names, return_index = True)
    for i, tf_name in enumerate(u_tf_names):
        if i < len(u_tf_names) - 1:
            cistromes_tf = a_cistromes[idx_tf_names[i]:idx_tf_names[i + 1]]
        else:
            cistromes_tf = a_cistromes[idx_tf_names[i]:]
        assert all([x.tf_name == tf_name for x in cistromes_tf])
        assert all([x.extended == cistromes_tf[0].extended for x in cistromes_tf])
        yield Cistrome(
            tf_name = tf_name,
            motifs = set.union(
                *[cistrome.motifs for cistrome in cistromes_tf]),
            target_regions = set.union(
                *[cistrome.target_regions for cistrome in cistromes_tf]),
            extended = cistromes_tf[0].extended)

def _cistromes_to_adata(cistromes: List[Cistrome]) -> anndata.AnnData:
    tf_names = [cistrome.tf_name for cistrome in cistromes]
    # join has to be done in order to be able to write the resulting anndata to disk as h5ad
    motifs = [",".join(cistrome.motifs) for cistrome in cistromes]
    union_target_regions= list(set.union(
            *[cistrome.target_regions for cistrome in cistromes]))
    cistrome_hit_mtx = np.zeros(
        (len(union_target_regions), len(tf_names)),
        dtype = bool)
    for i in range(len(tf_names)):
        cistrome_hit_mtx[:, i] = [
            region in cistromes[i].target_regions 
            for region in union_target_regions]
    cistrome_adata = anndata.AnnData(
        X = sparse.csc_matrix(cistrome_hit_mtx), dtype = bool,
        obs = pd.DataFrame(index = list(union_target_regions)),
        var = pd.DataFrame(index = tf_names))
    cistrome_adata.var["motifs"] = motifs
    return cistrome_adata

def get_and_merge_cistromes(
        menr: Dict[str, Union[DEM, Dict[str, cisTarget]]],
        scplus_regions: Set[str],
        direct_annotation: List[str] = ['Direct_annot'],
        extended_annotation: List[str] = ['Orthology_annot']
        ) -> Tuple[anndata.AnnData, anndata.AnnData]:
    """Generate cistromes from motif enrichment tables

    Parameters
    ---------
    menr: Dict[str, Union[DEM, Dict[str, cisTarget]]]
        A :Dict: of motif enrichment results generated by pycistarget
    scplus_regions: Set[str]
        A set of regions to be used in the SCENIC+ analysis
    direct_annotation: List[str] = ['Direct_annot']
        A list of annotations to use as annotations with direct evidence
    extended_annotation: List[str] = ['Orthology_annot']
        A list of annotations to use as annotations with extended evidence

    Returns
    -------
    A tuple of AnnData containing cistromes for TFs directly annotated to motifs
    and TFs annotated to motifs with extended evidence

    """
    # get cistromes
    cistromes = []
    for motif_enrichment_table, motif_hits in _signatures_to_iter(menr):
        cistromes.extend(
            _get_cistromes(
                motif_enrichment_table = motif_enrichment_table,
                motif_hits = motif_hits,
                scplus_regions = scplus_regions,
                direct_annotation = direct_annotation,
                extended_annotation = extended_annotation))
    # merge cistromes. Seperatly for direct and extended
    direct_cistromes = [cistrome for cistrome in cistromes if not cistrome.extended]
    extended_cistromes = [cistrome for cistrome in cistromes if cistrome.extended]
    merged_direct_cistromes = list(_merge_cistromes(direct_cistromes))
    merged_extended_cistromes = list(_merge_cistromes(extended_cistromes))
    adata_direct_cistromes = _cistromes_to_adata(merged_direct_cistromes)
    adata_extended_cistromes = _cistromes_to_adata(merged_extended_cistromes)
    adata_direct_cistromes.var['is_extended'] = False
    adata_extended_cistromes.var['is_extended'] = True
    return adata_direct_cistromes, adata_extended_cistromes
