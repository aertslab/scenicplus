"""Wrangle output from pycistarget into a format compatible with SCENIC+."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

import anndata
import h5py
import numpy as np
import pandas as pd
from pycistarget.utils import get_motifs_per_TF, get_TF_list
from scipy import sparse


@dataclass
class Cistrome:
    """
    A dataclass representing a transcription factor and its associated target regions.

    Attributes
    ----------
    tf_name : str
        The name of the transcription factor.
    motifs : Set[str]
        A set of motif names associated with the transcription factor.
    target_regions : Set[str]
        A set of target regions associated with the transcription factor.
    extended : bool
        A boolean indicating whether the target regions
        were identified using extended annotations.

    """

    tf_name: str
    motifs: Set[str]
    target_regions: Set[str]
    extended: bool

def _read_motif_hits(h5: h5py.Group) -> Dict[str, Dict[str, List[str]]]:
    """Helper function to read motif hits from a h5py Group object."""
    motif_hits: Dict[str, Dict[str, List[str]]] = {}
    for database_or_regionset in h5["motif_hits"]:
        motif_hits[database_or_regionset] = {}
        for motif_name in h5["motif_hits"][database_or_regionset]:
            motif_hits[database_or_regionset][motif_name] = list(map(
                        bytes.decode,
                        h5["motif_hits"][database_or_regionset][motif_name][:]))
    return motif_hits

def _signatures_to_iter(
        paths_to_motif_enrichment_results: List[str]):
    """Helper function to iterate over motif enrichment results."""
    for f in paths_to_motif_enrichment_results:
        if f.endswith(".hdf5"):
            with h5py.File(f, "r") as h5:
                for name in h5:
                    if "motif_enrichment" not in h5[name]:
                        continue
                    motif_enrichment = pd.read_hdf(
                        f, key = f"{name}/motif_enrichment"
                    )
                    motif_hits = _read_motif_hits(h5[name])
                    yield motif_enrichment, motif_hits["region_set"]

def _get_cistromes(
        motif_enrichment_table: pd.DataFrame,
        motif_hits: Dict[str, str],
        scplus_regions: Set[str],
        direct_annotation: List[str],
        extended_annotation: List[str]) -> List[Cistrome]:
    """
    Helper function to get region TF target regions based on motif hits.

    Parameters
    ----------
    motif_enrichment_table: pd.DataFrame
        Pandas DataFrame containing motif enrichment data
    motif_hits: Dict[str, str]
        dict of motif hits (mapping motifs to regions)
    scplus_regions: Set[str]
        set of regions in the scplus_obj
    direct_annotation: List[str]
        list of annotations to use as 'direct'
    extended_annotation: List[str]
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
            if motif in motif_hits:
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
            if motif in motif_hits:
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
    """Helper function to merge cistromes with the same TF name."""
    a_cistromes = np.array(cistromes, dtype = "object")
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
        assert all(x.tf_name == tf_name for x in cistromes_tf)
        assert all(x.extended == cistromes_tf[0].extended for x in cistromes_tf)
        yield Cistrome(
            tf_name = tf_name,
            motifs = set.union(
                *[cistrome.motifs for cistrome in cistromes_tf]),
            target_regions = set.union(
                *[cistrome.target_regions for cistrome in cistromes_tf]),
            extended = cistromes_tf[0].extended)

def _cistromes_to_adata(cistromes: List[Cistrome]) -> anndata.AnnData:
    """Helper function to convert cistromes to anndata."""
    tf_names = [cistrome.tf_name for cistrome in cistromes]
    # join has to be done in order to write the resulting anndata to disk as h5ad
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
        paths_to_motif_enrichment_results: List[str],
        scplus_regions: Set[str],
        direct_annotation: List[str] = ["Direct_annot"],  # noqa: B006
        extended_annotation: List[str] = ["Orthology_annot"]  # noqa: B006
        ) -> Tuple[anndata.AnnData, anndata.AnnData]:
    """
    Generate cistromes from motif enrichment tables.

    Parameters
    ----------
    paths_to_motif_enrichment_results: List[str]
        A list of paths to motif enrichment results generated by pycistarget
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
    for motif_enrichment_table, motif_hits in _signatures_to_iter(
        paths_to_motif_enrichment_results):
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
    adata_direct_cistromes.var["is_extended"] = False
    adata_extended_cistromes.var["is_extended"] = True
    return adata_direct_cistromes, adata_extended_cistromes
