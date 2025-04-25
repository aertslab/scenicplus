"""export SCENIC+ object to target genes and target regions loom file.

These files can be visualized in the SCope single cell viewer.

"""

import json
import logging
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import sys
from loomxpy.loomxpy import SCopeLoom
from loomxpy.utils import compress_encode
from ctxcore.genesig import Regulon
from typing import Dict, List, Mapping, Optional, Sequence
from multiprocessing import cpu_count
from collections import OrderedDict
import os
from sklearn.feature_extraction.text import CountVectorizer
from itertools import repeat, chain, islice
import loompy as lp
import re



def export_to_loom(scplus_obj,
                   signature_key: str,
                   out_fname: str,
                   eRegulon_metadata_key: Optional[str] = 'eRegulon_metadata',
                   auc_key: Optional[str] = 'eRegulon_AUC',
                   auc_thr_key: Optional[str] = 'eRegulon_AUC_thresholds',
                   keep_direct_and_extended_if_not_direct: Optional[bool] = False,
                   selected_features: Optional[List[str]] = None,
                   selected_cells: Optional[List[str]] = None,
                   cluster_annotation: List[str] = None,
                   tree_structure: Sequence[str] = (),
                   title: str = None,
                   nomenclature: str = "Unknown"):
    """
    Create SCope [Davie et al, 2018] compatible loom files 
    
    Parameters
    ---------
    scplus_obj: class::SCENICPLUS
        A SCENIC+ object with eRegulons, AUCs, and AUC thresholds computed
    signature_key: str
        Whether a 'Gene_based' or a 'Region_based' file should be produced. Possible values: 'Gene_based' or 'Region_based'
    out_fname: str
        Path to output file.
    eRegulon_metadata_key: str, optional
        Slot where the eRegulon metadata is stored.
    auc_key: str, optional
        Slot where the eRegulon AUC are stored
    auc_thr_key: str, optional
        Slot where AUC thresholds are stored
    keep_direct_and_extended_if_not_direct: bool, optional
        Keep only direct eregulons and add extended ones only if there is not a direct one.
    selected_features: List, optional
        A list with selected genes/region to use.
    selected_cells: List, optional
        A list with selected cells to use.
    cluster_annotations: List, optional
        A list with variables to use as cluster annotations. By default, those included in the DEGs/DARs dictionary will be used (plus those specificed here)
    tree_structure: sequence, optional
        A sequence of strings that defines the category tree structure. Needs to be a sequence of strings with three elements. Default: ()
    title: str, optional
        The title for this loom file. If None than the basename of the filename is used as the title. Default: None
    nomenclature: str, optional
        The name of the genome. Default: 'Unknown'

    References
    -----------
    Davie, K., Janssens, J., Koldere, D., De Waegeneer, M., Pech, U., Kreft, Ł., ... & Aerts, S. (2018). A single-cell transcriptome atlas of the
    aging Drosophila brain. Cell, 174(4), 982-998.
    Van de Sande, B., Flerin, C., Davie, K., De Waegeneer, M., Hulselmans, G., Aibar, S., ... & Aerts, S. (2020). A scalable SCENIC
    workflow for single-cell gene regulatory network analysis. Nature Protocols, 15(7), 2247-2276.
    """
    # Create logger
    level = logging.INFO
    log_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    log = logging.getLogger('SCENIC+')

    log.info('Formatting data')
    # Check-up
    if auc_key not in scplus_obj.uns.keys():
        log.error('Compute ' + auc_key + 'first!')
    if auc_thr_key not in scplus_obj.uns.keys():
        log.error('Compute ' + auc_thr_key + 'first!')
    if eRegulon_metadata_key not in scplus_obj.uns.keys():
        log.error('Compute ' + eRegulon_metadata_key + 'first!')

    # Set keys and subset if needed
    if signature_key == 'Gene_based':
        markers_key = 'DEGs'
        if selected_features is not None or selected_cells is not None:
            scplus_obj = scplus_obj.subset(
                scplus_obj, cells=selected_cells, genes=selected_features, return_copy=True)
        ex_mtx = scplus_obj.X_EXP.T
        feature_names = scplus_obj.gene_names
        cell_names = scplus_obj.cell_names
    else:
        markers_key = 'DARs'
        if selected_features is not None or selected_cells is not None:
            scplus_obj = scplus_obj.subset(
                scplus_obj, cells=selected_cells, regions=selected_features, return_copy=True)
        ex_mtx = scplus_obj.X_ACC
        feature_names = scplus_obj.region_names
        cell_names = scplus_obj.cell_names

    if markers_key in scplus_obj.uns.keys():
        cluster_markers = scplus_obj.uns[markers_key]
        if cluster_annotation is None:
            cluster_annotation = list(cluster_markers.keys())
        else:
            cluster_annotation = list(
                set(sum([cluster_annotation, list(cluster_markers.keys())], [])))

    else:
        cluster_markers = None

    # Extract cell data
    cell_data = scplus_obj.metadata_cell

    # Check ups
    if cluster_annotation is not None:
        for annotation in cluster_annotation:
            if annotation not in cell_data:
                log.error(
                    'The cluster annotation',
                    annotation,
                    ' is not included in scplus_obj.metadata_cell')

    # eGRN AUC values
    auc_mtx = scplus_obj.uns[auc_key][signature_key].loc[cell_names]
    auc_mtx.columns = [re.sub('_\(.*\)', '', x)
                       for x in auc_mtx.columns]
    auc_thresholds = scplus_obj.uns[auc_thr_key][signature_key]
    auc_thresholds.index = [re.sub('_\(.*\)', '', x)
                            for x in auc_thresholds.index]
    if auc_mtx.shape[1] > 900 and keep_direct_and_extended_if_not_direct is False:
        log.info('The number of regulons is more than > 900. keep_direct_and_extended_if_not_direct is set to True')
        keep_direct_and_extended_if_not_direct = True
    if keep_direct_and_extended_if_not_direct is True:
        direct_eRegulons = [x for x in auc_mtx.columns if not 'extended' in x]
        extended_eRegulons =  [x for x in auc_mtx.columns  if 'extended' in x and not x.replace('_extended', '') in direct_eRegulons]
        selected_eRegulons = direct_eRegulons + extended_eRegulons
        auc_mtx = auc_mtx[selected_eRegulons]
        auc_thresholds = auc_thresholds.loc[selected_eRegulons]

    # Add TF expression in Region_based
    if signature_key == 'Region_based':
        tf_names = list(set([x.split('_')[0] for x in auc_mtx.columns]))
        tf_mat = scplus_obj.to_df('EXP')[tf_names].T
        ex_mtx = sparse.vstack(
            [ex_mtx, sparse.csr_matrix(tf_mat.values)], format='csr')
        feature_names = feature_names.tolist() + tf_names

    # Format regulons
    if signature_key == 'Gene_based':
        regulons = {re.sub('_\(.*\)', '', x): ' '.join(list(set(scplus_obj.uns[eRegulon_metadata_key][scplus_obj.uns[eRegulon_metadata_key].Gene_signature_name == x]['Gene']))) for x in list(
            set(scplus_obj.uns[eRegulon_metadata_key].Gene_signature_name))}
        cv = CountVectorizer(lowercase=False)
    else:
        regulons = {re.sub('_\(.*\)', '', x): ' '.join(list(set(scplus_obj.uns[eRegulon_metadata_key][scplus_obj.uns[eRegulon_metadata_key].Region_signature_name == x]['Region']))) for x in list(
            set(scplus_obj.uns[eRegulon_metadata_key].Region_signature_name))}
        cv = CountVectorizer(
            lowercase=False, token_pattern=r'(?u)\b\w\w+\b:\b\w\w+\b-\b\w\w+\b')
    regulon_mat = cv.fit_transform(regulons.values())
    regulon_mat = pd.DataFrame(regulon_mat.todense(
    ), columns=cv.get_feature_names_out(), index=regulons.keys())
    regulon_mat = regulon_mat.reindex(columns=feature_names, fill_value=0).T
    if keep_direct_and_extended_if_not_direct is True:
        regulon_mat = regulon_mat[selected_eRegulons]

    # Cell annotations and metrics
    metrics = []
    annotations = []
    for var in cell_data:
        if isinstance(cell_data[var][0], np.bool_):
            annotations.append(cell_data[var])
        else:
            try:
                metrics.append(cell_data[var].astype('float64'))
                cell_data[var] = cell_data[var].astype('float64')
            except BaseException:
                if len(set(cell_data[var])) < 255:
                    annotations.append(cell_data[var].astype('str'))
                    cell_data[var] = cell_data[var].astype('str')
    metrics = pd.concat(metrics, axis=1).fillna(0) if len(metrics) > 0 else []
    annotations = pd.concat(annotations, axis=1) if len(
        annotations) > 0 else []

    # Embeddings. Cell embeddings in this case
    embeddings = scplus_obj.dr_cell
    
    #only keep first two dimensions of embedding
    for key in embeddings.keys():
        embeddings[key] = embeddings[key][embeddings[key].columns[0:2]]

    # Add linked_gene information
    if signature_key == 'Gene_based':
        linked_gene = scplus_obj.uns[eRegulon_metadata_key][[
            'Region', 'Gene']].groupby('Gene').agg(lambda x: '; '.join(set(x)))
        linked_gene = linked_gene.reindex(
            index=feature_names, fill_value='').loc[feature_names]
    else:
        linked_gene = scplus_obj.uns[eRegulon_metadata_key][['Region', 'Gene']].groupby(
            'Region').agg(lambda x: '; '.join(set(x)))
        linked_gene = linked_gene.reindex(
            index=feature_names, fill_value='').loc[feature_names]
    linked_gene.columns = ['0']
    # Create minimal loom
    log.info('Creating minimal loom')
    _export_minimal_loom(ex_mtx=ex_mtx,
                        cell_names=cell_names,
                        feature_names=feature_names,
                        out_fname=out_fname,
                        regulons=regulon_mat,
                        cell_annotations=None,
                        tree_structure=tree_structure,
                        title=title,
                        nomenclature=nomenclature,
                        embeddings=embeddings,
                        auc_mtx=auc_mtx,
                        auc_thresholds=auc_thresholds,
                        linked_gene=linked_gene)

    # Add annotations
    log.info('Adding annotations')
    path_to_loom = out_fname
    loom = SCopeLoom.read_loom(path_to_loom)
    if len(metrics):
        _add_metrics(loom, metrics)
    if len(annotations):
        _add_annotation(loom, annotations)

    # Add clusterings
    if cluster_annotation is not None:
        log.info('Adding clusterings')
        _add_clusterings(loom, pd.DataFrame(cell_data[cluster_annotation]))
    # Add markers
    if cluster_markers is not None:
        log.info('Adding markers')
        annotation_in_markers = [
            x for x in cluster_annotation if x in cluster_markers.keys()]
        annotation_not_in_markers = [
            x for x in cluster_annotation if x not in cluster_markers.keys()]
        for x in annotation_not_in_markers:
            log.info(x, 'is not in the cluster markers dictionary')
        cluster_markers = {
            k: v
            for k, v in cluster_markers.items()
            if k in annotation_in_markers
        }
        # Keep genes in data
        for y in cluster_markers:
            cluster_markers[y] = {
                x: cluster_markers[y][x][cluster_markers[y]
                                         [x].index.isin(feature_names)]
                for x in cluster_markers[y].keys()
            }
        _add_markers(loom, cluster_markers)

    log.info('Exporting')
    loom.export(out_fname)


def _export_minimal_loom(
    ex_mtx: sparse.csr_matrix,
    cell_names: List[str],
    feature_names: List[str],
    out_fname: str,
    regulons: pd.DataFrame = None,
    cell_annotations: Optional[Mapping[str, str]] = None,
    tree_structure: Sequence[str] = (),
    title: Optional[str] = None,
    nomenclature: str = "Unknown",
    num_workers: int = cpu_count(),
    embeddings: Mapping[str, pd.DataFrame] = {},
    auc_mtx=None,
    auc_thresholds=None,
    compress: bool = False,
    linked_gene=None
):
    """
    An internal function to create a minimal loom file
    """

    # Information on the general loom file format: http://linnarssonlab.org/loompy/format/index.html
    # Information on the SCope specific alterations: https://github.com/aertslab/SCope/wiki/Data-Format

    if cell_annotations is None:
        cell_annotations = dict(zip(cell_names, ['-'] * ex_mtx.shape[1]))

    # Create an embedding based on tSNE.
    # Name of columns should be "_X" and "_Y".
    if len(embeddings) == 0:
        embeddings = {
            "tSNE (default)": pd.DataFrame(data=TSNE().fit_transform(auc_mtx), index=cell_names, columns=['_X', '_Y'])
        }  # (n_cells, 2)

    id2name = OrderedDict()
    embeddings_X = pd.DataFrame(index=cell_names)
    embeddings_Y = pd.DataFrame(index=cell_names)
    for idx, (name, df_embedding) in enumerate(embeddings.items()):
        if len(df_embedding.columns) != 2:
            raise Exception('The embedding should have two columns.')

        # Default embedding must have id == -1 for SCope.
        embedding_id = idx - 1
        id2name[embedding_id] = name

        embedding = df_embedding.copy()
        embedding.columns = ['_X', '_Y']
        embeddings_X = pd.merge(
            embeddings_X,
            embedding['_X'].to_frame().rename(
                columns={'_X': str(embedding_id)}),
            left_index=True,
            right_index=True,
        )
        embeddings_Y = pd.merge(
            embeddings_Y,
            embedding['_Y'].to_frame().rename(
                columns={'_Y': str(embedding_id)}),
            left_index=True,
            right_index=True,
        )

    # Encode cell type clusters.
    # The name of the column should match the identifier of the clustering.
    name2idx = dict(map(reversed, enumerate(
        sorted(set(cell_annotations.values())))))
    clusterings = (
        pd.DataFrame(data=cell_names, index=cell_names, columns=['0'])
        .replace(cell_annotations)
        .replace(name2idx)
    )

    # Create meta-data structure.
    def create_structure_array(df):
        # Create a numpy structured array
        return np.array([tuple(row) for row in df.values], dtype=np.dtype(list(zip(df.columns, df.dtypes))))

    default_embedding = pd.DataFrame([embeddings_X.iloc[:,0], embeddings_Y.iloc[:,0]], columns=cell_names, index=['_X', '_Y']).T
    column_attrs = {
        "CellID": np.array(cell_names),
        "Embedding": create_structure_array(default_embedding),
        "RegulonsAUC": create_structure_array(auc_mtx),
        "Clusterings": create_structure_array(clusterings),
        "ClusterID": clusterings.values,
        'Embeddings_X': create_structure_array(embeddings_X),
        'Embeddings_Y': create_structure_array(embeddings_Y),
    }
    if linked_gene is None:
        row_attrs = {
            "Gene": np.array(feature_names),
            "Regulons": create_structure_array(regulons),
        }
    else:
        row_attrs = {
            "Gene": np.array(feature_names),
            "Regulons": create_structure_array(regulons),
            "linkedGene": np.array(linked_gene['0'])
        }

    def fetch_logo(context):
        for elem in context:
            if elem.endswith('.png'):
                return elem
        return ""

    regulon_thresholds = [
        {
            "regulon": name,
            "defaultThresholdValue": (threshold if isinstance(threshold, float) else threshold[0]),
            "defaultThresholdName": "gaussian_mixture_split",
            "allThresholds": {"gaussian_mixture_split": (threshold if isinstance(threshold, float) else threshold[0])},
            "motifData": "",
        }
        for name, threshold in auc_thresholds.iteritems()
    ]

    general_attrs = {
        "title": os.path.splitext(os.path.basename(out_fname))[0] if title is None else title,
        "MetaData": json.dumps(
            {
                "embeddings": [{'id': identifier, 'name': name} for identifier, name in id2name.items()],
                "annotations": [{"name": "", "values": []}],
                "clusterings": [
                    {
                        "id": 0,
                        "group": "celltype",
                        "name": "Cell Type",
                        "clusters": [{"id": idx, "description": name} for name, idx in name2idx.items()],
                    }
                ],
                "regulonThresholds": regulon_thresholds,
            }
        ),
        "Genome": nomenclature,
    }

    # Add tree structure.
    # All three levels need to be supplied
    assert len(tree_structure) <= 3, ""
    general_attrs.update(
        ("SCopeTreeL{}".format(idx + 1), category)
        for idx, category in enumerate(list(islice(chain(tree_structure, repeat("")), 3)))
    )

    # Compress MetaData global attribute
    if compress:
        general_attrs["MetaData"] = compress_encode(
            value=general_attrs["MetaData"])

    # Create loom file for use with the SCope tool.
    lp.create(
        filename=out_fname,
        layers=ex_mtx,
        row_attrs=row_attrs,
        col_attrs=column_attrs,
        file_attrs=general_attrs,
    )


def _get_metadata(loom):
    """
    A helper function to get metadata
    """
    annot_metadata = loom.get_meta_data()['annotations']
    annot_mt_column_names = [annot_metadata[x]['name']
                             for x in range(len(annot_metadata))]
    annot_mt = pd.concat([pd.DataFrame(loom.col_attrs[annot_mt_column_names[x]])
                          for x in range(len(annot_mt_column_names))], axis=1)
    annot_mt.columns = [annot_mt_column_names[x]
                        for x in range(len(annot_mt_column_names))]
    annot_mt.index = loom.get_cell_ids().tolist()
    return annot_mt


def _add_metrics(loom, metrics: pd.DataFrame):
    """
    A helper function to add metrics
    """
    md_metrics = []
    for metric in metrics:
        md_metrics.append({"name": metric})
        loom.col_attrs[metric] = np.array(metrics[metric])
    loom.global_attrs["MetaData"].update({'metrics': md_metrics})


def _add_annotation(loom, annots: pd.DataFrame):
    """
    A helper function to add annotations
    """
    md_annot = []
    for annot in annots:
        vals = list(annots[annot])
        uniq_vals = np.unique(vals)
        md_annot.append({
            "name": annot,
            "values": list(map(lambda x: str(x), uniq_vals.tolist()))
        })
        loom.col_attrs[annot] = np.array(annots[annot])
    loom.global_attrs["MetaData"].update({'annotations': md_annot})


def _add_clusterings(loom: SCopeLoom,
                    cluster_data: pd.DataFrame):
    """
    A helper function to add clusters
    """
    col_attrs = loom.col_attrs

    attrs_metadata = {}
    attrs_metadata["clusterings"] = []
    clusterings = pd.DataFrame(index=cluster_data.index.tolist())
    j = 0

    for cluster_name in cluster_data.columns:
        clustering_id = j
        clustering_algorithm = cluster_name

        clustering_resolution = cluster_name
        cluster_marker_method = 'Wilcoxon'

        num_clusters = len(np.unique(cluster_data[cluster_name]))
        cluster_2_number = {
            np.unique(cluster_data[cluster_name])[i]: i
            for i in range(num_clusters)
        }

        # Data
        clusterings[str(clustering_id)] = [cluster_2_number[x]
                                           for x in cluster_data[cluster_name].tolist()]

        # Metadata
        attrs_metadata["clusterings"] = attrs_metadata["clusterings"] + [{
            "id": clustering_id,
            "group": clustering_algorithm,
            "name": clustering_algorithm,
            "clusters": [],
            "clusterMarkerMetrics": [
                {
                    "accessor": "avg_logFC",
                    "name": "Avg. logFC",
                    "description": f"Average log fold change from {cluster_marker_method.capitalize()} test"
                }, {
                    "accessor": "pval",
                    "name": "Adjusted P-Value",
                    "description": f"Adjusted P-Value from {cluster_marker_method.capitalize()} test"
                }
            ]
        }]

        for i in range(0, num_clusters):
            cluster = {}
            cluster['id'] = i
            cluster['description'] = np.unique(cluster_data[cluster_name])[i]
            attrs_metadata['clusterings'][j]['clusters'].append(cluster)

        j += 1

    # Update column attribute Dict
    col_attrs_clusterings = {
        # Pick the first one as default clustering (this is purely
        # arbitrary)
        "ClusterID": clusterings["0"].values,
        "Clusterings": _df_to_named_matrix(clusterings)
    }

    col_attrs = {**col_attrs, **col_attrs_clusterings}
    loom.col_attrs = col_attrs
    loom.global_attrs["MetaData"].update(
        {'clusterings': attrs_metadata["clusterings"]}
    )


def _add_markers(loom: SCopeLoom,
                markers_dict: Dict[str, Dict[str, pd.DataFrame]]):
    """
    A helper function to add markers to clusterings
    """
    attrs_metadata = loom.global_attrs["MetaData"]
    row_attrs = loom.row_attrs
    for cluster_name in markers_dict:
        idx = [i for i in range(len(attrs_metadata['clusterings']))
               if attrs_metadata['clusterings'][i]["name"] == cluster_name][0]
        clustering_id = attrs_metadata['clusterings'][idx]["id"]
        num_clusters = len(attrs_metadata['clusterings'][idx]["clusters"])
        cluster_description = [
            attrs_metadata['clusterings'][idx]["clusters"][x]['description']
            for x in range(num_clusters)
        ]

        # Initialize
        cluster_markers = pd.DataFrame(
            index=loom.get_genes(),
            columns=[str(x) for x in np.arange(num_clusters)]
        ).fillna(0, inplace=False)
        cluster_markers_avg_logfc = pd.DataFrame(
            index=loom.get_genes(),
            columns=[str(x) for x in np.arange(num_clusters)]
        ).fillna(0, inplace=False)
        cluster_markers_pval = pd.DataFrame(
            index=loom.get_genes(),
            columns=[str(x) for x in np.arange(num_clusters)]
        ).fillna(0, inplace=False)

        # Populate
        for i in range(0, num_clusters):
            try:
                gene_names = markers_dict[cluster_name][cluster_description[i]].index.tolist(
                )
                pvals_adj = markers_dict[cluster_name][cluster_description[i]
                                                       ]['Adjusted_pval']
                logfoldchanges = markers_dict[cluster_name][cluster_description[i]]['Log2FC']
                i = str(i)
                num_genes = len(gene_names)

                # Replace
                cluster_markers.loc[gene_names, i] = np.int(1)
                cluster_markers_avg_logfc.loc[gene_names, i] = logfoldchanges
                cluster_markers_pval.loc[gene_names, i] = pvals_adj
            except BaseException:
                print('No markers for ', cluster_description[i])

        # Update row attribute Dict
        row_attrs_cluster_markers = {
            f"ClusterMarkers_{str(idx)}": _df_to_named_matrix(
                cluster_markers.astype(np.int8)),
            f"ClusterMarkers_{str(idx)}_avg_logFC": _df_to_named_matrix(cluster_markers_avg_logfc.astype(np.float32)),
            f"ClusterMarkers_{str(idx)}_pval": _df_to_named_matrix(cluster_markers_pval.astype(np.float32))
        }
        row_attrs = {**row_attrs, **row_attrs_cluster_markers}
        loom.row_attrs = row_attrs


def _get_regulons(loom):
    """
    A helper function to get regulons
    """
    regulon_dict = pd.DataFrame(
        loom.get_regulons(),
        index=loom.row_attrs['Gene']
    ).to_dict()
    for t in regulon_dict:
        regulon_dict[t] = {x: y for x, y in regulon_dict[t].items() if y != 0}
    motif_data = {
        x['regulon']: x['motifData']
        for x in loom.global_attrs['MetaData']['regulonThresholds']
    }
    regulon_list = [
        Regulon(
            name=x,
            gene2weight=regulon_dict[x],
            transcription_factor=x.split('_')[0],
            gene2occurrence=[],
            context=frozenset(
                list(
                    motif_data[x]))
        )
        for x in regulon_dict.keys()
    ]
    return regulon_list


def _df_to_named_matrix(df: pd.DataFrame):
    """
    A helper function to create metadata structure.
    """
    return np.array([tuple(row) for row in df.values],
                    dtype=np.dtype(list(zip(df.columns, df.dtypes))))
