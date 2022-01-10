import pandas as pd
from typing import Union, Dict, Sequence, Optional, List
import anndata
import scanpy as sc
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
from matplotlib.colors import to_rgba, to_hex
import numpy as np


def format_df_nx(df, key, var):
    """
    A helper function to format differential test results
    """
    df.index = df['names']
    df = pd.DataFrame(df['logfoldchanges'])
    df.columns = [var+'_Log2FC_'+key]
    df.index.name = None
    return df


def get_log2fc_nx(scplus_obj: 'SCENICPLUS',
                  variable,
                  features,
                  contrast: Optional[str] = 'gene'
                  ):
    """
    A helper function to derive log2fc changes
    """
    if contrast == 'gene':
        adata = anndata.AnnData(X=scplus_obj.X_EXP, obs=pd.DataFrame(
            index=scplus_obj.cell_names), var=pd.DataFrame(index=scplus_obj.gene_names))
    if contrast == 'region':
        adata = anndata.AnnData(X=scplus_obj.X_ACC.T, obs=pd.DataFrame(
            index=scplus_obj.cell_names), var=pd.DataFrame(index=scplus_obj.region_names))
    adata.obs = pd.DataFrame(scplus_obj.metadata_cell[variable])
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, features]
    sc.tl.rank_genes_groups(
        adata, variable, method='wilcoxon', corr_method='bonferroni')
    groups = adata.uns['rank_genes_groups']['names'].dtype.names
    diff_list = [format_df_nx(sc.get.rank_genes_groups_df(
        adata, group=group), group, variable) for group in groups]
    return pd.concat(diff_list, axis=1)


def create_nx_tables(scplus_obj: 'SCENICPLUS',
                     eRegulon_metadata_key='eRegulon_metadata',
                     subset_eRegulons=None,
                     subset_regions=None,
                     subset_genes=None,
                     add_differential_gene_expression=False,
                     add_differential_region_accessibility=False,
                     differential_variable=[]):
    """
    TO DO
    """
    er_metadata = scplus_obj.uns[eRegulon_metadata_key].copy()
    if subset_eRegulons is not None:
        subset_eRegulons = [x + '_[^a-zA-Z0-9]' for x in subset_eRegulons]
        er_metadata = er_metadata[er_metadata['Region_signature_name'].str.contains(
            '|'.join(subset_eRegulons))]
    if subset_regions is not None:
        er_metadata = er_metadata[er_metadata['Region'].str.contains(
            '|'.join(subset_regions))]
    if subset_genes is not None:
        er_metadata = er_metadata[er_metadata['Gene'].str.contains(
            '|'.join(subset_genes))]
    nx_tables = {}
    nx_tables['Edge'] = {}
    nx_tables['Node'] = {}
    # Generate edge tables
    r2g_columns = [x for x in er_metadata.columns if 'R2G' in x]
    tf2g_columns = [x for x in er_metadata.columns if 'TF2G' in x]
    nx_tables['Edge']['TF2R'] = er_metadata[er_metadata.columns.difference(
        r2g_columns + tf2g_columns)].drop('Gene', axis=1).drop_duplicates()
    nx_tables['Edge']['TF2R'] = nx_tables['Edge']['TF2R'][['TF', 'Region'] +
                                                          nx_tables['Edge']['TF2R'].columns.difference(['TF', 'Region']).tolist()]
    nx_tables['Edge']['R2G'] = er_metadata[er_metadata.columns.difference(
        tf2g_columns)].drop('TF', axis=1).drop_duplicates()
    nx_tables['Edge']['R2G'] = nx_tables['Edge']['R2G'][['Region', 'Gene'] +
                                                        nx_tables['Edge']['R2G'].columns.difference(['Region', 'Gene']).tolist()]
    nx_tables['Edge']['TF2G'] = er_metadata[er_metadata.columns.difference(
        r2g_columns)].drop('Region', axis=1).drop_duplicates()
    nx_tables['Edge']['TF2G'] = nx_tables['Edge']['TF2G'][['TF', 'Gene'] +
                                                          nx_tables['Edge']['TF2G'].columns.difference(['TF', 'Gene']).tolist()]
    # Generate node tables
    tfs = list(set(er_metadata['TF']))
    nx_tables['Node']['TF'] = pd.DataFrame(
        'TF', index=tfs, columns=['Node_type'])
    nx_tables['Node']['TF']['TF'] = tfs
    genes = list(set(er_metadata['Gene']))
    genes = [x for x in genes if x not in tfs]
    nx_tables['Node']['Gene'] = pd.DataFrame(
        'Gene', index=genes, columns=['Node_type'])
    nx_tables['Node']['Gene']['Gene'] = genes
    regions = list(set(er_metadata['Region']))
    nx_tables['Node']['Region'] = pd.DataFrame(
        'Region', index=regions, columns=['Node_type'])
    nx_tables['Node']['Region']['Region'] = regions
    # Add gene logFC
    if add_differential_gene_expression is True:
        for var in differential_variable:
            nx_tables['Node']['TF'] = pd.concat([nx_tables['Node']['TF'], get_log2fc_nx(
                scplus_obj, var, nx_tables['Node']['TF'].index.tolist(), contrast='gene')], axis=1)
            nx_tables['Node']['Gene'] = pd.concat([nx_tables['Node']['Gene'], get_log2fc_nx(
                scplus_obj, var, nx_tables['Node']['Gene'].index.tolist(), contrast='gene')], axis=1)
    if add_differential_region_accessibility is True:
        for var in differential_variable:
            nx_tables['Node']['Region'] = pd.concat([nx_tables['Node']['Region'], get_log2fc_nx(
                scplus_obj, var, nx_tables['Node']['Region'].index.tolist(), contrast='region')], axis=1)
    return nx_tables


def format_nx_table_internal(nx_tables, table_type, table_id, color_by={}, transparency_by={}, size_by={}, shape_by={}, label_size_by={}, label_color_by={}):
    """
    TO DO
    """
    nx_tb = nx_tables[table_type][table_id]
    # Color
    if table_id in color_by.keys():
        if 'fixed_color' not in color_by[table_id].keys():
            color_var = nx_tables[table_type][table_id][color_by[table_id]['variable']]
            if 'category_color' in color_by[table_id].keys():
                if color_by[table_id]['category_color'] is None:
                    random.seed(555)
                    categories = set(color_var)
                    color = list(map(
                        lambda i: "#" +
                        "%06x" % random.randint(
                            0, 0xFFFFFF), range(len(categories))
                    ))
                    color_dict = dict(zip(categories, color))
                else:
                    color_dict = color_by[table_id]['category_color']
                color = color_var.apply(
                    lambda x: to_rgba(color_dict[x])).to_numpy()
            elif 'continuous_color' in color_by[table_id].keys():
                if color_by[table_id]['continuous_color'] is None:
                    color_map = 'viridis'
                else:
                    color_map = color_by[table_id]['continuous_color']
                if 'v_min' in color_by[table_id].keys():
                    v_min = color_by[table_id]['v_min']
                else:
                    v_min = None
                if 'v_max' in color_by[table_id].keys():
                    v_max = color_by[table_id]['v_max']
                else:
                    v_max = None
                color = get_colors(color_var, color_map, v_min, v_max)
        else:
            color = np.array([color_by[table_id]['fixed_color']]
                             * nx_tables[table_type][table_id].shape[0])
    else:
        color = np.array([to_rgba('grey')] *
                         nx_tables[table_type][table_id].shape[0])

    # Transparency
    if table_id in transparency_by.keys():
        if 'fixed_alpha' not in transparency_by[table_id]['variable']:
            transparency_var = nx_tables[table_type][table_id][transparency_by[table_id]['variable']]
            if 'v_min' in transparency_by[table_id].keys():
                v_min = transparency_by[table_id]['v_min']
            else:
                v_min = None
            if 'v_max' in transparency_by[table_id].keys():
                v_max = transparency_by[table_id]['v_max']
            else:
                v_max = None
            if 'min_alpha' in transparency_by[table_id].keys():
                min_alpha = transparency_by[table_id]['min_alpha']
            else:
                min_alpha = 0.5
            norm = plt.Normalize(v_min, v_max)
            x = norm(transparency_var)
            x[x < min_alpha] = min_alpha
            color[:, -1] = x
        else:
            color[:, -1] = [transparency_by[table_id]['fixed_alpha']]

    # Size/Width
    if table_id in size_by.keys():
        if 'fixed_size' not in size_by[table_id].keys():
            sw_var = nx_tables[table_type][table_id][size_by[table_id]
                                                     ['variable']].to_numpy().flatten('F')
            if 'p_min' in size_by[table_id].keys():
                p_min = size_by[table_id]['min_size']
            else:
                p_min = 3
            if 'p_max' in size_by[table_id].keys():
                p_max = size_by[table_id]['max_size']
            else:
                p_max = 10
            s_min = sw_var[sw_var != 0].min()
            s_max = sw_var.max()
            sw_var[sw_var != 0] = p_min + \
                (sw_var[sw_var != 0] - s_min) * \
                ((p_max - p_min) / (s_max - s_min))
        else:
            sw_var = [size_by[table_id]['fixed_size']] * \
                nx_tables[table_type][table_id].shape[0]
    else:
        sw_var = [1] * nx_tables[table_type][table_id].shape[0]

    # Node shape
    if table_id in shape_by.keys():
        if 'fixed_shape' not in shape_by[table_id].keys():
            if not 'categorical_shape' in shape_by[table_id].keys():
                print(
                    'No categorical_shape dictionary provided, making all nodes circular!')
                shape_var = ['circular'] * \
                    nx_tables[table_type][table_id].shape[0]
            else:
                shape_dict = shape_by[table_id]['categorical_shape']
                shape_var = shape_var.apply(lambda x: shape_dict[x]).to_numpy()
        else:
            shape_var = np.array(
                [shape_by[table_id]['fixed_shape']]*nx_tables[table_type][table_id].shape[0])
    else:
        shape_var = ['ellipse'] * nx_tables[table_type][table_id].shape[0]

    # Label size
    if table_id in label_size_by.keys():
        if 'fixed_label_size' not in label_size_by[table_id].keys():
            if not 'categorical_label_size' in label_size_by[table_id].keys():
                print(
                    'categorical_label_size dictionary provided, using size 14 for all nodes!')
                label_size_var = 14 * nx_tables[table_type][table_id].shape[0]
            else:
                label_size_dict = label_size_by[table_id]['categorical_label_size']
                label_size_var = label_size_var.apply(
                    lambda x: label_size_var[x]).to_numpy()
        else:
            label_size_var = np.array(
                [label_size_by[table_id]['fixed_label_size']]*nx_tables[table_type][table_id].shape[0])
    else:
        label_size_var = [14] * nx_tables[table_type][table_id].shape[0]

    # Label color
    if table_id in label_color_by.keys():
        if 'fixed_label_color' not in label_color_by[table_id].keys():
            if not 'categorical_label_color' in label_color_by[table_id].keys():
                print(
                    'categorical_label_color dictionary provided, using black for all nodes!')
                label_color_var = np.array(
                    [to_rgba('black')]*nx_tables[table_type][table_id].shape[0])
            else:
                label_color_dict = label_color_by[table_id]['categorical_label_color']
                label_color_var = label_color_var.apply(
                    lambda x: to_rgba(label_color_var[x])).to_numpy()
        else:
            label_color_var = np.array(
                [label_color_by[table_id]['fixed_label_color']]*nx_tables[table_type][table_id].shape[0])
    else:
        label_color_var = np.array(
            [to_rgba('black')]*nx_tables[table_type][table_id].shape[0])

    color = [to_hex(x, keep_alpha=True) for x in color]
    label_color_var = [to_hex(x, keep_alpha=True) for x in label_color_var]
    if table_type == 'Edge':
        dt1 = nx_tb.iloc[:, 0:2].reset_index(drop=True)
        dt2 = pd.DataFrame([color, sw_var]).T.reset_index(drop=True)
    else:
        dt1 = nx_tb.iloc[:, 0:2].reset_index(drop=True)
        dt2 = pd.DataFrame([color, sw_var, shape_var, label_size_var,
                           label_color_var]).T.reset_index(drop=True)

    dt = pd.concat([dt1, dt2], axis=1)
    if table_type == 'Edge':
        dt.columns = ['source', 'target', 'color', 'width']
    else:
        dt.columns = ['group', 'label', 'color',
                      'size', 'shape', 'font_size', 'font_color']
    return dt


def get_colors(inp, cmap_name, vmin=None, vmax=None):
    """
    A function to get color values from a continuous vector and a color map
    """
    color_map = cm.get_cmap(cmap_name)
    norm = plt.Normalize(vmin, vmax)
    return color_map(norm(inp))


def create_nx_graph(nx_tables,
                    use_edge_tables,
                    color_edge_by={},
                    transparency_edge_by={},
                    width_edge_by={},
                    color_node_by={},
                    transparency_node_by={},
                    size_node_by={},
                    shape_node_by={},
                    label_size_by={},
                    label_color_by={}):
    """
    TO DO
    """
    # Get node table names
    use_node_tables = []
    if 'TF2R' in use_edge_tables:
        use_node_tables = ['TF', 'Region'] + use_node_tables
    if 'TF2G' in use_edge_tables:
        use_node_tables = ['TF', 'Gene'] + use_node_tables
    if 'R2G' in use_edge_tables:
        use_node_tables = ['Region', 'Gene'] + use_node_tables
    use_node_tables = sorted(list(set(use_node_tables)), reverse=True)

    # Create graph
    edge_tables = pd.concat([format_nx_table_internal(
        nx_tables, 'Edge', x, color_edge_by, transparency_edge_by, width_edge_by, {}) for x in use_edge_tables])
    G = nx.from_pandas_edgelist(edge_tables, edge_attr=True)
    # Add node tables
    node_tables = pd.concat([format_nx_table_internal(nx_tables, 'Node', x, color_node_by, transparency_node_by,
                            size_node_by, shape_node_by, label_size_by, label_color_by) for x in use_node_tables])
    node_tables.index = node_tables['label']
    node_tables_d = node_tables.to_dict()
    for key in node_tables_d.keys():
        if 'font' not in key:
            nx.set_node_attributes(G, node_tables_d[key], name=key)
    font_nt_d = node_tables[['font_size', 'font_color']]
    font_nt_d.columns = ['size', 'color']
    font_nt_d = font_nt_d.to_dict(orient='index')
    nx.set_node_attributes(G, font_nt_d, name='font')
    return G, edge_tables, node_tables
