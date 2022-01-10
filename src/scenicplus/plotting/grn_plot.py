
import numpy as np
from networkx.classes import Graph
from typing import List
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import FancyArrowPatch
from sklearn.cluster import AgglomerativeClustering
from ..utils import Groupby

LAYOUT_EDGE_NON_HIGHLIGHT = {
    'color': 'gray',
    'alpha': 0.1,
    'lw': 0.5
}

LAYOUT_EDGE_HIGHLIGHT = {
    'color': 'black',
    'alpha': 1,
    'lw': 1
}

LAYOUT_TF_NODE_NH = {
    'node_size': 150,
    'node_color': '#55505C',
    'alpha': 0.1,
    'label': False
}

LAYOUT_TF_NODE_H = {
    'node_size': 3000,
    'node_color': '#55505C',
    'alpha': 1,
    'label': True
}

LAYOUT_GENE_NODE_NH = {
    'node_size': 150,
    'node_color': '#55505C',
    'alpha': 0.1,
    'label': False
}

LAYOUT_GENE_NODE_H = {
    'node_size': 150,
    'node_color': '#55505C',
    'alpha': 1,
    'label': False
}

LAYOUT_REGION_NODE_NH = {
    'node_size': 3,
    'node_color': '#749C75',
    'alpha': 0.1,
    'label': False
}

LAYOUT_REGION_NODE_H = {
    'node_size': 10,
    'node_color': '#BEEE62',
    'alpha': 1,
    'label': False
}

POSSIBLE_LAYOUTS = LAYOUT_TF_NODE_NH.keys()


def _distance(p1, p2):
    x1 = p1[0]
    x2 = p2[0]
    y1 = p1[1]
    y2 = p2[1]

    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def _pairwise_distance(points):
    distances = np.zeros((points.shape[0], points.shape[0]))
    for i in range(points.shape[0]):
        for j in range(points.shape[0]):
            distances[i, j] = _distance(points[i], points[j])
    np.fill_diagonal(distances, np.NINF)
    return distances


def _line_two_points(p1, p2, return_func=True):
    x1 = p1[0]
    x2 = p2[0]
    y1 = p1[1]
    y2 = p2[1]

    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    if return_func:
        return lambda x: m * x + b
    else:
        return m, b


def _line_slope_point(m, p, return_func=True):
    x = p[0]
    y = p[1]
    b = y - m * x

    if return_func:
        return lambda x: m * x + b
    else:
        return m, b


def layout_grn(G: Graph,
               dist_genes=1,
               dist_TF=0.1):
    node_type = nx.get_node_attributes(G, 'type')
    TF_nodes = [n for n in G.nodes if node_type[n] == 'TF']
    region_nodes = [n for n in G.nodes if node_type[n] == 'region']
    gene_nodes = [n for n in G.nodes if node_type[n] == 'gene']

    # get regions with TFs as target
    tmp = pd.DataFrame(list(G.edges))
    regions_targetting_TFs = tmp.loc[np.isin(tmp[1], TF_nodes), 0].to_list()
    del(tmp)
    region_nodes = list(set(region_nodes) - set(regions_targetting_TFs))

    # layout regions in a circle
    n_region_nodes = len(region_nodes)

    theta = np.linspace(0, 1, n_region_nodes + 1)[:-1] * 2 * np.pi
    theta = theta.astype(np.float32)
    pos_regions = np.column_stack(
        [np.cos(theta), np.sin(theta), np.zeros((n_region_nodes, 0))]
    )

    # sort regions by target
    source_target_dict = {}
    target_source_dict = {}
    for edge in G.edges:
        source = edge[0]
        target = edge[1]

        if source in source_target_dict.keys():
            if target not in source_target_dict[source]:
                source_target_dict[source].append(target)
        else:
            source_target_dict[source] = [target]

        if target in target_source_dict.keys():
            if source not in target_source_dict[target]:
                target_source_dict[target].append(source)
        else:
            target_source_dict[target] = [source]

    region_nodes = sorted(region_nodes, key=lambda x: target_source_dict[x][0])
    pos_regions = dict(zip(region_nodes, pos_regions))

    # layout target genes in concentric circle around regions
    pos_genes = {}
    additional_genes_to_position = []
    for gene in gene_nodes:
        # get regions targetting this gene and their position
        regions = target_source_dict[gene]
        if all([r in regions_targetting_TFs for r in regions]):
            additional_genes_to_position.append(gene)
            continue
        pos_regions_gene = np.array(
            [pos_regions[r] for r in regions if r not in regions_targetting_TFs])
        if len(regions) > 1:
            # get the positions which are furthest apart and "draw" a line through them
            pairwise_distances = _pairwise_distance(pos_regions_gene)
            furthest_points = np.unravel_index(
                pairwise_distances.argmax(), pairwise_distances.shape)
            m, b = _line_two_points(
                pos_regions_gene[furthest_points[0]], pos_regions_gene[furthest_points[1]], return_func=False)
            # draw a perpendicular line through the first line and the mean position
            p_mean = pos_regions_gene.mean(0)
            m, b = _line_slope_point(-1/m, p_mean, return_func=False)
            # get the point which is a distance dist_genes away from point p_mean
            p_new = [p_mean[0] - dist_genes * np.sqrt(1 / (1 + m**2)),
                     p_mean[1] - m * dist_genes * np.sqrt(1 / (1 + m**2))]
            # check if point is within the circle, otherwise take the other point (radius of the circle is 1)
            if p_new[0]**2 + p_new[1]**2 < 1:
                p_new = [p_mean[0] + dist_genes * np.sqrt(1 / (1 + m**2)),
                         p_mean[1] + m * dist_genes * np.sqrt(1 / (1 + m**2))]
        else:
            # draw line trough origin and pos of region
            m, b = _line_two_points(pos_regions_gene[0], [
                                    0, 0], return_func=False)
            # get the point which is a distance dist_genes away from point pos_regions_gene
            p_new = [pos_regions_gene[0][0] - dist_genes * np.sqrt(1 / (1 + m**2)),
                     pos_regions_gene[0][1] - m * dist_genes * np.sqrt(1 / (1 + m**2))]
            # check if point is within the circle, otherwise take the other point (radius of the circle is 1)
            if p_new[0]**2 + p_new[1]**2 < 1:
                p_new = [pos_regions_gene[0][0] + dist_genes * np.sqrt(1 / (1 + m**2)),
                         pos_regions_gene[0][1] + m * dist_genes * np.sqrt(1 / (1 + m**2))]
        pos_genes[gene] = np.array(p_new)

    pos_TF = {}
    for TF in TF_nodes:
        # get regions targetted by this TF and their position
        regions = source_target_dict[TF]
        if all([r in regions_targetting_TFs for r in regions]):
            additional_genes_to_position.append(TF)
            continue
        pos_regions_TF = np.array(
            [pos_regions[r] for r in regions if r not in regions_targetting_TFs])
        if len(regions) > 1:
            # get the positions which are furthest apart and "draw" a line through them
            pairwise_distances = _pairwise_distance(pos_regions_TF)
            furthest_points = np.unravel_index(
                pairwise_distances.argmax(), pairwise_distances.shape)
            m, b = _line_two_points(
                pos_regions_TF[furthest_points[0]], pos_regions_TF[furthest_points[1]], return_func=False)
            # draw a perpendicular line through the first line and the mean position
            p_mean = pos_regions_TF.mean(0)
            m, b = _line_slope_point(-1/m, p_mean, return_func=False)
            # get the point which is a distance dist_genes away from point p_mean
            p_new = [p_mean[0] - dist_TF * np.sqrt(1 / (1 + m**2)),
                     p_mean[1] - m * dist_TF * np.sqrt(1 / (1 + m**2))]
            # check if point is within the circle, otherwise take the other point (radius of the circle is 1)
            if p_new[0]**2 + p_new[1]**2 > 1:
                p_new = [p_mean[0] + dist_TF * np.sqrt(1 / (1 + m**2)),
                         p_mean[1] + m * dist_TF * np.sqrt(1 / (1 + m**2))]
        else:
            # draw line trough origin and pos of region
            m, b = _line_two_points(
                pos_regions_TF[0], [0, 0], return_func=False)
            # get the point which is a distance dist_genes away from point pos_regions_gene
            p_new = [pos_regions_TF[0][0] - dist_TF * np.sqrt(1 / (1 + m**2)),
                     pos_regions_TF[0][1] - m * dist_TF * np.sqrt(1 / (1 + m**2))]
            # check if point is within the circle, otherwise take the other point (radius of the circle is 1)
            if p_new[0]**2 + p_new[1]**2 < 1:
                p_new = [pos_regions_TF[0][0] + dist_TF * np.sqrt(1 / (1 + m**2)),
                         pos_regions_TF[0][1] + m * dist_TF * np.sqrt(1 / (1 + m**2))]
        pos_TF[TF] = np.array(p_new)

    # layout TF nodes within circle
    #G_TF = G.subgraph(nodes = [*TF_nodes, *regions_targetting_TFs, *additional_genes_to_position])
    #additional_genes_to_position_init = {gene: [random.uniform(0, 1), random.uniform(0, 1)] for gene in additional_genes_to_position}
    #pos_TF = nx.spring_layout(G_TF, scale = 0.7, pos = {**pos_TF, **additional_genes_to_position_init})
    G_add = G.subgraph(nodes=additional_genes_to_position)
    pos_add = nx.spring_layout(G_add, scale=0.1)

    G_regions_TF = G.subgraph(nodes=[*regions_targetting_TFs, *TF_nodes])
    pos_regions_TF = nx.spring_layout(G_regions_TF, scale=1)
    pos_regions_TF = {k: pos_regions_TF[k] for k in pos_regions_TF.keys(
    ) if k in regions_targetting_TFs}

    return {**pos_TF, **pos_regions, **pos_genes, **pos_add, **pos_regions_TF}


def draw_TF_edges(G: Graph,
                  pos: dict,
                  ax: plt.axes,
                  d: float = 0.05,
                  distance_threshold: float = 4,
                  arrowstyle: str = '->',
                  mutation_scale: int = 6,
                  return_drawn_edges: bool = True,
                  TFs_to_highlight: List[str] = [],
                  non_high_light_layout=LAYOUT_EDGE_NON_HIGHLIGHT,
                  high_light_layout=LAYOUT_EDGE_HIGHLIGHT):
    node_type = nx.get_node_attributes(G, 'type')
    TF_nodes = [n for n in G.nodes if node_type[n] == 'TF']
    tmp = pd.DataFrame(list(G.edges))
    regions_targetting_TFs = tmp.loc[np.isin(tmp[1], TF_nodes), 0].to_list()
    del(tmp)

    source_target_dict = {}
    for edge in G.edges:
        source = edge[0]
        target = edge[1]

        if source in source_target_dict.keys():
            if target not in source_target_dict[source]:
                source_target_dict[source].append(target)
        else:
            source_target_dict[source] = [target]
    if return_drawn_edges:
        edges_drawn = []
    # this set operation makes sure TFs_to_highlight are drawn on top
    for TF in list(set(TF_nodes) - set(TFs_to_highlight)) + TFs_to_highlight:
        layout = high_light_layout if TF in TFs_to_highlight else non_high_light_layout
        TF_target_regions_on_circle = set(
            source_target_dict[TF]) - set(regions_targetting_TFs)
        TF_target_regions_on_circle_pos = np.array(
            [pos[r] for r in TF_target_regions_on_circle])
        if return_drawn_edges:
            for region in TF_target_regions_on_circle:
                edges_drawn.append((TF, region))
        # get position of TF
        posTF = pos[TF]

        # cluster regions on cicrle, to each cluster a straight line will be drawn
        clustering = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=None).fit(
            TF_target_regions_on_circle_pos
        )
        grouper = Groupby(clustering.labels_)

        # draw lines to each cluster
        for ind in grouper.indices:
            # get positions of subcluster
            region_pos = TF_target_regions_on_circle_pos[ind]
            # get midde postion of these regions and "draw" a virtual line from posTF to p_mean
            p_mean = region_pos.mean(0)
            m, b = _line_two_points(posTF, p_mean, return_func=False)
            # get a new point which is on this virtual line and d units towards posTF
            p_new = [p_mean[0] - d * np.sqrt(1 / (1 + m**2)),
                     p_mean[1] - m * d * np.sqrt(1 / (1 + m**2))]
            if _distance(p_new, posTF) > _distance(p_mean, posTF):
                p_new = [p_mean[0] + d * np.sqrt(1 / (1 + m**2)),
                         p_mean[1] + m * d * np.sqrt(1 / (1 + m**2))]
            # draw a straight line from posTF to p_new and curved lines from p_new to the region positions
            for target_point in region_pos:
                verts = [
                    tuple(posTF),  # start of straight line
                    tuple(p_new),  # end of straight line
                    tuple(p_mean),  # control point for bezier curve
                    tuple(target_point)  # end point
                ]
                codes = [
                    Path.MOVETO,
                    Path.LINETO,
                    Path.CURVE3,
                    Path.CURVE3
                ]
                path = Path(verts, codes)
                con = FancyArrowPatch(
                    path=path, arrowstyle=arrowstyle, mutation_scale=mutation_scale, **layout)
                ax.add_artist(con)
    if return_drawn_edges:
        return edges_drawn

# main function


def draw_grn(G: Graph,
             ax: plt.axes,
             dist_genes=1,
             dist_TF=0.1,
             d: float = 0.05,
             distance_threshold: float = 4,
             arrowstyle: str = '->',
             mutation_scale: int = 6,
             TFs_to_highlight: List[str] = [],
             all_target_genes_need_to_be_targeted_by_TF: bool = False,
             label_font_color='#FAA916',
             return_pos=True,
             pos=None):
    """
    Draws eGRN with TFs in the middle, regions forming a cirle and target genes forming a concentric circle around the regions

    Parameters
    ----------
    G
        An instance of :class:`networkx.DiGraph` directed graph
    ax
        An instance of :class:`matplotlib.pyplot.axes`
    dist_genes
        A float specifying how far the target genes should be pushed away from the regions
        default: 1
    dist_TF
        A float specifying how far the TFs should be pushed away from the regions
        default: 0.1
    d
        A float specifying until which distance (calculated from the regions) a straight line should be drawn from TF to regions
        default: 0.05
    distance_threshold
        Threshold on AgglomerativeClustering function to define region clusters in space
        default: 4
    arrowstyle
        Style of the arrows drawn by networkx draw function
        default: '->'
    mutation_scale
        Scale factor of the arrows
        default: 6
    TFs_to_highlight
        List specifying which TF to highlight
        default: []
    all_target_genes_need_to_be_targeted_by_TF
        If set to True only highlight a target gene when it is only targetted by TFs in the list :param:`TFs_to_highlight`
        default: False
    label_font_color
        Font color of the labels
        default: '#FAA916'
    return_pos
        Boolean specifying wether or not to return calculated node positions
        default: True
    pos
        Parameter to provide precomputed positions.
        Default: None

    """

    # layout nodes
    node_type = nx.get_node_attributes(G, 'type')
    TF_nodes = [n for n in G.nodes if node_type[n] == 'TF']
    region_nodes = [n for n in G.nodes if node_type[n] == 'region']
    gene_nodes = [n for n in G.nodes if node_type[n] == 'gene']

    source_target_dict = {}
    target_source_dict = {}
    for edge in G.edges:
        source = edge[0]
        target = edge[1]

        if source in source_target_dict.keys():
            if target not in source_target_dict[source]:
                source_target_dict[source].append(target)
        else:
            source_target_dict[source] = [target]

        if target in target_source_dict.keys():
            if source not in target_source_dict[target]:
                target_source_dict[target].append(source)
        else:
            target_source_dict[target] = [source]

    # if TF in TF to highlight use highlight layout
    TF_node_layout = {}
    for TF in TF_nodes:
        if TF in TFs_to_highlight:
            TF_node_layout[TF] = LAYOUT_TF_NODE_H
        else:
            TF_node_layout[TF] = LAYOUT_TF_NODE_NH

    # if region is targeted by highlighted TF, use highlight layout
    region_node_layout = {}
    for region in region_nodes:
        if any([TF in TFs_to_highlight for TF in target_source_dict[region]]):
            region_node_layout[region] = LAYOUT_REGION_NODE_H
        else:
            region_node_layout[region] = LAYOUT_REGION_NODE_NH

    # if gene is targeted by a/all region(s) which is targetded by a highlighed TF, use highlight layout
    gene_node_layout = {}
    for gene in gene_nodes:
        regions_targetting_gene = target_source_dict[gene]
        if all_target_genes_need_to_be_targeted_by_TF:
            if all([tf in TFs_to_highlight for r in regions_targetting_gene for tf in target_source_dict[r]]):
                gene_node_layout[gene] = LAYOUT_GENE_NODE_H
            else:
                gene_node_layout[gene] = LAYOUT_GENE_NODE_NH
        else:
            if any([tf in TFs_to_highlight for r in regions_targetting_gene for tf in target_source_dict[r]]):
                gene_node_layout[gene] = LAYOUT_GENE_NODE_H
            else:
                gene_node_layout[gene] = LAYOUT_GENE_NODE_NH

    node_layouts = {**TF_node_layout, **region_node_layout, **gene_node_layout}
    # set node labels
    for node in node_layouts.keys():
        if node_layouts[node]['label']:
            node_layouts[node]['label'] = node
        else:
            node_layouts[node]['label'] = ''

    if pos is None:
        # get node position
        pos = layout_grn(
            G=G,
            dist_genes=dist_genes,
            dist_TF=dist_TF
        )

    layouts = {k: [node_layouts[node][k] for node in G.nodes]
               for k in (set(POSSIBLE_LAYOUTS) - set(['label']))}

    # draw nodes with labels for h
    nx.draw_networkx_nodes(
        G,
        pos=pos,
        ax=ax,
        **layouts
    )
    labels = {k: node_layouts[k]['label'] for k in node_layouts.keys()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=20, font_family='sans-serif',
                            font_color=label_font_color, font_weight='bold', ax=ax)

    # draw TF edges
    edges_drawn = draw_TF_edges(
        G,
        pos,
        ax,
        d,
        distance_threshold,
        arrowstyle,
        mutation_scale,
        True,
        TFs_to_highlight)

    # layout other edges
    remaining_edges = list(set(G.edges) - set(edges_drawn))

    layouts_edges = []
    # if edge is region is targeted by highlighted TF, highlight
    for edge in remaining_edges:
        if edge[0] in region_nodes:
            if any([tf in TFs_to_highlight for tf in target_source_dict[edge[0]]]):
                layouts_edges.append(LAYOUT_EDGE_HIGHLIGHT)
            else:
                layouts_edges.append(LAYOUT_EDGE_NON_HIGHLIGHT)
        elif edge[0] in TF_nodes:
            if edge[0] in TFs_to_highlight:
                layouts_edges.append(LAYOUT_EDGE_HIGHLIGHT)
            else:
                layouts_edges.append(LAYOUT_EDGE_NON_HIGHLIGHT)
    possible_layouts = layouts_edges[0].keys()
    layouts = {k: [l[k] for l in layouts_edges] for k in possible_layouts}
    if 'color' in layouts.keys():
        layouts['edge_color'] = layouts.pop('color')
    if 'lw' in layouts.keys():
        layouts['width'] = layouts.pop('lw')
    # draw other edges
    nx.draw_networkx_edges(
        G,
        edgelist=remaining_edges,
        pos=pos,
        ax=ax,
        **layouts
    )
    ax.axis('off')

    if return_pos:
        return pos
