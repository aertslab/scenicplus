import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.cluster import AgglomerativeClustering
import scanpy as sc
from matplotlib.cm import ScalarMappable

def _cluster_labels_to_idx(labels):
        counter = 0
        order = np.zeros(labels.shape, dtype = int)
        for i in range(0, labels.max() + 1):
                for j in np.where(labels == i)[0]:
                        order[j] = counter
                        counter += 1
        idx = [np.where(order == i)[0][0] for i, _ in enumerate(order)]
        return idx

def dotplot_given_ax(expr_mat: pd.DataFrame,
                     grouping_vect: pd.Series,
                     df_cistromes: pd.DataFrame,
                     ax: plt.axes,
                     log_transform_expr: bool = True,
                     normalize_expr: bool = True,
                     normalize_expr_target_sum: int = 1e4,
                     cluster: str = 'both', #can be group, TF or both
                     n_clust = 2,
                     min_point_size: float = 3,
                     max_point_size: float = 30,
                     cmap: str = 'cool',
                     vmin: float = 0,
                     vmax: float = None,
                     x_tick_rotation: float = 45,
                     x_tick_ha: str = 'right', 
                     fontsize: float = 9,
                     z_score_expr: bool = True,
                     eRegulons: list = None,
                     eRegulon_lw: float = 1,
                     eRegulon_lc: str = 'black',
                     grid_color = 'grey', 
                     grid_lw = 0.5):

    # Preprocess expression data
    if log_transform_expr or normalize_expr:
        adata = sc.AnnData(expr_mat)
    
    if normalize_expr:
        sc.pp.normalize_total(adata, target_sum = normalize_expr_target_sum)

    if log_transform_expr:
        sc.pp.log1p(adata)

    if log_transform_expr or normalize_expr:
        expr_mat = adata.to_df()

    # Get mean expression for TFs
    grouping_vect.name = 'group' # Set grouping_vector name to 'group' for later merging
    mean_expr_mat = expr_mat.groupby(grouping_vect).mean()
    mean_expr_mat = mean_expr_mat[set(df_cistromes['TF']) & set(mean_expr_mat.columns)]
    mean_expr = mean_expr_mat.melt(ignore_index = False)
    mean_expr.reset_index(inplace = True)
    mean_expr.rename({'variable': 'TF', 'value': 'mean_expr'}, axis = 1, inplace = True)

    # Combine mean expression and max nes in single df
    df_dotplot = df_cistromes[['group', 'max_nes', 'TF']].drop_duplicates()
    df_dotplot = df_dotplot.merge(mean_expr, on = ['group', 'TF'])

    # Seperate dotsize data (max nes) and dot color data (mean expr)
    dotsizes = df_dotplot[['group', 'TF', 'max_nes']].pivot_table(index = 'group', columns = 'TF').fillna(0)['max_nes']
    dotcolors = df_dotplot[['group', 'TF', 'mean_expr']].pivot_table(index = 'group', columns = 'TF').fillna(0)['mean_expr']

    # Cluster
    if cluster == 'both' or cluster == 'group':
        clustering_groups = AgglomerativeClustering(n_clusters = n_clust).fit(dotsizes)
        ordered_idx_grps = _cluster_labels_to_idx(clustering_groups.labels_)
        dotsizes = dotsizes.iloc[ordered_idx_grps, :]
        dotcolors = dotcolors.iloc[ordered_idx_grps, :]
    if cluster == 'both' or cluster == 'TF':
        clustering_TFs = AgglomerativeClustering(n_clusters = n_clust).fit(dotsizes.T)
        ordered_idx_TFs = _cluster_labels_to_idx(clustering_TFs.labels_)
        dotsizes = dotsizes.iloc[:, ordered_idx_TFs]
        dotcolors = dotcolors.iloc[:, ordered_idx_TFs]
    
    # Calculate Z-score for expression values
    if z_score_expr:
        u = dotcolors.mean()
        sd = dotcolors.std()
        dotcolors = (dotcolors - u) / sd
        vmin = dotcolors.min().min()
    
    # Generate plotting data
    n_group = len(set(df_dotplot['group']))
    n_TF = len(set(df_dotplot['TF']))

    # Generate a grid
    x = np.tile( np.arange(n_group), n_TF)
    y = [int(i / n_group) for i, _ in enumerate(x)]

    # Get dot sizes
    s = dotsizes.to_numpy().flatten('F')

    # Scale dotsizes
    s_min = (s[s != 0]).min()
    s_max = s.max()
    
    #scale values between min_point_size and max_point_size keep zero values at 0
    s[s != 0] = min_point_size + (s[s != 0]- s_min) * ((max_point_size - min_point_size) / (s_max - s_min)) 

    # Get dot colors
    c = dotcolors.to_numpy().flatten('F')

    # Get edges for eRegulons
    TFs = dotsizes.columns
    groups = dotsizes.index
    linewidths = [eRegulon_lw if (TF, celltype) in eRegulons else 0 for 
                    TF, celltype in zip(np.repeat(TFs, n_group), np.tile(groups, n_TF))]

    if vmax is None:
        vmax = c.max()
    
    norm = Normalize(vmin = vmin, vmax = vmax)

    ax.set_axisbelow(True)
    ax.grid(color = grid_color, linewidth = grid_lw)
    scat = ax.scatter(x, y, s = s, c = c, cmap = cmap, norm = norm, edgecolors = eRegulon_lc, linewidths = linewidths)
    # set x ticks
    ax.set_xticks( np.arange(n_group) )
    ax.set_xticklabels( dotsizes.index, rotation = x_tick_rotation, ha = x_tick_ha, fontdict = {'fontsize' : fontsize} )
    # set y ticks
    ax.set_yticks( np.arange(n_TF) )
    ax.set_yticklabels(dotsizes.columns, fontdict = {'fontsize': fontsize})

    
    #draw colorbar
    cbar = plt.colorbar(mappable = ScalarMappable(norm = norm, cmap = cmap), 
                 ax = ax, location = 'bottom', orientation = 'horizontal', 
                 aspect = 10, shrink = 0.4, pad = 0.10, anchor = (0, 0))
    cbar_label = 'Z-score mean expression' if z_score_expr else 'Mean expression'
    cbar.set_label(cbar_label)
    
    L = plt.legend(*scat.legend_elements("sizes", num=3), 
               loc = 'lower right', bbox_to_anchor = (0.8, -0.2), frameon = False, title = 'max nes', ncol = 1, mode = None)
    #recalculate original scale
    to_int = lambda x: int(''.join(i for i in x if i.isdigit()))
    labels = np.array([to_int(t.get_text()) for t in L.get_texts()])
    re_scale = lambda x: (x - min_point_size) * ((s_max - s_min) / (max_point_size - min_point_size)) + s_min
    rescaled = re_scale(labels).round(2)
    for new, text in zip(rescaled, L.get_texts()):
        text.set_text(new)
    return ax
