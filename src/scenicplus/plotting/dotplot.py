import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.cluster import AgglomerativeClustering
import scanpy as sc
from matplotlib.cm import ScalarMappable
import plotly
import pycistarget
import kaleido
import matplotlib.backends.backend_pdf
from typing import Dict, List, Tuple
from typing import Optional, Union
import re
from pycistarget.motif_enrichment_dem import DEM

## Utils
def flatten(A):
    """
    Utils function to flatten lists
    """
    rt = []
    for i in A:
        if isinstance(i,list): rt.extend(flatten(i))
        else: rt.append(i)
    return rt

# For motif enrichment
def generate_dotplot_df_motif_enrichment(scplus_obj: 'SCENICPLUS',
                       enrichment_key: str,
                       group_variable: str = None,
                       barcode_groups: dict = None,
                       subset: list = None,
                       subset_TFs: list = None,
                       use_pseudobulk: bool = False,
                       use_only_direct: bool = False,
                       normalize_expression: bool = True,
                       standardize_expression:bool = False):
    """
    Function to generate dotplot dataframe from motif enrichment results
    
    Parameters
    ---------
    scplus_obj: `class::SCENICPLUS`
        A :class:`SCENICPLUS` object with motif enrichment results from pycistarget (`scplus_obj.menr`).
    enrichment_key: str
        Key of the motif enrichment result to use.
    group_variable: str, optional
        Group variable to use to calculate TF expression per group. Levels of this variable should match with the entries in the
        selected motif enrichment dictionary. Only required if barcode groups is not provided.
    barcode_groups: Dict, optional
        A dictionary containing cell barcodes per class to calculate TF expression per group. Keys of the dictionary should match
        the keys in the selected motif enrichment dictionary. Only required if group_variable is not provided.
    subset: list, optional
        Subset of classes to use. Default: None (use all)
    subset_TFs: list, optional
        List of TFs to use. Default: None (use all)
    use_pseudobulk: bool, optional
        Whether to use pseudobulk to calculate gene expression. Only available if there is pseudobulk profiles for group_variable.
        Default: False
    use_only_direct: bool, optional
        Whether to only use NES values of directly annotated motifs. Default: False
    normalize_expression: bool, optional
        Whether to log(CPM) normalize gene expression. Default: True
    standardize_expression: bool, optional
        Wheter to standarize gene expression between o and 1. Default: False
        
    Returns
    ---------
        A data frame with enrichment values per TF and cell group to use as input for `dotplot()`.
    """
    # Get data
    ## Gene expression
    ### If using pseudobulk
    if use_pseudobulk:
        dgem = scplus_obj.uns['Pseudobulk'][group_variable]['Expression'].copy()
        if group_variable is not None:
            cell_data = pd.DataFrame([x.rsplit('_', 1)[0] for x in dgem.columns], 
                                 index=dgem.columns).iloc[:,0]
    ### If using all
    else:
        dgem = pd.DataFrame(scplus_obj.X_EXP, index=scplus_obj.cell_names, columns=scplus_obj.gene_names).copy().T
        if group_variable is not None:
            cell_data = scplus_obj.metadata_cell.loc[scplus_obj.cell_names, group_variable]
    ## Should gene expression be normalized?
    if normalize_expression:
        dgem = dgem.T / dgem.T.sum(0) * 10**6
        dgem = np.log1p(dgem).T
    ### Checking motif enrichment data
    menr = scplus_obj.menr[enrichment_key]
    if isinstance(menr, DEM):
        menr = scplus_obj.menr[enrichment_key].motif_enrichment.copy()
        menr_df = pd.concat([menr[x] for x in menr.keys()])
        score_keys = ['Log2FC', 'Adjusted_pval']
        columns = flatten(['Contrast', score_keys])
    else:
        menr = scplus_obj.menr[enrichment_key].copy()
        menr_df = pd.concat([menr[x].motif_enrichment for x in menr.keys()])
        score_keys = ['NES']
        columns = flatten(['Region_set', score_keys])
        
    if use_only_direct == True:
        columns = columns + 'Direct_annot'
        menr_df = menr_df[columns]
        menr_df.columns = flatten(['Region_set', score_keys, 'Direct_annot'])
        menr_df['TF'] = menr_df['Direct_annot']
        menr_df = menr_df.drop(['Direct_annot'])
    else:
        annot_columns = list(filter(lambda x:'annot' in x, menr_df.columns)) 
        columns = columns +  annot_columns
        menr_df = menr_df[columns]
        menr_df.columns = flatten(['Region_set', score_keys, annot_columns])
        for column in annot_columns:
            menr_df[column] = menr_df[column].str.split(', ')
            
        menr_df['TF'] = [menr_df[annot_columns[0]]+ menr_df[col] for col in annot_columns[1:]][0]
        menr_df = menr_df[flatten(['TF', 'Region_set', score_keys])]
        menr_df.columns = flatten(['TF', 'Group', score_keys])
        menr_df = menr_df.explode('TF')
        menr_df = menr_df.groupby(['TF', 'Group']).max().reset_index()
            
    ## Check for cistrome subsets   
    if subset_TFs is None:
        subset_TFs = list(set(menr_df['TF']))
    ## Check that cistromes are in the AUC matrix
    subset_TFs = set(subset_TFs).intersection(menr_df['TF'].tolist())
    subset_TFs = set(subset_TFs).intersection(dgem.index.tolist())
    ## Subset matrices
    ### By cistrome
    tf_expr = dgem.loc[subset_TFs,:]
    menr_df = menr_df.loc[menr_df['TF'].isin(subset_TFs),:]
    ### By cells
    if subset is not None:
        if group_variable is not None:
            subset_cells = cell_data[cell_data.isin(subset)].index.tolist()
            cell_data = cell_data.loc[subset_cells]
        if barcode_groups is not None:
            barcode_groups = {x: barcode_groups[x] for x in subset}
            subset_cells = list(set(sum([barcode_groups[x] for x in subset],[])))
        tf_expr = tf_expr.loc[:, tf_expr.columns.isin(subset_cells)]
    ### Take barcode groups per variable level
    if barcode_groups is not None:
        levels = sorted(list(barcode_groups.keys()))
        barcode_groups = [barcode_groups[group] for group in levels]
    if group_variable is not None:
        levels = sorted(list(set(cell_data.tolist())))
        barcode_groups = [cell_data[cell_data.isin([group])].index.tolist() for group in levels]
    ### Calculate mean expression
    tf_expr_mean = pd.concat([tf_expr.loc[:,tf_expr.columns.isin(barcodes)].mean(axis=1) for barcodes in barcode_groups], axis=1)
    tf_expr_mean.columns = levels
    ### Scale
    if standardize_expression: 
        tf_expr_mean = tf_expr_mean.T
        tf_expr_mean=(tf_expr_mean-tf_expr_mean.min()+0.00000001)/(tf_expr_mean.max()-tf_expr_mean.min()+0.00000001)
        tf_expr_mean = tf_expr_mean.T
    tf_expr_mean = tf_expr_mean.stack().reset_index()
    tf_expr_mean.columns = ['TF', 'Group', 'TF_expression']
    menr_df = menr_df[menr_df['Group'].isin(set(tf_expr_mean['Group']))]
    # Merge by column
    dotplot_df = pd.merge(tf_expr_mean, menr_df, how = 'outer', on = ['TF', 'Group'])
    dotplot_df = dotplot_df.replace(np.nan,0)
    return dotplot_df
    
def generate_dotplot_df_cistrome_AUC(scplus_obj: 'SCENICPLUS',
                       enrichment_key: str,
                       group_variable: str,
                       subset: list = None,
                       subset_cistromes: list = None,
                       use_pseudobulk: bool = False,
                       normalize_expression: bool = True,
                       standardize_expression: bool = False,
                       standardize_auc: bool = False):
    """
    Function to generate dotplot dataframe from cistrome AUC enrichment
    
    Parameters
    ---------
    scplus_obj: `class::SCENICPLUS`
        A :class:`SCENICPLUS` object with motif enrichment results from pycistarget (`scplus_obj.menr`).
    enrichment_key: str
        Key of the motif enrichment result to use.
    group_variable: str
        Group variable to use to calculate TF expression per group. Levels of this variable should match with the entries in the
        selected motif enrichment dictionary. 
    subset: list, optional
        Subset of classes to use. Default: None (use all)
    subset_cistromes: list, optional
        List of cistromes to use. Default: None (use all)
    use_pseudobulk: bool, optional
        Whether to use pseudobulk to calculate gene expression. Only available if there is pseudobulk profiles for group_variable.
        Default: False
    normalize_expression: bool, optional
        Whether to log(CPM) normalize gene expression. Default: True
    standardize_expression: bool, optional
        Wheter to standarize gene expression between o and 1. Default: False
    standardize_auc: bool, optional
        Wheter to standarize AUC values between o and 1. Default: False
        
    Returns
    ---------
        A data frame with enrichment values per TF and cell group to use as input for `dotplot()`.
    """
    # Get data
    ## If using pseudobulk
    if use_pseudobulk:
        dgem = scplus_obj.uns['Pseudobulk'][group_variable]['Expression'].copy()
        cistromes_auc = scplus_obj.uns['Pseudobulk'][group_variable]['Cistromes_AUC'][enrichment_key].copy()
        cell_data = pd.DataFrame([x.rsplit('_', 1)[0] for x in cistromes_auc.columns], 
                                 index=cistromes_auc.columns).iloc[:,0]
    ## If using all
    else:
        dgem = pd.DataFrame(scplus_obj.X_EXP, index=scplus_obj.cell_names, columns=scplus_obj.gene_names).copy().T
        cistromes_auc = scplus_obj.uns['Cistromes_AUC'][enrichment_key].copy().T
        cell_data = scplus_obj.metadata_cell.loc[cistromes_auc.columns, group_variable]
    ## Should gene expression be normalized?
    if normalize_expression:
        dgem = dgem.T / dgem.T.sum(0) * 10**6
        dgem = np.log1p(dgem).T
    ## Check for cistrome subsets   
    if subset_cistromes is None:
        subset_cistromes = scplus_obj.uns['Cistromes'][enrichment_key].keys()
    ## Check that cistromes are in the AUC matrix
    subset_cistromes = set(subset_cistromes).intersection(cistromes_auc.index.tolist())
    ## Take TF names
    subset_tfs = [re.sub('_(.*)', '',(re.sub('_extended', '', cistrome_name))) for cistrome_name in subset_cistromes]
    check_df = pd.DataFrame([subset_cistromes, subset_tfs]).T
    check_df.columns = ['Subset_cistromes', 'Subset_TFs']
    check_df = check_df[check_df['Subset_TFs'].isin(dgem.index.tolist())]
    subset_tfs = list(set(check_df['Subset_TFs'].tolist()))
    subset_cistromes = list(set(check_df['Subset_cistromes'].tolist()))
    ## Subset matrices
    ### By cistrome
    tf_expr = dgem.loc[subset_tfs,:]
    cistromes_auc_tf = cistromes_auc.loc[subset_cistromes,:]
    ### By cells
    if subset is not None:
        subset_cells = cell_data[cell_data.isin(subset)].index.tolist()
        cell_data = cell_data.loc[subset_cells]
        tf_expr = tf_expr.loc[:, subset_cells]
        cistromes_auc_tf = cistromes_auc_tf.loc[:,subset_cells]
    ### Take barcode groups per variable level
    levels = sorted(list(set(cell_data.tolist())))
    barcode_groups = [cell_data[cell_data.isin([group])].index.tolist() for group in levels]
    ### Calculate mean expression
    tf_expr_mean = pd.concat([tf_expr.loc[:,barcodes].mean(axis=1) for barcodes in barcode_groups], axis=1)
    tf_expr_mean.columns = levels
    ### Scale
    if standardize_expression: 
        tf_expr_mean = tf_expr_mean.T
        tf_expr_mean=(tf_expr_mean-tf_expr_mean.min()+0.00000001)/(tf_expr_mean.max()-tf_expr_mean.min())
        tf_expr_mean = tf_expr_mean.T
    tf_expr_mean = tf_expr_mean.stack().reset_index()
    tf_expr_mean.columns = ['TF', 'Group', 'TF_expression']
    ### Calculate mean AUC
    cistromes_auc_tf_mean = pd.concat([cistromes_auc_tf.loc[:,barcodes].mean(axis=1) for barcodes in barcode_groups], axis=1)
    cistromes_auc_tf_mean.columns = levels
    if standardize_auc:
        cistromes_auc_tf_mean = cistromes_auc_tf_mean.T
        cistromes_auc_tf_mean=(cistromes_auc_tf_mean-cistromes_auc_tf_mean.min()+0.00000001)/(cistromes_auc_tf_mean.max()-cistromes_auc_tf_mean.min())
        cistromes_auc_tf_mean = cistromes_auc_tf_mean.T
    cistromes_auc_tf_mean = cistromes_auc_tf_mean.stack().reset_index()
    cistromes_auc_tf_mean.columns = ['Cistrome', 'Group', 'Cistrome_AUC']
    cistromes_auc_tf_mean['TF'] = cistromes_auc_tf_mean['Cistrome'].replace('_.*', '', regex=True)
    # Merge by column
    dotplot_df = pd.merge(tf_expr_mean, cistromes_auc_tf_mean, how = 'outer', on = ['TF', 'Group'])
    return dotplot_df
    
def _cluster_labels_to_idx(labels):
    """
    A helper function to convert cluster labels to idx
    """
    counter = 0
    order = np.zeros(labels.shape, dtype = int)
    for i in range(0, labels.max() + 1):
            for j in np.where(labels == i)[0]:
                    order[j] = counter
                    counter += 1
    idx = [np.where(order == i)[0][0] for i, _ in enumerate(order)]
    return idx

def dotplot(df_dotplot: 'pd.DataFrame',
             ax: plt.axes = None,
             enrichment_variable: str = 'Cistrome_AUC',
             region_set_key: str = 'Cistrome',
             size_var: str = 'TF_expression',
             color_var: str = 'Cistrome_AUC',
             order_group: list = None,
             order_cistromes: list = None,
             order_cistromes_by_max: str = 'TF_expression',
             cluster: str = None, #can be group, TF or both
             n_clust: int = 2,
             min_point_size: float = 3,
             max_point_size: float = 30,
             cmap: str = 'viridis',
             vmin: float = 0,
             vmax: float = None,
             x_tick_rotation: float = 45,
             x_tick_ha: str = 'right', 
             fontsize: float = 9,
             z_score_expr: bool = True,
             z_score_enr: bool = True,
             grid_color = 'grey', 
             grid_lw = 0.5,
             highlight = None, 
             highlight_lw = 1,
             highlight_lc = 'black',
             figsize: Optional[Tuple[float, float]] = (10, 10),
             use_plotly = True,
             plotly_height = 1000,
             save = None):
    """
    Function to generate dotplot
    
    Parameters
    ---------
    df_dotplot: pd.DataFrame
        A pd.DataFrame with a column Group and varaibles to use for the dotplot
    ax: plt.axes
        An ax object, only require if use_plotly == False.
    enrichment_variable: str, optional
        Variable to use for motif/TF enrichment. Default: 'Cistrome_AUC;
    region_set_key: str, optional
        Name of the column where TF/cistromes are. Default: 'Cistrome'
    size_var: str, optional
        Variable in df_dotplot to code the dot size with. Default: 'Cistrome_AUC'
    color_var: str, optional
        Variable in df_dotplot to code the dot color with. Default: 'Cistrome_AUC'
    order_group: list, optional
        Order to plot the groups by. Default: None (given)
    order_cistromes: list, optional
        Order to plot the cistromes by. Default: None (given)
    order_cistromes_by_max: str, optional
        Variable to order the cistromes with by values in group, to form a diagonal. Default: None
    cluster: str, optional
        Whether to cluster cistromes (or TFs)/groups. It will be overwritten if order_cistromes/order_group/order_cistromes_by_max
        are specified. 
    n_clust: int, optional
        Number of clusters to form if cluster is specified. Default: 2
    min_point_size: int, optional
        Minimal point size. Only available with Cistrome AUC values.
    max_point_size: int, optional
        Maximum point size. Only available with Cistrome AUC values.
    cmap: str, optional
        Color map to use.
    vmin: int, optional
        Minimal color value.
    vmax: int, optional
        Maximum color value
    x_tick_rotation: int, optional
        Rotation of the x-axis label. Only if `use_plotly = False`
    fontsize: int, optional
        Labels fontsize
    z_score_expr: bool, optional
        Whether to z-normalize expression values
    z_score_enr: bool, optional
        Whether to z-normalize enrichment values
    grid_color: str, optional
        Color for grid. Only if `use_plotly = False`
    grid_lw: float, optional
        Line width of grid. Only if `use_plotly = False`
    highlight: list, optional
        Whether to highlight specific TFs/cistromes. Only if `use_plotly = False`
    highlight_lw: float, optional
        Line width of highlight. Only if `use_plotly = False`
    highlight_lc: str, optional
        Color of highlight. Only if `use_plotly = False`
    figsize: tuple, optional
        Figure size. Only if `use_plotly = False`
    use_plotly: bool, optional
        Whether to use plotly for plotting. Default: True
    plotly_height: int, optional
        Height of the plotly plot. Width will be adjusted accordingly
    save: str, optional
        Path to save dotplot as file
    """
    
    
    # Seperate dotsize data (motif_enrichment) and dot color data (mean expr)
    dotsizes = df_dotplot[['Group', region_set_key, enrichment_variable]].pivot_table(index = 'Group', columns = region_set_key).fillna(0)[enrichment_variable]
    dotcolors = df_dotplot[['Group', region_set_key, 'TF_expression']].pivot_table(index = 'Group', columns = region_set_key).fillna(0)['TF_expression']
    
    category_orders = {}
    
    # Cluster
    if cluster == 'both' or cluster == 'group':
        clustering_groups = AgglomerativeClustering(n_clusters = n_clust).fit(dotsizes)
        ordered_idx_grps = _cluster_labels_to_idx(clustering_groups.labels_)
        dotsizes = dotsizes.iloc[ordered_idx_grps, :]
        dotcolors = dotcolors.iloc[ordered_idx_grps, :]
        category_orders['Group'] = dotcolors.index.tolist()
    if cluster == 'both' or cluster == 'TF':
        clustering_TFs = AgglomerativeClustering(n_clusters = n_clust).fit(dotsizes.T)
        ordered_idx_TFs = _cluster_labels_to_idx(clustering_TFs.labels_)
        dotsizes = dotsizes.iloc[:, ordered_idx_TFs]
        dotcolors = dotcolors.iloc[:, ordered_idx_TFs]
        category_orders[region_set_key] = dotsizes.columns

    # If order TFs/cistromes by max         
    if order_cistromes_by_max == enrichment_variable:
        df = dotsizes.idxmax(axis=0)
        order_cistromes = pd.concat([df[df == x] for x in  dotsizes.index.tolist() if len(df[df == x]) > 0]).index.tolist()
        dotsizes = dotsizes.loc[:,order_cistromes[::-1]]
        dotcolors = dotcolors.loc[:, order_cistromes[::-1]]
        category_orders[region_set_key] = order_cistromes
        
    if order_cistromes_by_max == 'TF_expression':
        df = dotcolors.idxmax(axis=0)
        order_cistromes = pd.concat([df[df == x] for x in  dotcolors.index.tolist() if len(df[df == x]) > 0]).index.tolist()
        dotsizes = dotsizes.loc[:,order_cistromes[::-1]]
        dotcolors = dotcolors.loc[:, order_cistromes[::-1]]
        category_orders[region_set_key] = order_cistromes
    
    # Order by given order
    if order_group is not None:
        dotsizes = dotsizes.loc[order_group[::-1],:]
        dotcolors = dotcolors.loc[order_group[::-1],:]
        category_orders['Group'] = order_group
    if order_cistromes is not None:
        dotsizes = dotsizes.loc[:,order_cistromes[::-1]]
        dotcolors = dotcolors.loc[:, order_cistromes[::-1]]
        category_orders[region_set_key] = order_cistromes
    
    # Get dot sizes
    if color_var == enrichment_variable and size_var == 'TF_expression':
        x = dotcolors
        y = dotsizes
        dotcolors = y
        dotsizes = x
        
    # Scale dotsizes
    s = dotsizes.to_numpy().flatten('F')
    s_min = (s[s != 0]).min()
    s_max = s.max()
        
    if not use_plotly:
        if save is not None:
            pdf = matplotlib.backends.backend_pdf.PdfPages(save)
        fig = plt.figure(figsize=figsize)
        # Calculate Z-score for expression values
        if z_score_expr:
            u = dotcolors.mean()
            sd = dotcolors.std()
            dotcolors = (dotcolors - u) / sd

        if vmin is None:
            vmin = dotcolors.min().min()

        if z_score_enr:
            u = dotsizes.mean()
            sd = dotsizes.std()
            dotsizes = (dotsizes - u) / sd

        # Generate plotting data
        n_group = len(set(df_dotplot['Group']))
        n_TF = len(set(df_dotplot[region_set_key]))

        # Generate a grid
        x = np.tile( np.arange(n_group), n_TF)
        y = [int(i / n_group) for i, _ in enumerate(x)]

        # Scale values between min_point_size and max_point_size keep zero values at 0
        s[s != 0] = min_point_size + (s[s != 0]- s_min) * ((max_point_size - min_point_size) / (s_max - s_min)) 

        # Get dot colors
        c = dotcolors.to_numpy().flatten('F')

        # Get edges for eRegulons
        if highlight is not None:
            TFs = dotsizes.columns
            groups = dotsizes.index
            linewidths = [highlight_lw if TF in highlight else 0 for 
                            TF in np.repeat(TFs, n_group)]
        else:
            linewidths = None

        if vmax is None:
            vmax = c.max()

        norm = Normalize(vmin = vmin, vmax = vmax)

        ax.set_axisbelow(True)
        ax.grid(color = grid_color, linewidth = grid_lw)
        scat = ax.scatter(x, y, s = s, c = c, cmap = cmap, norm = norm, edgecolors = highlight_lc, linewidths = linewidths)
        # set x ticks
        ax.set_xticks( np.arange(n_group) )
        ax.set_xticklabels(dotsizes.index, rotation = x_tick_rotation, ha = x_tick_ha, fontdict = {'fontsize' : fontsize} )
        # set y ticks
        ax.set_yticks( np.arange(n_TF) )
        ax.set_yticklabels(dotsizes.columns, fontdict = {'fontsize': fontsize})

        # Draw colorbar
        cbar = plt.colorbar(mappable = ScalarMappable(norm = norm, cmap = cmap), 
                     ax = ax, location = 'bottom', orientation = 'horizontal', 
                     aspect = 10, shrink = 0.4, pad = 0.10, anchor = (0, 0))
        cbar_label = color_var
        cbar.set_label(cbar_label)

        L = plt.legend(*scat.legend_elements("sizes", num=3), 
                   loc = 'lower right', bbox_to_anchor = (0.8, -0.2), frameon = False, title = size_var, ncol = 1, mode = None)
        # Recalculate original scale
        to_int = lambda x: int(''.join(i for i in x if i.isdigit()))
        labels = np.array([to_int(t.get_text()) for t in L.get_texts()])
        re_scale = lambda x: (x - min_point_size) * ((s_max - s_min) / (max_point_size - min_point_size)) + s_min
        rescaled = re_scale(labels).round(2)
        for new, text in zip(rescaled, L.get_texts()):
            text.set_text(new)
            
        if save is not None:
            fig.save(save, bbox_inches='tight')
        plt.show()
    
    else:
        import plotly
        import plotly.express as px
        df = dotplot_df.copy()
        if min_point_size != 0 and enrichment_variable == 'Cistrome_AUC':
            df[enrichment_variable][df[enrichment_variable] != 0] = min_point_size + (df[enrichment_variable][df[enrichment_variable] != 0]- s_min) * ((max_point_size - min_point_size) / (s_max - s_min))
        if color_var == 'TF_expression' and size_var == enrichment_variable:
            fig = px.scatter(df, y=region_set_key, x="Group", color="TF_expression", size=enrichment_variable,
                           size_max=max_point_size, category_orders = category_orders, color_continuous_scale=cmap)
        else:
            fig = px.scatter(df, y=region_set_key, x="Group", color= enrichment_variable, size="TF_expression",
                           size_max=max_point_size, category_orders = category_orders, color_continuous_scale=cmap)

        fig.update_layout(
            height=plotly_height,  # Added parameter
            legend= {'itemsizing': 'trace'},
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis = dict(tickfont = dict(size=fontsize))
        )
        if save is not None:
            fig.write_image(save)
        fig.show()