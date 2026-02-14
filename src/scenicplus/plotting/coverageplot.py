"""Plot chromatin accessibility profiles and region to gene arcs.

"""

from typing import Mapping
import pyBigWig
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pyranges as pr
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib
from typing import Union, List



def _region_to_chrom_start_end(x): return [x.replace(':', '-').split('-')[0],
                                          int(x.replace(
                                              ':', '-').split('-')[1]),
                                          int(x.replace(':', '-').split('-')[2])]


def coverage_plot(SCENICPLUS_obj,
                  bw_dict: Mapping[str, str],
                  region: str,
                  genes_violin_plot: Union[str, List] = None,
                  genes_arcs: Union[str, List] = None,
                  gene_height: int = 1,
                  exon_height: int = 4,
                  meta_data_key: str = None,
                  pr_consensus_bed: pr.PyRanges = None,
                  region_bed_height: int = 1,
                  pr_gtf: pr.PyRanges = None,
                  pr_interact: pr.PyRanges = None,
                  bw_ymax: float = None,
                  color_dict: Mapping[str, str] = None,
                  cmap='tab20',
                  plot_order: list = None,
                  figsize: tuple = (6, 8),
                  fontsize_dict={'bigwig_label': 9,
                                 'gene_label': 9,
                                 'violinplots_xlabel': 9,
                                 'violinplots_ylabel': 9,
                                 'title': 12,
                                 'bigwig_tick_label': 5},
                  gene_label_offset=3,
                  arc_rad=0.5,
                  arc_lw=1,
                  cmap_violinplots='Greys',
                  violinplots_means_color='black',
                  violinplots_edge_color='black',
                  violoinplots_alpha=1,
                  width_ratios_dict={'bigwig': 3,
                                     'violinplots': 1},
                  height_ratios_dict={'bigwig_violin': 1,
                                      'genes': 0.5,
                                      'arcs': 5,
                                      'custom_ax': 2},
                  sort_vln_plots: bool = False,
                  add_custom_ax: int = None) -> plt.Figure:
    """
    Inspired by: https://satijalab.org/signac/reference/coverageplot

    Generates a figure showing chromatin accessibility coverage tracks, gene expression violin plots and region to gene interactions.

    Parameters
    ----------
    SCENICPLUS_obj
        An instance of class `~scenicplus.scenicplus_class.SCENICPLUS`.
    bw_dict
        A dict containing celltype annotations/cell groups as keys and paths to bigwig files as values.
    region
        A string specifying the region of interest, in chr:start-end format.
    genes_violin_plot
        A list or string specifying for which gene(s) to plot gene expression violin plots.
        default: None
    genes_arcs
        A list or string specifying for which gene(s) to plot arcs
        default: None (i.e. all genes in window)
    gene_height
        An int specifying the size of non-exon parts of a gene (shown underneath the coverage plot).
        default: 1
    exon_height
        An int specifying the size of nexon parts of a gene (shown underneath the coverage plot).
        default: 2
    meta_data_key
        A key specifying were to find annotations corresponding to the keys in `bw_dict` in `SCENICPLUS_obj.metadata_cell`
        default: None
    pr_consensus_bed
        An instance of class `pyranges.PyRanges` containing consensus peaks. Names of these peaks should correspond to annotations.
        default: None
    region_bed_height
        An int specifying with which height to draw lines corresponding to regions in `pr_consensus_bed`
        default: 1
    pr_gtf
        An instance of class `pyranges.PyRanges` containing gtf-formatted information about gene structure.
        default: None
    pr_interact
        An instance of class `pyranges.PyRanges` containing region to gene link information in ucsc interact format (use `scenicplus.utils.get_interaction_pr` to create such an object).
        default: None
    bw_ymax
        A float specifying the maximum height at which to draw coverage plots.
        default: None
    color_dict
        A dict specifying colors for each key in `bw_dict`.
        default: None
    cmap
        A string specifying a `matplotlib.cm` colormap to use for coloring coverage plots.
        default: 'tab20'
    plot_order
        A list specifying the order to use for plotting coverage plots and violin plots.
        default: None
    figsize
        A tuple specifying the fig size/
        default: (6, 8)
    fontsize_dict
        A dictionary specifying the fontsize of various labels in the plot.
        default: {'bigwig_label': 9, 'gene_label': 9, 'violinplots_xlabel': 9, 'title': 12, 'bigwig_tick_label': 5}
    gene_label_offset
        The y-offset of the label underneath each gene.
        default: 3
    arc_rad
        The amount of radians the region-to-gene arc should be bend.
        default: 0.5
    arc_lw
        The linewidth of the region-to-gene arcs.
        default: 1
    cmap_violinplots
        A string specifying a `matplotlib.cm` colormap to use for coloring violinplots. 
        default: 'Greys'
    violinplots_means_color
        A string specifying which color to use to indicate the mean value in violinplots.
        default: 'black'
    violinplots_edge_color
        A string specifying which color to use for the edge of the violinplots.
        default: 'black'
    violoinplots_alpha
        The alpha value of the violinplots facecolor.
        default: 1
    width_ratios_dict
        A dict specifying the ratio in vertical direction each part of the plot should use.
        default: {'bigwig': 3, 'violinplots': 1}
    height_ratios_dict
        A dict specifying the ratio in horizontal direction each part of the plot should use.
        default: {'bigwig_violin': 1, 'genes': 0.5, 'arcs': 5}

    Returns:
    --------
    plt.Figure
    """
    fig_nrows = len(bw_dict.keys())
    if pr_interact is not None:
        fig_nrows += 1
    if pr_gtf is not None:
        fig_nrows += 1
    if add_custom_ax:
        fig_nrows += add_custom_ax
    height_ratios = [height_ratios_dict['bigwig_violin']
                     for i in range(len(bw_dict.keys()))]
    if pr_gtf is not None:
        height_ratios += [height_ratios_dict['genes']]
    if pr_interact is not None:
        height_ratios += [height_ratios_dict['arcs']]
    if add_custom_ax is not None:
        height_ratios += [height_ratios_dict['custom_ax']
                          for i in range(add_custom_ax)]

    if genes_violin_plot is not None:
        ncols = 2
        width_ratios = [width_ratios_dict['bigwig'],
                        width_ratios_dict['violinplots']]
    else:
        ncols = 1
        width_ratios = [1]
    fig, axs = plt.subplots(nrows=fig_nrows,
                            ncols=ncols,
                            gridspec_kw={
                                'height_ratios': height_ratios, 'width_ratios': width_ratios},
                            figsize=figsize)

    if genes_violin_plot is not None:
        axs_vln = axs[:, 1]
        axs_bw = axs[:, 0]
    else:
        axs_bw = axs

    subplot_idx = 0
    # get coordinates from region string
    chrom, start, end = _region_to_chrom_start_end(region)
    pr_region = pr.PyRanges(chromosomes=[chrom], starts=[start], ends=[end])
    # calculate bw_ymax for scaling
    bw_ymax_dict = {}
    for key in bw_dict.keys():
        bw_file = bw_dict[key]
        # calculate max value of the bigwig within our region
        bw = pyBigWig.open(bw_file)
        y = bw.values(chrom, start, end)
        y = np.nan_to_num(y)
        bw_ymax_dict[key] = y.max()

    x = np.array(range(start, end, 1))

    # set y_max
    if bw_ymax is None:
        bw_ymax = max(bw_ymax_dict.values())

    # set colors if not provided
    if color_dict is None:
        cmap = cm.get_cmap(cmap)
        color_dict = {k: cmap(i) for i, k in enumerate(bw_dict.keys())}

    # iterate over all bigwigs
    if plot_order is None:
        plot_order = bw_dict.keys()
    for key in plot_order:
        # create a new subplot
        ax = axs_bw[subplot_idx]
        subplot_idx += 1

        # open the bigwig
        bw_file = bw_dict[key]

        bw = pyBigWig.open(bw_file)
        y = bw.values(chrom, start, end)
        y = np.nan_to_num(y)

        # now plot the bigwig in the gridspec
        ax.fill_between(x, y1=y, y2=0, step="mid",
                        linewidth=0, color=color_dict[key])
        ax.patch.set_alpha(0)

        # figure settings
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim(
            [-2, bw_ymax]) if pr_consensus_bed is not None else ax.set_ylim([0, bw_ymax])
        ax.set_xticks([])
        ax.tick_params(axis='both', which='major',
                       labelsize=fontsize_dict['bigwig_tick_label'])

        # remove negative yticks
        y_ticks = ax.get_yticks()
        #y_tick_labels = ax.get_yticklabels()
        y_ticks_to_keep = np.where(y_ticks >= 0)[0]
        y_ticks = y_ticks[y_ticks_to_keep]
        #y_tick_labels = [y_tick_labels[i] for i in y_ticks_to_keep]
        ax.set_yticks(y_ticks)
        # ax.set_yticklabels(list(y_tick_labels))

        ax.text(x=x.min(), y=bw_ymax + 1, s=key,
                fontsize=fontsize_dict['bigwig_label'])

        sns.despine(top=True, right=True, left=True, bottom=True, ax=ax)

        if pr_consensus_bed is not None:
            consensus_bed_region_intersect = pr_consensus_bed.intersect(
                pr_region)
            for _, r in consensus_bed_region_intersect.df.iterrows():
                region_names = [x.split('_peak')[0]
                                for x in r['Name'].split(',')]
                if key in region_names:
                    bed_chrom = r['Chromosome']
                    bed_start = r['Start']
                    bed_end = r['End']
                    if pr_interact is not None:
                        pr_tmp = pr.PyRanges(chromosomes=[bed_chrom], starts=[
                                             bed_start], ends=[bed_end])
                        if len(pr_interact.intersect(pr_tmp)) > 0:
                            color = 'black'
                        else:
                            color = 'grey'
                    else:
                        color = 'grey'
                    rect = mpatches.Rectangle(
                        (bed_start, -2), bed_end-bed_start, region_bed_height, fill=True, color=color, linewidth=0)
                    ax.add_patch(rect)

    # draw the genes of interest, from our gtf
    # intersect genes gtf with the region of interest
    if pr_gtf is not None:
        # intersect gtf with region and get first 9 columns
        gtf_region_intersect = pr_gtf.intersect(pr_region)
        # only keep exon and gene info
        gtf_region_intersect = gtf_region_intersect[np.logical_and(
            np.logical_or(gtf_region_intersect.Feature == 'gene',
                          gtf_region_intersect.Feature == 'exon'),
            gtf_region_intersect.gene_type == 'protein_coding')]
        # iterate over all genes in intersect
        ax = axs_bw[subplot_idx]
        subplot_idx += 1
        genes_in_window = set(gtf_region_intersect.gene_name)
        n_genes_in_window = len(genes_in_window)
        for idx, _gene in enumerate(genes_in_window):
            _gene_height = gene_height / n_genes_in_window
            _gene_bottom = -gene_height/2 + _gene_height * idx
            _exon_bottom = _gene_bottom - \
                (((exon_height / n_genes_in_window) / 2) - _gene_height)
            _exon_height = (((exon_height / n_genes_in_window) / 2) - _gene_height) + \
                _gene_height + \
                (((exon_height / n_genes_in_window) / 2) - _gene_height)
            # don't plot non-protein coding transcripts (e.g. nonsense_mediated_decay)
            if not all(gtf_region_intersect.df.loc[gtf_region_intersect.df['gene_name'] == _gene, 'transcript_type'].dropna() == 'protein_coding'):
                continue
            # plot the gene parts (gene body and gene exon)
            # iterate over all parts to plot them
            for _, part in gtf_region_intersect.df.loc[gtf_region_intersect.df['gene_name'] == _gene].iterrows():
                # make exons thick
                if part['Feature'] == 'exon':
                    exon_start = part['Start']
                    exon_end = part['End']
                    # draw rectangle for exon
                    rect = mpatches.Rectangle(
                        (exon_start, _exon_bottom), exon_end-exon_start, _exon_height, fill=True, color="k", linewidth=0)
                    ax.add_patch(rect)
                # make the gene body a thin line, drawn at the end so it will always display on top
                elif part['Feature'] == 'gene':
                    gene_start = part['Start']
                    gene_end = part['End']
                    rect = mpatches.Rectangle(
                        (gene_start, _gene_bottom), gene_end-gene_start, _gene_height, fill=True, color="k", linewidth=0)
                    ax.add_patch(rect)
                ax.text(gene_start, _gene_bottom - gene_label_offset,
                        _gene, fontsize=fontsize_dict['gene_label'])
            # figure settings
            ax.set_ylim([-exon_height/2, exon_height/2])
            #ax.set_xlabel(gene, fontsize=10)
            ax.set_xlim([x.min(), x.max()])
            sns.despine(top=True, right=True, left=True, bottom=True, ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.patch.set_alpha(0)  # make sure that each individual subplot is transparent! otherwise the underlying plots won't be shown. this is important e.g. for the dar/peak visualisaton, since the DARs are drawn directly on top of the peaks. if the DAR plot is not transparent, no peaks will be visible!!

    # Draw arcs from region to gene
    if pr_interact is not None and pr_gtf is not None:
        ax = axs_bw[subplot_idx]
        subplot_idx += 1
        # intersect region with pr_interact
        pr_region = pr.PyRanges(
            chromosomes=[chrom], starts=[start], ends=[end])
        region_interact_intersect = pr_interact.intersect(pr_region)
        # only keep interactions to genes within window
        if genes_arcs is not None:
            if type(genes_arcs) == str:
                genes_arcs = [genes_arcs]
            region_interact_intersect = region_interact_intersect[np.isin(
                region_interact_intersect.targetName, list(set(genes_in_window) & set(genes_arcs)))]
        else:
            region_interact_intersect = region_interact_intersect[np.isin(
                region_interact_intersect.targetName, list(genes_in_window))]
        for _, r2g in region_interact_intersect.df.sort_values('value').iterrows():
            posA = (int(r2g['sourceStart']), 0)
            posB = (int(r2g['targetStart']), 0)
            # this to ensure arcs are always down
            sign = '-' if posA[0] > posB[0] else ''
            color = r2g['color']
            if (posA[0] > x.min() and posA[0] < x.max()) and (posB[0] > x.min() and posB[0] < x.max()):
                arrow = mpatches.FancyArrowPatch(posA=posA,
                                                 posB=posB,
                                                 connectionstyle=f"arc3,rad={sign}{arc_rad}",
                                                 color=color,
                                                 lw=arc_lw)
                ax.add_patch(arrow)
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim(-1, 0)
        sns.despine(top=True, right=True, left=True, bottom=True, ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])

    if genes_violin_plot is not None:
        # plot expression of gene/genes

        if meta_data_key not in SCENICPLUS_obj.metadata_cell.columns:
            raise ValueError(
                f'key {meta_data_key} not found in SCENICPLUS_obj.metadata_cell.columns')

        # check that bw_dict keys is a subset of scplus_obj.metadata_cell[meta_data_key]
        annotations = SCENICPLUS_obj.metadata_cell[meta_data_key].to_numpy()
        if len(set(plot_order) - set(annotations)) != 0:
            not_found = set(plot_order) - set(annotations)
            raise ValueError(
                f'Following keys were not found in SCENICPLUS_obj.metadata_cell[{meta_data_key}]\n{", ".join(not_found)}')

        # normalize scores
        mtx = np.log1p((SCENICPLUS_obj.X_EXP.T /
                       SCENICPLUS_obj.X_EXP.sum(1) * (10 ** 4)).T)

        # get expression values for gene/genes
        idx_gene = list(SCENICPLUS_obj.gene_names).index(genes_violin_plot) if type(
            genes_violin_plot) == str else [list(SCENICPLUS_obj.gene_names).index(g) for g in genes_violin_plot]
        expr_vals = mtx[:, idx_gene]

        norm = matplotlib.colors.Normalize(vmin=0, vmax=len(plot_order))
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap_violinplots)
        expr_min = np.Inf
        expr_max = np.NINF
        for idx, annotation in enumerate(plot_order):
            ax = axs_vln[idx]
            cells = np.where(annotations == annotation)
            expr_vals_sub = expr_vals[cells]

            if sort_vln_plots:
                idx_sorted = np.argsort(expr_vals_sub.mean(0))[::-1]
                expr_vals_sub = expr_vals_sub[:, idx_sorted]
                genes_violin_plot = np.array(
                    list(genes_violin_plot))[idx_sorted]

            expr_min = min(expr_min, expr_vals_sub.min())
            expr_max = max(expr_max, expr_vals_sub.max())

            vln_plot_part = ax.violinplot(
                expr_vals_sub, vert=False, showmeans=True, showextrema=False)
            for i, pc in enumerate(vln_plot_part['bodies']):
                facecolor = mapper.to_rgba(i)
                pc.set_facecolor(color_dict[annotation])
                pc.set_edgecolor(violinplots_edge_color)
                pc.set_alpha(violoinplots_alpha)
            vln_plot_part['cmeans'].set_edgecolor(violinplots_means_color)
            n_labels = 1 if type(genes_violin_plot) == str else len(
                genes_violin_plot)
            ax.set_yticks(ticks=np.arange(1, n_labels + 1))
            ax.set_yticklabels(genes_violin_plot,
                               fontsize=fontsize_dict['violinplots_ylabel'])
            ax.set_xlim(xmin=round(expr_min), xmax=round(expr_max + 0.5))
            if idx < len(plot_order) - 1:
                sns.despine(top=True, right=True,
                            left=True, bottom=True, ax=ax)
                ax.set_xticks([])
            else:
                sns.despine(top=True, right=True,
                            left=True, bottom=False, ax=ax)
                ax.set_xlabel('log-normalised\nExpression counts',
                              fontsize=fontsize_dict['violinplots_xlabel'])
        for i in range(idx + 1, len(axs_vln)):
            ax = axs_vln[i]
            sns.despine(top=True, right=True, left=True, bottom=True, ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])

    # finally, add a little text that shows which regions you're plotting
    length = round((end-start)/1000)
    label = f'{region} ({length} kb)'

    fig.suptitle(label, fontsize=fontsize_dict['title'])

    if add_custom_ax:
        return axs[axs.shape[0] - add_custom_ax: axs.shape[0]], fig
    else:
        return fig
