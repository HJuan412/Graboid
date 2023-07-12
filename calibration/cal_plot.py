#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:42:33 2023

@author: nano

Plot the calibration results
"""

#%% libraries
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns

#%% functions
# seaborn modified
def despine(ax):
    sides = 'top bottom right left'.split()
    for side in sides:
        ax.spines[side].set_visible(False)

def relative_luminance(color):
    """Calculate the relative luminance of a color according to W3C standards
    Parameters
    ----------
    color : matplotlib color or sequence of matplotlib colors
        Hex code, rgb-tuple, or html color name.
    Returns
    -------
    luminance : float(s) between 0 and 1
    """
    rgb = mpl.colors.colorConverter.to_rgba_array(color)[:, :3]
    rgb = np.where(rgb <= .03928, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    lum = rgb.dot([.2126, .7152, .0722])
    try:
        return lum.item()
    except ValueError:
        return lum
    
def annotate_heatmap(annot, X, Y, ax, mesh, annot_size):
    """Add textual labels with the value in each cell."""
    mesh.update_scalarmappable()
    height, width = annot.shape
    
    xpos, ypos = np.meshgrid(X, Y)
    for x, y, m, color, val in zip(xpos.flat, ypos.flat,
                                   mesh.get_array(), mesh.get_facecolors(),
                                   annot.flat):
        lum = relative_luminance(color)
        text_color = ".15" if lum > .408 else "w"
        text_kwargs = dict(color=text_color, ha="center", va="center")
        text_kwargs.update({"size": annot_size, 'ha':'center', 'va':'center'})
        ax.text(x, y, val, **text_kwargs)

def custom_heatmap(data, annot, win_poss, cell_size, ax, cax, cmap, annot_size):
    # use this to build a customized heatmap when window sizes are variable
    # get mesh internal dimensions
    C = data.to_numpy()
    # X and Y should have shapes (16, 4)
    X = np.zeros((C.shape[0] + 1, C.shape[1] + 1))
    Y = np.zeros((C.shape[0] + 1, C.shape[1] + 1))
    for iidx, i in enumerate(np.arange(len(data) + 1) * cell_size):
        X[-iidx-1] = i
    for jidx, j in enumerate(win_poss * cell_size):
        Y[:,jidx] = j
    
    # generate heatmap
    mesh = ax.pcolormesh(Y, X, C, shading='flat', cmap=cmap) # pcolormesh lets us vary the cell widths
    # get cell centres
    x_centre = X[1:,0] + cell_size/2
    y_centre = Y[0,:-1] + np.diff(Y[0]) / 2
    annotate_heatmap(annot, y_centre, x_centre, ax, mesh, annot_size)
    despine(ax) # remove heatmap borders
    # TODO: fit heatmap to it's axis boundaries
    # generate colorbar
    cb = plt.colorbar(cm.ScalarMappable(cmap=cmap), cax=cax, orientation='horizontal')
    cb.outline.set_linewidth(0) # remove colorbar borders
    return x_centre, y_centre

# heatmap construction
def build_heatmap(data,
                  param_annots,
                  windows,
                  title,
                  metric,
                  cell_size=45,
                  dpi=300,
                  height_pad=5,
                  ratio_pad=10,
                  hspace=0.01,
                  max_cols=10,
                  custom=False):
    # set color map
    viridis = cm.get_cmap('viridis', 256)
    grey = np.array([128/256, 128/256, 128/256, 1])
    viridis.set_bad(grey)
    
    # get column widths
    windows = windows[data.columns]
    win_widths = windows[:,1] - windows[:,0]
    win_ratios = win_widths / win_widths.min()
    win_poss = np.concatenate([[0], np.cumsum(win_ratios)])
    if win_poss.sum() > max_cols:
        # ensure total width doesn't exceed that of max_cols
        sum_to_max = win_poss.sum() / max_cols
        win_poss /= sum_to_max
    column_labels = [f'{w0} - {w1}' for w0, w1 in windows]
    
    # calculate heatmap and figure dimensions
    map_width = data.shape[1] * cell_size
    if custom:
        # adjust map width if using custom windows
        map_width = win_ratios.sum() * cell_size
    map_height = (data.shape[0] + height_pad) * cell_size # height pad is used to compensate for the space taken by the title and window labels
    fig_width = map_width / dpi
    fig_height = map_height / dpi
    
    annot_size = 2
    ytick_size = 4
    xtick_size = 3
    hm_lab_size = 5
        
    # initialize figure
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios = [data.shape[0] + ratio_pad, 1], hspace = hspace) # use this to get a fine control over the colorbar position and size
    # top ax contains the heatmap, bottom ax contains the colorbar
    ax_hm = fig.add_subplot(gs[0])
    ax_cb = fig.add_subplot(gs[1])
    
    # build heatmap
    if custom:
        yticks, xticks = custom_heatmap(data,
                                        param_annots,
                                        win_poss,
                                        cell_size,
                                        ax_hm,
                                        ax_cb,
                                        viridis,
                                        annot_size)
    else:
        sns.heatmap(data,
                    square = True,
                    cmap = viridis,
                    annot = param_annots,
                    fmt='',
                    annot_kws={"size": annot_size, 'ha':'center', 'va':'center'},
                    ax = ax_hm,
                    cbar_kws = {'orientation':'horizontal'},
                    cbar_ax = ax_cb,
                    vmin=0,
                    vmax=1)
        yticks = np.arange(data.shape[0]) + 0.5
        xticks = np.arange(data.shape[1]) + 0.5
    
    # place labels for all windows and taxa
    ax_hm.set_yticks(yticks)
    ax_hm.set_yticklabels(data.index, size = ytick_size, rotation = 30, va='top')
    ax_hm.set_xticks(xticks)
    ax_hm.set_xticklabels(column_labels, size = xtick_size, rotation=60, ha='left')
    # relocate window labels to the top of the plot
    ax_hm.xaxis.tick_top()
    ax_hm.xaxis.set_label_position('top')
    # name axes
    ax_hm.set_ylabel('Taxa', size=hm_lab_size)
    ax_hm.set_xlabel('Windows', size=hm_lab_size)
    
    # adjust colorbar labels
    ax_cb.tick_params(labelsize = 4)
    ax_cb.set_xlabel(metric, size = 4)
    ax_hm.set_title(title, size=5)
    
    return fig

def extract_data(report, guide, ranks, windows, collapse=True):
    # Add lineage codes to the report
    report['LinCode'] = guide.loc[report.taxID, 'LinCode'].values
    report['lin_name'] = report.LinCode + ' ' + report.taxon
    report.sort_values(['window', 'LinCode'], inplace=True)
    
    # generate data tables for each rank
    for rk in ranks:
        rk_report = report.loc[report['rank'] == rk]
        lincodes = pd.unique(rk_report.lin_name)
        
        # initialize table of rows: taxa in rank, columns: window indexes
        # populate each cell with the index of the report row with results for
        # each taxon/window
        rk_tab = pd.DataFrame(-1, index=lincodes, columns=windows, dtype=int)
        for idx, row in rk_report.iterrows():
            rk_tab.loc[row.lin_name, row.window] = idx
        
        rk_tab.sort_index(inplace=True)
        
        # data contains the scroe for every taxon/window
        # annot contains the winning parameter combination for every taxon/window
        locs = (rk_tab >= 0).to_numpy()
        data = np.full(rk_tab.shape, np.nan)
        annot = np.full(rk_tab.shape, np.nan, dtype=object)
        
        # populate data and annot
        for col_idx, col_locs in enumerate(locs.T):
            indexes = rk_tab.to_numpy()[col_locs, col_idx]
            col_scores_annots = rk_report.loc[indexes, ['score', 'annotation']]
            data[col_locs, col_idx] = col_scores_annots.score.values
            annot[col_locs, col_idx] = col_scores_annots.annotation.values
        
        data = pd.DataFrame(data, index=lincodes, columns=rk_tab.columns)
        annot = pd.DataFrame(annot, index=lincodes, columns=rk_tab.columns)
        
        # remove rows and columns with only empty and 0 score values
        discarded_taxa = []
        if collapse:
            nonnull_locs = (rk_tab >= 0).values
            ne_cols = nonnull_locs.sum(0) > 0
            ne_rows = nonnull_locs.sum(1) > 0
            
            nonzero_locs = (data > 0).values
            nz_cols = nonzero_locs.sum(0) > 0
            nz_rows = nonzero_locs.sum(1) > 0
            
            filtered_cols = ne_cols & nz_cols
            filtered_rows = ne_rows & nz_rows
            
            discarded_taxa = sorted([re.sub('^[^ ]+ ', '\t', tax) for tax in rk_tab.index.values[~filtered_rows]]) # regular expression removes the LinCode
            data = data.loc[filtered_rows, filtered_cols]
            annot = annot.loc[filtered_rows, filtered_cols]
        yield rk, data, annot, discarded_taxa
        
def plot_results(report, guide, ranks, windows, metric, prefix, collapse=True, custom=False):
    # collapse eliminates taxa with no values over 0 to reduce heatmap size, if set to True, a file of rejected taxa is generated
    
    discarded_dict = {}
    for rank, data, annot, discarded in extract_data(report, guide, ranks, np.arange(windows.shape[0]), collapse):
        title = f'{metric} scores for rank: {rank}\nN\nK\nmethod (u = unweighted, w = wKNN, d = dwKNN)'
        
        # heatmaps have at most 1000 rows, if a rank (usually only species, MAYBE genus) exceeds the limit, split it
        if len(data) <= 1000:
            fig = build_heatmap(data, annot, windows, title, metric, custom=custom)
            fig.savefig(prefix + f'/{metric}_{rank}.png', format='png', bbox_inches='tight')
            plt.close()
        else:
            for n_subplot, subplot_idx in enumerate(np.arange(0, len(data), 1000)):
                fig = build_heatmap(data.iloc[subplot_idx: subplot_idx + 1000], annot.iloc[subplot_idx: subplot_idx + 1000], windows, title, metric, custom=custom)
                fig.savefig(prefix + f'/{metric}_{rank}.{n_subplot + 1}.png', format='png', bbox_inches='tight')
                plt.close()
        
        if len(discarded) > 0:
            discarded_dict[rank] = discarded
            
    if len(discarded_dict) > 0:
        with open(prefix + f'/{metric}_rejected.txt', 'w') as handle:
            handle.write(f'The following taxa yielded no {metric} scores over 0 and were ommited from the plots:\n')
            for rk, dcarded in discarded_dict.items():
                handle.write(f'{rk} ({len(dcarded)}):\n' + '\n'.join(dcarded) + '\n')