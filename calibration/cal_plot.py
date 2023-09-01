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

# TODO: this function does something weird with the colors of the phylum heatmaps
def custom_heatmap(data, annot, win_poss, cell_size, ax, cax, cmap, annot_size):
    """use this to build a customized heatmap when window sizes are variable"""
    # data: 2d-array of shape (# taxa, # windows) containing metric scores
    # annots: 2d-array of shape (# taxa, # windows) containing parameter annotations
    
    # get mesh internal dimensions
    # X and Y should have shapes (16, 4)
    X = np.zeros((data.shape[0] + 1, data.shape[1] + 1))
    Y = np.zeros((data.shape[0] + 1, data.shape[1] + 1))
    for iidx, i in enumerate(np.arange(data.shape[0] + 1) * cell_size):
        X[-iidx-1] = i
    for jidx, j in enumerate(win_poss * cell_size):
        Y[:,jidx] = j
    
    # generate heatmap
    mesh = ax.pcolormesh(Y, X, data, shading='flat', cmap=cmap) # pcolormesh lets us vary the cell widths
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
                  annots,
                  indexes,
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
    # data: 2d-array of shape (# taxa, # windows) containing metric scores
    # annots: 2d-array of shape (# taxa, # windows) containing parameter annotations
    # indexes: 1d-array, contains tax names and codes
    # windows: 2d-array of shape (# windows, 2) containing window coordinates
    # title: pre-generated figure title
    # metric: metric name
    # cell_size: pixels per cell
    # dpi: resolution (dots per inch)
    # height_pad: used to compensate for the space taken by the title and window labels
    # ratio_pad: related to colorbar position
    # hspace: related to colorbar position
    # max_cols: set the maximum number of columns in custom heatmaps
    # custom: adjust colum width to relative window widths
    
    # set color map
    viridis = cm.get_cmap('viridis', 256)
    grey = np.array([128/256, 128/256, 128/256, 1])
    viridis.set_bad(grey)
    
    # get column relative widths
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
                                        annots,
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
                    annot = annots,
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
    ax_hm.set_yticklabels(indexes, size = ytick_size, rotation = 30, va='top')
    ax_hm.set_xticks(xticks)
    ax_hm.set_xticklabels(column_labels, size = xtick_size, rotation=60, ha='left')
    # relocate window labels to the top of the plot
    ax_hm.xaxis.tick_top()
    ax_hm.xaxis.set_label_position('top')
    # name axes
    ax_hm.set_ylabel('Taxa', size=hm_lab_size)
    ax_hm.set_xlabel('Windows (start - end)', size=hm_lab_size)
    
    # adjust colorbar labels
    ax_cb.tick_params(labelsize = 4)
    ax_cb.set_xlabel(metric + '\nu = unweighted, w = wKNN, d = dwKNN', size = 4)
    ax_hm.set_title(title, size=5)
    
    return fig

def get_APRF_heatmap_arrays(report):
    """Prepare heatmap arrays for the given APRF report"""
    # returns:
        # scores, containing the best score for each taxon per window (missing taxa are None != score 0)
        # annots, contains parameter annotations (missing taxa get an empty string)
    shape = (report.shape[0], len(report.columns.levels[0]))
    scores = np.full(shape, np.nan)
    annots = np.full(shape, '', dtype=object)
    
    win_headers = []
    for w_idx, (win, win_tab) in enumerate(report.groupby(level=0, axis=1, sort=False)):
        win_headers.append(win)
        win_array = win_tab.to_numpy().astype(float)
        valid_rows = ~np.isnan(win_array[:,3])
        scores[valid_rows, w_idx] = win_array[valid_rows, 3]
        param_array = win_array[valid_rows,:3].astype(int)
        annots[valid_rows, w_idx] = np.array(list(map(lambda x : f'n: {x[0]}\nk: {x[1]}\n{"uwd"[x[2]]}', param_array)))
    scores = pd.DataFrame(scores, index=report.index, columns=win_headers)
    annots = pd.DataFrame(annots, index=report.index, columns=win_headers)
    return scores, annots

def plot_APRF(report_tab, metric, windows, out_dir, lincodes, collapse_hm=True, custom=True):
    """Build heatmap for one of the aprf metrics"""
    # report_tab: table containing a given metric's summary report (best scores per tax per window)
    # metric: full name of the current metric, used to generate the plot title
    # windows: dataframe containing window coordinates
    # out_dir: directory where the generated plots will be stored
    # lincodes: series with index: SciName and values LinCode + ' ' + SciName
    # custom: adjust column width to relative window lengths
    
    # load report table, extract ranks
    scores, annots = get_APRF_heatmap_arrays(report_tab)
    
    # fix report headers
    windows = windows.loc[scores.columns].to_numpy()
    
    # build a hetamap for each rank
    for rk, rk_scores in scores.groupby(level=0, sort=False):
        title = f'{metric} scores for rank: {rk}'
        rk_annots = annots.loc[rk]
        
        # extract score and annotation data
        data = rk_scores.to_numpy()
        annot = rk_annots.to_numpy()
        
        tax_names = rk_scores.droplevel(0).index.values
        indexes = lincodes.loc[tax_names].values
        
        # filter out empty rows (only unknown or 0 support values)
        non_empty_cells = data > 0
        empty_rows = non_empty_cells.sum(1) == 0 # locations of rows with no scores greater than 0
        
        if collapse_hm:
            # remove empty rows from data, annotation and tax names
            data = data[~empty_rows]
            annot = annot[~empty_rows]
            indexes = indexes[~empty_rows]
        
        # heatmaps have at most 1000 rows, if a rank (usually only species, MAYBE genus) exceeds the limit, split it
        if len(data) <= 1000:
            fig = build_heatmap(data, annot, indexes, windows, title, metric, custom=custom)
            fig.savefig(out_dir + f'/{metric}_{rk}.png', format='png', bbox_inches='tight')
            plt.close()
        else:
            for n_subplot, subplot_idx in enumerate(np.arange(0, len(data), 1000)):
                fig = build_heatmap(data[subplot_idx: subplot_idx + 1000], annot[subplot_idx: subplot_idx + 1000], indexes[subplot_idx: subplot_idx + 1000], windows, title, metric, custom=custom)
                fig.savefig(out_dir + f'/{metric}_{rk}.{n_subplot + 1}.png', format='png', bbox_inches='tight')
                plt.close()

def plot_CE(report_tab, windows, out_dir, figsize=(12,7)):
    """Plot cross entrpoy reuslts"""
    # report_tab: dataframe containing cross entropy summary (avg CE per rank per param combo)
    # windows: dataframe containing window coordinates
    # out_dir: directory where the generated plots will be stored
    # figsize: guess
    
    win_labels = windows.apply(lambda x : f'[{x[0]} - {x[1]}', axis=1)
    win_labels = win_labels.loc[report_tab.columns.levels[0]]
    y_coords = pd.Series(np.arange(len(report_tab.columns.levels[0]))[::-1], index = report_tab.columns.levels[0])
    
    for rk, rk_row in report_tab.iterrows():
        fig, ax = plt.subplots(figsize = figsize)
        for win, win_subtab in rk_row.groupby(level=0):
            y = np.full(len(win_subtab),y_coords.loc[win])
            x = win_subtab.values
            best_x = win_subtab.min()
            best_params = win_subtab.sort_values().index[0]
            best_label = f'n: {best_params[1]}, k: {best_params[2]}, mth: {"uwd"[best_params[3]]}'
            ax.scatter(x, y, s = 5, marker = 'o', color = 'k')
            # annotate minimum entropy
            ax.scatter(best_x, y_coords.loc[win], s = 5.5, marker = 'o', color = 'r')
            ax.text(best_x - 0.2, y_coords.loc[win] + 0.2, best_label, size=10, ha='left')
        
        # post process plot
        ax.set_xlim(-0.5, 11) # CE values exist within the [0, 10] range
        ax.set_ylim(-0.5, y_coords.max() + 0.5) # add vertical padding
        ax.set_yticks(np.unique(y_coords))
        ax.set_yticklabels(win_labels.values)
        ax.set_ylabel('Windows')
        ax.set_xlabel('Cross entropy')
        ax.set_title('Cross entropy for rank: ' + rk)
        fig.savefig(out_dir + f'/cross_entropy_{rk}.png', format='png')
        plt.close()
