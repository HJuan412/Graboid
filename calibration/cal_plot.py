#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:42:33 2023

@author: nano

Plot the calibration results
"""

#%% libraries
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#%% functions
def build_heatmap(data,
                  param_annots,
                  title,
                  metric,
                  cell_size=45,
                  dpi=300,
                  height_pad=5,
                  ratio_pad=10,
                  hspace=0.01):
    # set color map
    viridis = cm.get_cmap('viridis', 256)
    grey = np.array([128/256, 128/256, 128/256, 1])
    viridis.set_bad(grey)
    
    # calculate heatmap and figure dimensions
    map_width = data.shape[1] * cell_size
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
    hm = sns.heatmap(data,
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
    
    # place labels for all windows and taxa
    hm.set_yticks(np.arange(data.shape[0]) + 0.5)
    hm.set_yticklabels(data.index, size = ytick_size, rotation = 30, va='top')
    hm.set_xticks(np.arange(data.shape[1]) + 0.5)
    hm.set_xticklabels(data.columns, size = xtick_size, rotation=60, ha='left')
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

def build_annot(params):
    # this function generates an annotations array for a given rank's report
    def get_annot(cell):
        if isinstance(cell, tuple):
            return '%d\n%d\n%s' % cell
        return ''
    transform = np.vectorize(get_annot)
    annot = transform(params)
    return annot

def collapse_report(report):
    # remove rows that have all null or 0 values
    max_vals = report.max(axis=1)
    filtered_taxa = max_vals.loc[max_vals > 0].index
    rejected_taxa = max_vals.loc[max_vals == 0].index.to_numpy()
    collapsed = report.loc[filtered_taxa]
    return collapsed, rejected_taxa

def plot_results(report, params, metric, prefix, ranks, collapse=True):
    # collapse eliminates taxa with no values over 0 to reduce heatmap size, if set to True, a file of rejected taxa is generated
    collapse_dict = {rk:[] for rk in ranks}
    for rk_idx, rk in enumerate(ranks):
        rk_report = report.loc[rk]
        sort_taxa = np.argsort(rk_report.index)
        rk_report = rk_report.sort_index()
        rk_params = params[rk_idx][sort_taxa]
        if collapse:
            rk_idx = rk_report.index.to_numpy()
            rk_report, rk_rejected = collapse_report(rk_report)
            rk_params = rk_params[(~np.isin(rk_idx, rk_rejected))]
            collapse_dict[rk] = rk_rejected
        rk_annot = build_annot(rk_params)
        title = f'{metric} scores for rank: {rk}\nN\nK\nmethod (u = unweighted, w = wKNN, d = dwKNN)'
        
        # heatmaps have at most 1000 rows, if a rank (usually only species, MAYBE genus) exceeds the limit, split it
        if len(rk_report) <= 1000:
            fig = build_heatmap(rk_report, rk_annot, title, metric)
            fig.savefig(prefix + f'/{metric}_{rk}.png', format='png', bbox_inches='tight')
            plt.close()
        else:
            for n_subplot, subplot_idx in enumerate(np.arange(0, len(rk_report), 1000)):
                fig = build_heatmap(rk_report.iloc[subplot_idx: subplot_idx + 1000], rk_annot[subplot_idx: subplot_idx + 1000], title, metric)
                fig.savefig(prefix + f'/{metric}_{rk}.{n_subplot + 1}.png', format='png', bbox_inches='tight')
                plt.close()
    if collapse:
        with open(prefix + f'/{metric}_rejected.txt', 'w') as handle:
            handle.write('The following taxa yielded no {metric} scores over 0 and were ommited from the plots:\n')
            for rk in ranks:
                if len(collapse_dict[rk]) == 0:
                    continue
                handle.write(f'{rk}:\n')
                for tax in collapse_dict[rk]:
                    handle.write(f'\t{tax}\n')
