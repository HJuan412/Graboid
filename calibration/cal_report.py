#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:41:27 2023

@author: nano

Build calibration report
"""

#%% libraries
import numpy as np
import pandas as pd

#%% functions
def build_prereport(metrics, report_by, tax_tab):
    # metrics is the list of metrics results per window generated by get_metricas
    # report_by indicates the metric that will be reported. 0 : accuracy, 1 : precision, 2 : recall, 3 : f1 score
    # tax_tab is the extended taxonomy table of the calibrator
    
    # returns:
        # a dataframe with multiindex (ranks, taxa) and columns : [col0,...,col_n]
        # a list of #ranks 3d arrays of shape (rk_taxa, n_windows, 2) containing the indexes for the winner parameter combinations (nk, method) for each taxon/window
    
    
    # find total taxa per rank
    rank_taxa = [[] for rk in range(tax_tab.shape[1])]
    for met in metrics:
        for rk, rk_taxa in enumerate(met[4]):
            rank_taxa[rk].append(rk_taxa)
    rank_taxa = [np.unique(np.concatenate(rk)) for rk in rank_taxa]
    
    # final report (metric) shape: (taxa, windows)
    n_windows = len(metrics)
    # initialize report matrices
    # report tab has a multiindex with levels rank, taxon, and a column for each window
    report_tab = pd.DataFrame(index=pd.MultiIndex.from_tuples([(idx, tax) for idx, rk in enumerate(rank_taxa) for tax in rk]), columns = np.arange(n_windows), dtype=np.float64)
    # rank_params has n_ranks 3d arrays of shape (len(rank), n_windows, 2), it houses the parameter indexes for each winner cell in the grid
    rank_params = [np.full((rk.shape[0], n_windows, 2), -1) for rk in rank_taxa]
    
    for win, win_metrics in enumerate(metrics):
        met_array = win_metrics[report_by]
        for rk, rk_array in enumerate(met_array):
            win_taxa = win_metrics[4][rk] # retrieve the taxa present in the window
            best_tax_nks = np.max(rk_array, 1) # get the best metrics per nk combination, returns an array of shape (taxa, methods)
            best_nk_idxs = np.argmax(rk_array, 1) # get indexes of the best nk combinations for each array and method
            best_meth_idx = np.argmax(best_tax_nks, 1) # get the index of the best method in best_tax_nks, array of shape(taxa,)
            best_metrics = best_tax_nks[np.arange(best_tax_nks.shape[0]), best_meth_idx] # retireve the best metric for each taxa
            # build an array with the best parameter combination for each taxa
            best_nks = best_nk_idxs[np.arange(best_nk_idxs.shape[0]), best_meth_idx]
            best_params = np.array([best_nks, best_meth_idx]).T
            
            indexes = [(rk, tax) for tax in win_taxa]
            report_tab.loc[indexes, win] = best_metrics
            rank_params[rk][np.isin(rank_taxa[rk], win_taxa), win] = best_params
    return report_tab, rank_params

def translate_params(params, n_range, k_range, methods='uwd'):
    # generate a tuple with the winning parameter combination (n, k, method) for each window/taxon pair
    # for each rank, generate a 2d array of shape(rk_taxa, n_windows)
    param_datum = [np.full((rk.shape[0], rk.shape[1]), np.nan, object) for rk in params]
    
    grid_indexes = [(n_idx, k_idx) for n_idx in np.arange(len(n_range)) for k_idx in np.arange(len(k_range))]
    for rk_idx, rk in enumerate(params):
        # each parameter array in params has two layers, the first one contains the index of the (n/k) pair, the second layer contains the index for the classification method
        nk_plane = rk[..., 0]
        meth_plane = rk[..., 1]
        # get the location of every valid (non -1) cell
        positions = np.argwhere(nk_plane >= 0)
        for pos0, pos1 in positions:
            # retrieve the parameter combinations for each cell
            nk_idx = nk_plane[pos0, pos1]
            meth_idx = meth_plane[pos0, pos1]
            n = n_range[grid_indexes[nk_idx][0]]
            k = k_range[grid_indexes[nk_idx][1]]
            m = methods[meth_idx]
            # update the corresponding cell
            param_datum[rk_idx][pos0, pos1] = (n,k,m)
    return param_datum

def build_report(win_list, metrics, metric, tax_ext, guide, n_range, k_range, win_indexes):
    met_codes = {'acc':0, 'prc':1, 'rec':2, 'f1':3}
    pre_report, params = build_prereport(metrics, met_codes[metric], tax_ext)
    # process report
    pre_report.columns = [f'W{w_idx} [{win.start} - {win.end}]' for w_idx, win in zip(win_indexes, win_list)]
    index_datum = guide.loc[pre_report.index.get_level_values(1)]
    pre_report.index = pd.MultiIndex.from_arrays([index_datum.Rank, index_datum.SciName])
    
    params = translate_params(params, n_range, k_range)
    return pre_report, params