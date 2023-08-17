#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:41:27 2023

@author: nano

Build calibration report
"""

#%% libraries
import glob
import numpy as np
import pandas as pd

#%%
def compare_cross_entropy(files):
    # returns array of shape: # files * 3 (unweighted, wknn, dwknn), 4 (window, n, k, method, ) + # ranks
    cross_entropy_report = []
    for fil in files:
        fil_dict = np.load(fil)
        fil_params = fil_dict['params'] # [window_idx, n, k]
        fil_cross_entropy = fil_dict['cross_entropy'] # 2-d array of shape: 3 (unweighted, wknn, dwknn), # ranks
        package_report = np.zeros((fil_cross_entropy.shape[0], fil_cross_entropy.shape[1]+4))
        package_report[:, 4:] = fil_cross_entropy
        package_report[:,:3] = fil_params
        package_report[:,3] = np.arange(1,4)
        cross_entropy_report.append(package_report)
    return np.concatenate(cross_entropy_report)

def build_ce_report(report, ranks):
    """Build the cross entropy dataframe"""
    # report: 2d-array of shape (#param combinations, (window, n, k, method, ranks...))
    # ranks: list of taxonomic rank names
    # returns datafrmae of shape(#param combinations, (window, n, k, method, ranks...))
    
    # name dataframe columns
    columns = ['Window', 'n', 'k', 'Method'] + ranks
    method_dict = {-1:'', 1:'u', 2:'w', 3:'d'} # assign method codes
    ce_report = pd.DataFrame(report, columns=columns)
    # set column data types
    ce_report[['Window', 'n', 'k']] = ce_report[['Window', 'n', 'k']].astype(np.int16)
    ce_report.Method.replace(method_dict, inplace=True)
    return ce_report

def build_aprf_report(template, window_indexes, window_reports, col):
    """Build a report for a given aprf metric"""
    # template: template dataframe for the metric report
    # window_indexes: array of window indexes
    # window_reports: list of tuples (tax_data, winner_vals, params) (one per window)
    # col: index of the current metric column ({acc:0, prc:1, rec:2, f1:4})
    # returns a copy of template, populated with the corresponding values
    
    report = template.copy()
    # extract values from each window report (window reports are unpacked into tax_data, winner_vals, params_report)
    for win_idx, (tax_data, winner_vals, params_report) in zip(window_indexes, window_reports):
        indexes = list(map(tuple, tax_data)) # get row indexes
        vals = winner_vals[:,col] # get scores for the current metric
        n, k, met = params_report[:,col].T # get params for the winning scores
        report.loc[indexes, win_idx] = np.array((n, k, met, vals)).T # populate report
    return report

def postprocess_aprf(report, guide):
    """Update aprf reports to human-readable"""
    # report: build_aprf_report output dataframe
    # guide: taxonomy guide (contains LinCode)
    
    # prepare human-readable index (replace rank and tax ids with full names)
    tax_ids = report.index.get_level_values(1)
    tax_names = (guide.LinCode + ' ' + guide.SciName).loc[tax_ids]
    rank_names = guide.Rank.loc[tax_ids]
    new_index = pd.MultiIndex.from_arrays((rank_names, tax_names), names=['Rank', 'Taxon'])
    
    # update column datatypes and apply method codes
    win_indexes = report.columns.get_level_values(0).unique()
    method_dict = {-1:'', 0:'u', 1:'w', 2:'d'}
    for win_idx in win_indexes:
        report[[(win_idx, 'n'), (win_idx, 'k')]] = report[[(win_idx, 'n'), (win_idx, 'k')]].astype(np.int16)
        report[(win_idx, 'Method')].replace(method_dict, inplace=True)
    report.index = new_index
    
def build_reports(win_indexes, metrics_dir, out_dir, ranks, guide):
    """Process metrics files and build user-readable reports"""
    # win_indexes: array containing the indexes of the selected windows
    # metrics_dir: directory containing the metric files
    
    cross_entropy_array = []
    aprf_arrays = []
    
    for win_idx in win_indexes:
        # open window metrics and retrieve data
        window_reports = glob.glob(metrics_dir + f'/{win_idx}*')
        cross_entropy_array.append(compare_cross_entropy(window_reports))
        aprf_arrays.append(compare_metrics(window_reports)) # retrieve the BEST metric values and generating parameter combinations for each window
        # aprf_arrays contains tuples of (tax_data, winner_vals, params). One tuple per window
    
    # build cross entropy report (dataframe of columns: window, n, k, metod, ranks...)
    cross_entropy_array = np.concatenate(cross_entropy_array)
    cross_entropy_report = build_ce_report(cross_entropy_array, ranks)
    cross_entropy_report.to_csv(out_dir + '/cross_entropy.csv')
    
    # prepare aprf report index
    # get all taxa represented across the different windows
    merged_taxa = np.concatenate([mt_array[0] for mt_array in aprf_arrays]) # remember, mt_array is a 3 element tuple, first one is the tax data array
    uniq_taxa, uniq_locs = np.unique(merged_taxa[:,1], return_index=True)
    merged_taxa = merged_taxa[uniq_locs]
    merged_taxa = merged_taxa[np.lexsort((merged_taxa[:,1], merged_taxa[:,0]))] # merged taxonomy sorted by rank and tax ids
    # build indexes
    report_index = pd.MultiIndex.from_arrays((merged_taxa[:,0], merged_taxa[:,1]), names=['Rank', 'Taxon'])
    report_columns = pd.MultiIndex.from_product((win_indexes, ['n', 'k', 'Method', 'Score']), names=['Window', ''])
    template_report = pd.DataFrame(-1., index=report_index, columns=report_columns)
    
    # build a report for each metric
    for met_col, met in enumerate(['acc', 'prc', 'rec', 'f1']):
        met_report = build_aprf_report(template_report, win_indexes, aprf_arrays, met_col)
        postprocess_aprf(met_report, guide)
        met_report.to_csv(out_dir + f'/{met}_report.csv')
#%% OLD functions
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

def build_report(win_list, metrics, metric, tax_ext, guide, n_range, k_range):
    met_codes = {'acc':0, 'prc':1, 'rec':2, 'f1':3}
    pre_report, params = build_prereport(metrics, met_codes[metric], tax_ext)
    # process report
    pre_report.columns = [(win.start, win.end) for win in win_list]
    index_datum = guide.loc[pre_report.index.get_level_values(1)]
    pre_report.index = pd.MultiIndex.from_arrays([index_datum.Rank, index_datum.SciName])
    
    params = translate_params(params, n_range, k_range)
    return pre_report, params

def compare_metrics(files):
    # for each metric (accuracy, precision, recall, f1) generate a report of the parameter combinations that yield the best results for each taxon
    # returns acc_report, prc_report, rec_report, f1_report. 2d arrays of shape: # taxa, rank, taxon, window_idx, n, k, method, score. Only the best parameter combination and the generated score are recorded per taxon
    params = []
    metrics = []
    tax_data = []
    for fil in files:
        fil_dict = np.load(fil)
        params.append(fil_dict['params']) # [window_idx, n, k]
        metrics.append(fil_dict['metrics'][:,1:,:]) # 3d array of shape: # taxa, 6 (rank_id, tax_id, acc, prc, rec, f1), 3 (unweighted, wknn, dwknn)
        tax_data.append(fil_dict['metrics'][:,:2,0]) # retrieve taxonomic data (first 2 columns of first layer (any layer would do))
    
    params = np.array(params)
    tax_data = np.concatenate(tax_data).astype(np.int32)
    taxa, tax_loc = np.unique(tax_data[:,1], return_index = True) # get number of taxon and index location to match tax and rank

    template_tab = np.zeros((len(taxa), 6, len(params))) # rows: taxa, columns: rank, taxon, acc, prec, rec, f1, layers: param (n-k) combos
    template_tab[:, 0] = np.tile(tax_data[tax_loc, 0], (len(params), 1)).T
    template_tab[:, 1] = np.tile(tax_data[tax_loc, 1], (len(params), 1)).T
    
    idx_dict = {tax:tax_idx for tax_idx,tax in enumerate(taxa)} # used to place a taxon on nk_tab and method_tab
    nk_tab = template_tab.copy() # holds the scores of the winning method for each taxon for each metric, for each param combination
    method_tab = template_tab.copy().astype(np.int32) # holds the indexes of the winning methods for each cell in nk_tab
    
    # select the winning method for each metric for each n-k combination
    for met_idx, mets in enumerate(metrics):
        indexes = [idx_dict[mt] for mt in mets[:,0,0]] # get the indexes of the sequences that appeared in this parameter combination
        nk_tab[indexes, 2:, met_idx] = np.max(mets[:,1:], axis = 2) # update nk_tab with best scores for the current n-k combination
        method_tab[indexes, 2:, met_idx] = np.argmax(mets[:,1:], axis = 2) # record methods that generated the best scores for each taxon for each metric
    
    nk_winner = np.max(nk_tab[:,2:], axis = 2) # get the best score for each taxon for each n-k combination
    nk_idxs = np.argmax(nk_tab[:,2:], axis = 2) # get the indexes of the n-k combinations that generated the best scores for each taxon for each metric
    method_winners = method_tab[:,2:,:][np.indices(nk_idxs.shape)[0], np.indices(nk_idxs.shape)[1], nk_idxs] # retrieve the methods that generated the best scores for each taxon for each metric for each n-k combination
    
    template_report = np.zeros((len(taxa), 2 + 4 + 1), dtype=np.float32) # first 2 columns are rank & taxon, next 4 are window, n, k, method, last 1 is best score. Table includes only the best scores and the param combination that generated them
    template_report[:, :2] = tax_data[tax_loc]
    acc_report = template_report.copy()
    prc_report = template_report.copy()
    rec_report = template_report.copy()
    f1_report = template_report.copy()

    # record winner n-k
    acc_report[:, 2:5] = params[nk_idxs[:,0]]
    prc_report[:, 2:5] = params[nk_idxs[:,1]]
    rec_report[:, 2:5] = params[nk_idxs[:,2]]
    f1_report[:, 2:5] = params[nk_idxs[:,3]]

    # record winner method
    acc_report[:, 5] = method_winners[:, 0]
    prc_report[:, 5] = method_winners[:, 1]
    rec_report[:, 5] = method_winners[:, 2]
    f1_report[:, 5] = method_winners[:, 3]

    # record best metrics
    acc_report[:, -1] = nk_winner[:,0]
    prc_report[:, -1] = nk_winner[:,1]
    rec_report[:, -1] = nk_winner[:,2]
    f1_report[:, -1] = nk_winner[:,3]
    
    return acc_report, prc_report, rec_report, f1_report