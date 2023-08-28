#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:39:47 2023

@author: nano

Calculate metrics (accuracy, precision, recall, f1 score) for calibration
"""

#%% libraries
import concurrent.futures
import glob
import numpy as np
import os
import pandas as pd

#%% functions
def get_aprf(pred, real):
    """Calculate accuracy, precision, recall and f1 score (aprf)"""
    # pred is the predicted array, shape (#seqs, #ranks)
    # real is the real taxonomy of the calibration sequences, shape (#seqs, #ranks)
    # returns array of shape (#taxa, 6) with coluns: rk_idx tax_idx acc prc rec f1
    
    # only caluclate metrics of sequences with KNOWN taxonomy
    known_real = ~np.isnan(real)
    
    metrics = []
    for rk_idx, (rk_pred, rk_real, rk_known) in enumerate(zip(pred.T, real.T, known_real.T)):
        # extract sequences with known taxonomy at the current RANK
        pred_valid = rk_pred[rk_known]
        real_valid = rk_real[rk_known]
        
        uniq_tax = np.unique(real_valid) # list unique taxa
        
        # prepare metrics table
        rk_metrics = np.zeros((len(uniq_tax), 6)) # columns: rk_idx tax_idx acc prc rec f1, rows: #taxa
        rk_metrics[:, 0] = rk_idx
        rk_metrics[:, 1] = uniq_tax
        
        for tax_idx, tax in enumerate(uniq_tax):
            real_loc = real_valid == tax
            pred_loc = pred_valid == tax
            
            # count real/false positives/negatives
            tp = (real_loc & pred_loc).sum()
            tn = (~real_loc & ~pred_loc).sum()
            fp = pred_loc[~real_loc].sum()
            fn = (~pred_loc[real_loc]).sum()
            
            # calculate metrics
            accuracy = tp / (tp + tn + fp + fn)
            if tp == 0:
                # no predictions for current taxon, values go to 0
                precision = 0
                recall = 0
                f1_score = 0
            else:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1_score = (2 * precision * recall) / (-1 if precision + recall == 0 else precision + recall)
            rk_metrics[tax_idx,2:] = [accuracy, precision, recall, f1_score]
        metrics.append(rk_metrics)
    return np.concatenate(metrics)

def get_cross_entropy(supports, real):
    known_real = ~np.isnan(real)
    clipped_supports = np.clip(supports, np.exp(-10), 1) # max loss is 10, min support is exp(-10) because log function is defined for > 0 values
    log_supports = -np.log(clipped_supports)
    cross_entropy = np.array([col[known].sum() for col, known in zip(log_supports.T, known_real.T)]) / known_real.sum(0)
    return cross_entropy

def get_metrics(results, real_tax):
    """Calculate performance metrix for the predictions obtained with a single window/n/k combination"""
    # metrics is a 3d array of shape (3 (unweighted, wknn, dwknn), # taxa, 6 (rank_id, tax_id, acc, prc, rec, f1))
    # cross entropy is a 2d array of shape: 3 (unweighted, wknn, dwknn), # ranks
    metrics = np.array([get_aprf(results['predicted_u'], real_tax),
                        get_aprf(results['predicted_w'], real_tax),
                        get_aprf(results['predicted_d'], real_tax)])
    
    # Cross Entropy Loss
    cross_entropy = np.array([get_cross_entropy(results['real_u_support'], real_tax),
                              get_cross_entropy(results['real_w_support'], real_tax),
                              get_cross_entropy(results['real_d_support'], real_tax)])
    return metrics, cross_entropy

def aprf_full_report(metrics_dir, metric, guide):
    """Merge metrics for all parameter combinations for all calibrated taxa"""
    # returns a dataframe with row index (Rank, Taxon) and column index (Window, n, k, mth)
    
    metrics_cols = {'a':2, 'p':3, 'r':4, 'f':5}
    metric_idx = metrics_cols[metric[0].lower()]
    
    # list all metrics files
    metrics_list = os.listdir(metrics_dir)
    
    # first pass, exploratory, list uniqe taxa and parameter combinations
    uniq_taxa = []
    params = []
    for fl in metrics_list:
        file_npz = np.load(metrics_dir + '/' + fl)
        tax_data = file_npz['metrics'][0][:,:2].astype(int)
        uniq_taxa.append(tax_data)
        
        fl_params = np.zeros((3,4), int)
        fl_params[:,:3] = file_npz['params']
        fl_params[:,3] = [0,1,2]
        params.append(fl_params)
    
    
    # build index, sorted by rank
    uniq_taxa = np.concatenate(uniq_taxa)
    uniq_taxa = uniq_taxa[np.unique(uniq_taxa[:,1], return_index=True)[1]]
    uniq_taxa = uniq_taxa[np.argsort(uniq_taxa[:,0])]
    tab_index = pd.MultiIndex.from_frame(guide.loc[uniq_taxa[:,1], ['Rank', 'SciName']], names=['Rank', 'Taxon']) # tab index contains full names, added after table is complete
    # build columns, multiindex with 4 levels
    params = np.concatenate(params, dtype=np.int16)
    params = params[np.lexsort((params[:,3], params[:,2], params[:,1], params[:,0]))]
    tab_columns = pd.MultiIndex.from_arrays(params.T, names=['Window', 'n', 'k', 'mth'])
    # build metrics table
    metrics_tab = pd.DataFrame(index=uniq_taxa[:,1], columns = tab_columns, dtype=np.float16)
    
    # second pass, populate table
    for fl in metrics_list:
        file_npz = np.load(metrics_dir + '/' + fl)
        params = file_npz['params']
        metrics = file_npz['metrics']
        # metrics is a 3d array of shape (3 (unweighted, wknn, dwknn), # taxa, 6 (rank_id, tax_id, acc, prc, rec, f1))
        
        for mth_idx, mth_metrics in enumerate(metrics):
            taxa = mth_metrics[:,1]
            pars = tuple(np.append(params, mth_idx))
            metrics_tab.loc[taxa, pars] = mth_metrics[:,metric_idx]
    
    metrics_tab.index = tab_index
    return metrics_tab

def get_window_CE(window_files, window_taxa, guide):
    """Build a table with cross entropy for each taxon for each parameter combination in a given window"""
    params = []
    # first pass, exploratory. Retrieve parameters
    for fl in window_files:
        npz = np.load(fl)
        
        fl_params = np.zeros((3,4), int)
        fl_params[:,:3] = npz['params']
        fl_params[:,3] = [0,1,2]
        params.append(fl_params)
        
    # Build results and count tables
    params = np.concatenate(params)
    columns = pd.MultiIndex.from_arrays(params.T, names='Window n k mth'.split()).sort_values()
    index = np.unique(window_taxa)
    index = pd.Index(index[~np.isnan(index)].astype(int))
    
    results = pd.DataFrame(-1, index=index, columns=columns, dtype=np.float16)
    tax_counts = pd.Series(0, index=index, dtype=int)
    
    # hierarchic sort taxa
    sorted_tax = np.lexsort(window_taxa.T)
    real_taxa = window_taxa[sorted_tax]
    
    # second pass, populate tables
    for fl in window_files:
        npz = np.load(fl)
        fl_params = npz['params']
        real_supports = np.array([npz['real_u_support'],
                                  npz['real_w_support'],
                                  npz['real_d_support']])
        
        # clip and log supports
        clipped_supports = np.clip(real_supports, np.exp(-10), 1) # max loss is 10, min support is exp(-10) because log function is defined for > 0 values
        log_supports = -np.log(clipped_supports)
        
        # calculate Cross Entropy for each parameter combination in the window
        # calculate entropy
        for rk_idx, rk_taxa in enumerate(real_taxa.T):
            known_real = ~np.isnan(rk_taxa)
            rk_log_supports = log_supports[:,:,rk_idx].T[sorted_tax][known_real]
            
            rk_taxa = rk_taxa[known_real]
            
            taxa, taxa_start, taxa_count = np.unique(rk_taxa, return_index=True, return_counts=True)
            taxa_end = taxa_start + taxa_count
            
            for tax, tax_start, tax_end, tax_count in zip(taxa, taxa_start, taxa_end, taxa_count):
                tax_log_supp = rk_log_supports[tax_start:tax_end]
                tax_CE = tax_log_supp.sum(0) / tax_count
                results.loc[tax, tuple(fl_params)] = tax_CE
                tax_counts.loc[tax] = tax_count
    
    # reindex results, replace taxIDs with (Rank, Taxon)
    index = pd.MultiIndex.from_frame(guide.loc[results.index, ['Rank', 'SciName']], names='Rank Taxon'.split())
    results.index = index
    tax_counts.index = index    
    return results, tax_counts

def CE_full_report(window_list, window_indexes, classif_dir, taxonomy, guide):
    """Get the full CE for each taxon for each window, """
    results = []
    counts = []
    for win_idx, window in zip(window_indexes, window_list):
        win_taxa = taxonomy.loc[window.taxonomy]
        win_files = glob.glob(classif_dir + f'/{win_idx}*')
        window_CE, tax_count = get_window_CE(win_files, win_taxa.to_numpy(), guide = guide)
        results.append(window_CE)
        tax_count.name = win_idx
        counts.append(tax_count.to_frame())
    results = pd.concat(results, axis=1)
    counts = pd.concat(counts, axis=1).fillna(0).astype(int)
    
    # sort results and counts by rank
    ranks = taxonomy.columns.tolist()
    rank_dict = {rk:idx for idx,rk in enumerate(ranks)}
    keyfunk = np.vectorize(lambda x : rank_dict[x])
    results.sort_index(level=0, key = keyfunk, inplace=True)
    counts.sort_index(level=0, key = keyfunk, inplace=True)
    return results, counts

def read_full_report_tab(file):
    # read the csv file generated by CE_full_report and aprf_full_report, process column level datatypes
    tab = pd.read_csv(file, index_col=[0,1], header=[0,1,2,3])
    tab.columns = tab.columns.set_levels(tab.columns.levels[0].astype(int), level=0)
    tab.columns = tab.columns.set_levels(tab.columns.levels[1].astype(int), level=1)
    tab.columns = tab.columns.set_levels(tab.columns.levels[2].astype(int), level=2)
    tab.columns = tab.columns.set_levels(tab.columns.levels[3].astype(int), level=3)
    return tab

def groupby_np(array, col):
    # array must be sorted by column
    values, starts, counts = np.unique(array[:,col], return_index=True, return_counts=True)
    ends = starts + counts
    for val, start, end in zip(values, starts, ends):
        yield val.astype(int), array[start:end]
        
def build_confusion(pred, real):
    """Builds the confusion matrix for a given prediction"""
    # pred is the predicted array, shape (#seqs, #ranks)
    # real is the real taxonomy of the calibration sequences, shape (#seqs, #ranks)
    
    # returns pandas dataframe of shape (#taxa, #taxa) with index multiindex(rank, real_taxon) and columns multiindex(rank, pred_taxon)
    # contains COUNT of predictions
    n_ranks = real.shape[1]
    n_seqs = real.shape[0]
    rank_indexes = np.tile(np.arange(n_ranks, dtype=int), (n_seqs, 1))
    
    known_real = ~np.isnan(real)
        
    flat_known = known_real.flatten()
    flat_real = real.flatten()[flat_known].astype(int)
    flat_pred = pred.flatten()[flat_known].astype(int)
    flat_ranks = rank_indexes.flatten()[flat_known]
    
    sorted_headers = np.lexsort((flat_real, flat_ranks))
    flat_ranks = flat_ranks[sorted_headers]
    flat_real = flat_real[sorted_headers]
    flat_pred = flat_pred[sorted_headers]
    
    collapsed_taxa, collapsed_idxs = np.unique(flat_real, return_index=True)
    collapsed_idxs = np.sort(collapsed_idxs)
    
    confusion_index = pd.MultiIndex.from_arrays((flat_ranks[collapsed_idxs], flat_real[collapsed_idxs].astype(int)), names=['rank', 'real_taxon'])
    confusion_columns = pd.MultiIndex.from_arrays((flat_ranks[collapsed_idxs], flat_real[collapsed_idxs].astype(int)), names=['rank', 'pred_taxon'])
    confusion = pd.DataFrame(0, index=confusion_index, columns=confusion_columns)
    
    for rk, rk_subtab in groupby_np(np.array((flat_ranks, flat_real, flat_pred)).T, 0):
        for real_tax, tax_subtab in groupby_np(rk_subtab, 1):
            pred_taxa, pred_counts = np.unique(tax_subtab, return_counts=True)
            confusion.loc[rk, rk].loc[real_tax, pred_taxa] = pred_counts
    
    # get fractions
    # (confusion.T / confusion.sum(1)).T # get fraction of pred_values / real_values (of the real values of taxon, what proportions are predicted as each taxon)
    # confusion / confusion.sum(0) # get fraction of real_values / pred_values (of the predicted values of taxon, what proportion belong to each REAL taxon)
    return confusion

#%% OLD functions
def get_window_metrics(win_classif, window_tax):
    # calcualtes the metrics for every parameter combination and taxonomic rank in win_classif
    
    # retrieve the (n_sites-k, neighbours) pairs used as keys in win_classif
    nk_pairs = list(win_classif.keys())
    # one 3d-array per taxonomic rank: (taxon, method, n-k)
    
    # reorganize: make #ranks 3d arrays of shape (n-k, seq_classif, methods(3))
    rank_mats = []
    for rk in np.arange(window_tax.shape[1]):
        rk_mat = []
        for nk in nk_pairs:
            rk_mat.append(win_classif[nk][:,rk,:])
        rank_mats.append(np.array(rk_mat))
    
    # calculate the metrics for all the ranks
    # return a tuple of (accuracies, precisions, recalls and f1 scores arrays for each taxonomic rank)
    # each metrics array has shape (n-k, rank_taxa, methods(3))
    return calculate_metrics(rank_mats, window_tax)

def calculate_metrics(rank_mats, tax_tab):
    rk_accs = []
    rk_prcs = []
    rk_recs = []
    rk_f1s = []
    rk_tax = [] # store the taxa included in the current window
    # for every rank, calculate metrics using the true (rk) and predicted (rk_mat) arrays
    for rk, rk_mat in zip(tax_tab.T, rank_mats):
        # get unique taxa in the current rank
        rk_taxa = np.unique(rk)
        rk_taxa = rk_taxa[~np.isnan(rk_taxa)]
        
        acc = []
        prc = []
        rec = []
        f1 = []
        
        # calculate metrics for each taxon, for each parameter combination
        for tax_idx, tax in enumerate(rk_taxa):
            # get locations of taxon in the true and predicted arrays
            
            tax_loc = rk == tax
            # tax_loc is originally an array of shape (#seqs,) in order to use it in logical comparisons with tax_classif_loc (n-k, seq_classif, methods(3)) we need to transform tax_loc in the following line
            tax_loc_cube = np.array([np.tile(tax_loc, (rk_mat.shape[2],1)).T]*rk_mat.shape[0])
            # np.tile repeats tax_loc n_methods times, resulting in a 2d array of shape (n_methods, n_seqs)
            # external np.array of [np.tile.T]*rk_mat.shape[0] transposes internal array and repeats it for the number of n-k combinations
            # generated array has shape (n-k, n_seqs, n_methods)
            
            tax_classif_loc = rk_mat == tax
            
            # get metrics, true positives and negatives calculated by comparing the location arrays
            tp = (tax_loc_cube & tax_classif_loc).sum(1)
            tn = (~tax_loc_cube & ~tax_classif_loc).sum(1)
            fp = []
            fn = []
            
            # iterate over the first dimension (n-k pairs) of the rk_mat locations
            tax_loc_sample = tax_loc_cube[0].T # take the first layer of tax_loc (all are the same) and transpose it, will use it to get false positives and false negatives
            for nk_tax_classif_loc in tax_classif_loc:
                # calculate false positives and degatives for each column (method) in nk_tax_classif_loc
                # this must be done sepparately because sub arrays for each method may have different lengths
                fp.append(np.array([(~tl_samp[nk_tc_loc]).sum() for tl_samp, nk_tc_loc in zip(tax_loc_sample, nk_tax_classif_loc.T)]))
                fn.append(np.array([(~nk_tc_loc[tl_samp]).sum() for tl_samp, nk_tc_loc in zip(tax_loc_sample, nk_tax_classif_loc.T)]))
            fp = np.array(fp)
            fn = np.array(fn)
            
            
            # calculate metrics (accuracy, precision, recall and F1)
            tax_acc = (tp + tn) / (tp + tn + fp + fn)
            pred_pos = tp + fp # predicted positives
            all_pos = tp + fn # total positives
            # values of 0 would generate a division by 0 error, replace for -1
            pred_pos[pred_pos == 0] = -1
            all_pos[all_pos == 0] = -1
            # clip ensures negative values are set to 0
            tax_prc = np.clip(tp / (pred_pos), 0, 1)
            tax_rec = np.clip(tp / (all_pos), 0, 1)
            pr = tax_prc + tax_rec # precision * recall
            # again, values of 0 would generate an error, set to -1 and clip division
            pr[pr == 0] = -1
            tax_f1 = np.clip((2 * tax_prc * tax_rec)/pr, 0, 1)
            
            acc.append(tax_acc)
            prc.append(tax_prc)
            rec.append(tax_rec)
            f1.append(tax_f1)
        rk_accs.append(np.array(acc))
        rk_prcs.append(np.array(prc))
        rk_recs.append(np.array(rec))
        rk_f1s.append(np.array(f1))
        rk_tax.append(rk_taxa)
    return rk_accs, rk_prcs, rk_recs, rk_f1s, rk_tax

def get_metrics_OLD(win_list, win_classifs, tax_ext, threads=1):
    window_taxes = [tax_ext.loc[win.taxonomy].to_numpy() for win in win_list]
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        metrics = [future for future in executor.map(get_window_metrics, win_classifs, window_taxes)]
    return metrics

def get_metrics_per_func(pred, real):
    rk_idxs = np.arange(real.shape[1])
    
    metrics = []
    for rk in rk_idxs:
        rk_real = real.T[rk]
        rk_pred = pred.T[rk]
        
        uniq_tax = np.unique(rk_real)
        
        rk_metrics = np.zeros((len(uniq_tax), 6))
        rk_metrics[:, 0] = rk
        rk_metrics[:, 1] = uniq_tax
        
        for tax_idx, tax in enumerate(uniq_tax):
            real_loc = rk_real == tax
            pred_loc = rk_pred == tax
            
            tp = (real_loc & pred_loc).sum()
            tn = (~real_loc & ~pred_loc).sum()
            fp = pred_loc[~real_loc].sum()
            fn = (~pred_loc[real_loc]).sum()
            
            accuracy = tp / (tp + tn + fp + fn)
            if tp == 0:
                precision = 0
                recall = 0
                f1_score = 0
            else:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1_score = (2 * precision * recall) / (-1 if precision + recall == 0 else precision + recall)
            rk_metrics[tax_idx,2:] = [accuracy, precision, recall, f1_score]
        metrics.append(rk_metrics)
    return np.concatenate(metrics)

def get_cross_entropy_OLD(supports, valid):
    clipped_supports = np.clip(supports, np.exp(-10), 1) # max loss is 10
    log_supports = -np.log(clipped_supports)
    cross_entropy = np.array([col[val].sum() for col, val in zip(log_supports.T, valid.T)]) / valid.sum(0)
    return cross_entropy

def get_metrics0(results, real_tax):
    # metrics is a 3d array of shape: # taxa, 6 (rank_id, tax_id, acc, prc, rec, f1), 3 (unweighted, wknn, dwknn)
    # real_tax is a 2d array with the (REAL) tax ids of the query sequences
    # cross entropy is a 2d array of shape: 3 (unweighted, wknn, dwknn), # ranks
    metrics = np.array([get_metrics_per_func(results['predicted_u'], real_tax).T,
                        get_metrics_per_func(results['predicted_w'], real_tax).T,
                        get_metrics_per_func(results['predicted_d'], real_tax).T]).T
    
    valid = real_tax != -2 # get the locations of all known classifications
    total_valid = valid.sum(0) # array of shape #ranks, counts known taxons per rank
    # Cross Entropy Loss
    cross_entropy = np.array([get_cross_entropy_OLD(results['real_u_support'], valid),
                              get_cross_entropy_OLD(results['real_w_support'], valid),
                              get_cross_entropy_OLD(results['real_d_support'], valid)])
    return metrics, cross_entropy, total_valid
