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
    
    # returns 2d-array of shape (#taxa, #taxa + 1)
        # first column contains the taxIDs (already sorted by rank)
        # rows: REAL taxa
        # columns: PREDICTED taxa
        # confusion[i, j]: instances belonging to taxon i classified as taxon j
        # confusion[:,1:].sum(axis=0) gives the total PREDICTIONS for each taxon
        # confusion[:,1:].sum(axis=1) gives the total instances of REAL taxa for which a prediction was made, real counts for each taxa in each column are stored in the taxa_report
    
    # sort rows of pred and real arrays by hierarchical order based on the real taxa table
    sorted_rows = np.lexsort(list((col for col in real.T[::-1])))
    real = real[sorted_rows]
    pred = pred[sorted_rows]

    # pre build confusion table
    taxa = np.array([], dtype=int)
    for row in real.T:
        taxa = np.concatenate((taxa, np.unique(row)))
    taxa = taxa[taxa >= 0]
    confusion = pd.DataFrame(0, index=taxa, columns=taxa, dtype=int)
    
    for real_col, pred_col in zip(real.T, pred.T):
        real_taxa, starts, counts = np.unique(real_col, return_index=True, return_counts=True)
        ends = starts + counts
        starts = starts[real_taxa >= 0]
        ends = ends[real_taxa >= 0]
        counts = counts[real_taxa >= 0]
        real_taxa = real_taxa[real_taxa >= 0]
        for real_tax, start, end, count in zip(real_taxa, starts, ends, counts):
            pred_tax_array = pred_col[start:end]
            pred_tax, pred_counts = np.unique(pred_tax_array, return_counts=True)
            confusion.loc[real_tax, pred_tax] = pred_counts
    try:
        confusion.drop(columns=-1, inplace=True)
    except KeyError:
        pass
    return confusion

def confusion_matrix(win_taxa, window, classif_file, method, guide):
    """Build a confusion matrix dataframe with multiindex rank, taxon for real (rows) and predicted (columns) taxa"""
    # win_taxa: npz file generated during grid search, contains the taxonomy for every collapsed window
    # window: window index
    # classif_file: file containing the classification results for a given parameter combination (must belong to the given window)
    # method: u/w/d, weighting method used
    # guide: taxonomy guide containing scientific name and rank for each taxID
    
    # returns dataframe of shape (#taxa, #taxa), index:multiindex (Rank, Taxa), columns:multiindex (Rank, Taxa)
    
    real = np.load(win_taxa)[str(window)]
    pred = np.load(classif_file)[f'predicted_{method[0].lower()}']
    confusion = build_confusion(pred, real)
    index = pd.MultiIndex.from_frame(guide.loc[confusion.index, ['Rank', 'SciName']].rename(columns={'SciName':'Taxon'}))
    cols = pd.MultiIndex.from_frame(guide.loc[confusion.columns, ['Rank', 'SciName']].rename(columns={'SciName':'Taxon'}))
    confusion.index = index
    confusion.columns = cols
    return confusion
