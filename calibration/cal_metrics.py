#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:39:47 2023

@author: nano

Calculate metrics (accuracy, precision, recall, f1 score) for calibration
"""

#%% libraries
import concurrent.futures
import numpy as np

#%%
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

def get_metrics(win_list, win_classifs, tax_ext, threads=1):
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
            fn = ~pred_loc[real_loc].sum()
            
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

def get_cross_entropy(supports):
    clipped_supports = np.clip(supports, np.exp(-5), 1)
    cross_entropy = np.log(clipped_supports).sum(axis=0)
    return cross_entropy

def get_metrics0(results, real_tax):
    # results is a dict containing results arrays generated by cal_classify.get_supports
    # acc, prec, rec, f1
    metrics = np.array([get_metrics_per_func(results['predicted_u'], real_tax),
                        get_metrics_per_func(results['predicted_w'], real_tax),
                        get_metrics_per_func(results['predicted_d'], real_tax)])
    
    # Cross Entropy Loss
    cross_entropy = np.array([get_cross_entropy(results['real_u_support']),
                              get_cross_entropy(results['real_w_support']),
                              get_cross_entropy(results['real_d_support'])])
    return metrics, cross_entropy
