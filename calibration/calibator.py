#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 11:06:10 2021

@author: hernan
"""

import sys
sys.path.append('preprocess')
sys.path.append('classif')
#%% libraries
import numpy as np
import pandas as pd
from classif import classification
from classif import cost_matrix
from preprocess import feature_selection as fsele
from preprocess import windows

#%% functions
def get_metrics(confusion):
    taxons = confusion.index.tolist()
    metrics = pd.DataFrame(index=confusion.index, columns=['taxon', 'accuracy', 'precision', 'recall', 'F1_score'])
    
    for tax in taxons:
        tp = confusion.loc[tax, tax]
        tn = confusion.loc[confusion.index != tax, confusion.columns != tax].to_numpy().sum()
        fp = confusion.loc[confusion.index != tax, tax].to_numpy().sum()
        fn = confusion.loc[tax, confusion.columns != tax].to_numpy().sum()

        acc = (tp + tn) / (tp + tn + fp + fn)
        prc = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = (2 * prc * rec)/(prc + rec)
        
        tax_metrics = pd.Series({'taxon':tax,
                                 'accuracy':acc,
                                 'precision':prc,
                                 'recall':rec,
                                 'F1 score':f1})
        metrics = np.concat([metrics, tax_metrics], ignore_index=True)
    
    metrics.fillna(0, inplace = True)
    return metrics

def get_report_filename(out_tab, mat_path):
    split_file = mat_path.split('/')[-1].split('.mat')[0].split('_')
    filename = f'{split_file[0]}_{split_file[2]}_{split_file[3]}.csv'
    return f'{out_tab}/{filename}'

def loo_generator(nrecs):
    # generates the indexes for the training dataset and the testing instance in leave-one-out calibration
    record_idxs = np.arange(nrecs)
    for idx in record_idxs:
        train_idx = np.delete(record_idxs, idx)
        test_idx = idx
        yield train_idx, test_idx

def build_confusion(pred, real):
    # build the confusion matrix (for a given RANK)
    uniq_taxes = np.unique(real)
    confusion = pd.DataFrame(data=0, index=uniq_taxes, columns=uniq_taxes, dtype=int)
    for p,r in zip(pred, real):
        # rows: real value
        # cols: predicted value
        confusion.at[r,p] += 1
    return confusion
    
def build_cal_tab(pred_tax, real_tax):
    cal_tab = pd.DataFrame(columns=['rank',
                                    'taxon',
                                    'w_start',
                                    'w_end',
                                    'K',
                                    'n_sites',
                                    'accuracy',
                                    'precision',
                                    'recall',
                                    'F1_score'])
    
    for rank in real_tax.columns:
        rank_confusion = build_confusion(pred_tax[rank].to_numpy(), real_tax[rank].to_numpy())
        rank_metrics = get_metrics(rank_confusion)
        rank_metrics['rank'] = rank
        cal_tab = pd.concat([cal_tab, rank_metrics], ignore_index=True)
    return cal_tab
    
#%% classes
class Calibrator():
    def __init__(self, taxon, marker, in_dir, out_dir, tmp_dir, warn_dir, row_thresh=0.2, col_thresh=0.2, min_seqs=10):
        self.taxon = taxon
        self.marker = marker
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.tmp_dir = tmp_dir
        self.warn_dir = warn_dir
        self.loader = windows.WindowLoader(taxon, marker, in_dir, out_dir, tmp_dir, warn_dir)
        self.row_thresh = row_thresh
        self.col_thresh = col_thresh
        self.min_seqs = min_seqs
    
    # leave one out calibration
    def loo_calibrate(self, w_size, w_step, max_k, step_k, max_sites, step_sites, dist_mat, mode='majority', support_func=classification.wknn):
        # set up parameter ranges
        # window coordinates
        max_pos = self.loader.dims[1]
        start_range = np.arange(0, max_pos - w_size, w_step)
        if start_range[-1] < max_pos - w_size:
            # add a tail window, if needed, to cover the entire sequence
            np.append(start_range, max_pos - w_size)
        end_range = start_range + w_size
        w_coords = np.array([start_range, end_range])
        # k & sites ranges
        k_range = np.arange(1, max_k, step_k)
        site_range = np.arange(5, max_sites, step_sites)
        
        # begin calibration
        calibration_result = pd.DataFrame(columns=['rank',
                                                   'taxon',
                                                   'w_start',
                                                   'w_end',
                                                   'K',
                                                   'n_sites',
                                                   'accuracy',
                                                   'precision',
                                                   'recall',
                                                   'F1_score'])

        for start, end in w_coords:
            window = self.loader.get_window(start, end, self.row_thresh, self.col_thresh)
            selector = fsele.Selector(window.cons_mat, window.cons_tax)
            selector.select_taxons(minseqs = self.min_seqs)
            selector.generate_diff_tab()
            for k in k_range:
                for n_sites in site_range:
                    selector.select_sites(n_sites)
                    t_mat, t_tax = selector.get_training_data()
                    # post collapse, generates the final data matrix and taxonomy table
                    x, y = windows.collapse(t_mat, t_tax.reset_index())

                    loo_classif = pd.DataFrame(columns = y.columns)
                    # iterate trough the data
                    idx_gen = loo_generator(x.shape[0])
                    for train_idx, test_idx in idx_gen:
                        query = x[test_idx]
                        train_data = x[train_idx]
                        train_tax = y.iloc[train_idx]
                        results = classification.classify(query, k, train_data, train_tax, dist_mat, q_name=test_idx, mode=mode, support_func=support_func)
                        classif = classification.get_classif(results, mode)
                        loo_classif.at[test_idx] = classif
                    calib_tab = build_cal_tab(loo_classif, y)
                    calib_tab['w_start'] = start
                    calib_tab['w_end'] = end
                    calib_tab['K'] = k
                    calib_tab['n sites'] = n_sites
                    calibration_result = pd.concat([calibration_result, calib_tab], ignore_index=True)
        
        return calibration_result

#%% test
taxon = 'nematoda'
marker = '18s'
in_dir = '/home/hernan/PROYECTOS/nematoda_18s/out_dir'
out_dir = '/home/hernan/PROYECTOS/nematoda_18s/out_dir'
tmp_dir = '/home/hernan/PROYECTOS/nematoda_18s/tmp_dir'
warn_dir = '/home/hernan/PROYECTOS/nematoda_18s/warn_dir'
#%%
loader = windows.WindowLoader(taxon, marker, in_dir, out_dir, tmp_dir, warn_dir)
# def loo_calibrate(self, w_size, w_step, max_k, step_k, max_sites, step_sites, dist_mat, mode='majority', support_func=classification.wknn):
w_size = 200
w_step = 15
max_k = 15
step_k = 2
max_sites = 30
step_sites = 5
dist_mat = cost_matrix.cost_matrix()
mode = 'majority'
support_func=classification.wknn
row_thresh = 0.2
col_thresh = 0.2
min_seqs = 10

# set up parameter ranges
# window coordinates
max_pos = loader.dims[1]
start_range = np.arange(0, max_pos - w_size, w_step)
if start_range[-1] < max_pos - w_size:
    # add a tail window, if needed, to cover the entire sequence
    start_range = np.append(start_range, max_pos - w_size)
end_range = start_range + w_size
w_coords = np.array([start_range, end_range]).T
# k & sites ranges
k_range = np.arange(1, max_k, step_k)
site_range = np.arange(5, max_sites, step_sites)

# begin calibration
calibration_result = pd.DataFrame(columns=['rank',
                                           'taxon',
                                           'w_start',
                                           'w_end',
                                           'K',
                                           'n_sites',
                                           'accuracy',
                                           'precision',
                                           'recall',
                                           'F1_score'])

for start, end in w_coords[15:16] :
    print(f'Window {start} - {end}')
    window = loader.get_window(start, end, row_thresh, col_thresh)
    selector = fsele.Selector(window.cons_mat, window.cons_tax)
    selector.select_taxons(minseqs = min_seqs)
    selector.generate_diff_tab()
    for k in k_range:
        print(f'\t{k} neighbours')
        for n_sites in site_range:
            print(f'\t\t{n_sites} sites')
            selector.select_sites(n_sites)
            t_mat, t_tax = selector.get_training_data('family')
            # post collapse, generates the final data matrix and taxonomy table
            x, y = windows.collapse(t_mat, t_tax.reset_index())

            loo_classif = pd.DataFrame(columns = y.columns)
            # iterate trough the data
            idx_gen = loo_generator(x.shape[0])
            for train_idx, test_idx in idx_gen:
                query = x[test_idx]
                train_data = x[train_idx]
                train_tax = y.iloc[train_idx]
                results = classification.classify(query, k, train_data, train_tax, dist_mat, q_name=test_idx, mode=mode, support_func=support_func)
                classif = classification.get_classif(results, mode)
                loo_classif.at[test_idx] = classif
            calib_tab = build_cal_tab(loo_classif, y)
            calib_tab['w_start'] = start
            calib_tab['w_end'] = end
            calib_tab['K'] = k
            calib_tab['n sites'] = n_sites
            calibration_result = pd.concat([calibration_result, calib_tab], ignore_index=True)
