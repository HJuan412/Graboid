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
from preprocess import preproc
from preprocess import windows

#%% functions
def get_metrics(confusion):
    taxons = confusion.index.tolist()
    idxs = np.arange(len(taxons))
    metrics = pd.DataFrame(index=idxs, columns=['taxon', 'accuracy', 'precision', 'recall', 'F1_score'])
    
    conf_mat = confusion.to_numpy()
    for idx, tax in enumerate(taxons):
        complement = np.delete(idxs, idx)

        tp = conf_mat[idx, idx].sum()
        tn = conf_mat[complement, complement].sum()
        fp = conf_mat[complement, idx].sum()
        fn = conf_mat[idx, complement].sum()
        
        acc = 0
        prc = 0
        rec = 0
        f1 = 0
        sum_check = tp + fn != 0 # make sure metric calculation is possible
        if sum_check:
            acc = (tp + tn) / (tp + tn + fp + fn)
            prc = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1 = (2 * prc * rec)/(prc + rec)
        
        tax_metrics = pd.Series({'taxon':tax,
                                 'accuracy':acc,
                                 'precision':prc,
                                 'recall':rec,
                                 'F1_score':f1})
        # metrics = metrics.append(tax_metrics, ignore_index=True)
        metrics.at[idx] = tax_metrics
    
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
        if p in uniq_taxes:
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

        for idx, (start, end) in enumerate(w_coords):
            print(f'Window {start} - {end}')
            window = self.loader.get_window(start, end, self.row_thresh, self.col_thresh)
            if len(window.cons_mat) == 0:
                continue
            selector = fsele.Selector(window.cons_mat, window.cons_tax)
            selector.select_taxons(minseqs = self.min_seqs)
            selector.generate_diff_tab()
            for k in k_range:
                print(f'\t{k} neighbours')
                for n_sites in site_range:
                    print(f'\t\t{n_sites} sites')
                    # post collapse, generates the final data matrix and taxonomy table
                    x,y, super_c = preproc.preprocess(selector, self.row_thresh, self.col_thresh, minseqs=min_seqs, nsites=n_sites)

                    loo_classif = pd.DataFrame(columns = y.columns)
                    # iterate trough the data
                    if x.shape[1] == 0:
                        continue
                    idx_gen = loo_generator(x.shape[0])
                    
                    for train_idx, test_idx in idx_gen:
                        query = x[test_idx]
                        train_data = x[train_idx]
                        train_tax = y.iloc[train_idx]
                        # return x, query, k, train_data, train_tax, dist_mat, test_idx, mode, support_func
                        results = classification.classify(query, k, train_data, train_tax, dist_mat, q_name=test_idx, mode=mode, support_func=support_func)
                        classif = classification.get_classif(results, mode)
                        loo_classif.at[test_idx] = classif
                    calib_tab = build_cal_tab(loo_classif, y)
                    calib_tab['w_start'] = start
                    calib_tab['w_end'] = end
                    calib_tab['K'] = k
                    calib_tab['n_sites'] = n_sites
                    calibration_result = pd.concat([calibration_result, calib_tab], ignore_index=True)
        return calibration_result

    def loo_calibrate2(self, w_size, w_step, max_k, step_k, max_sites, step_sites, dist_mat, mode='majority', support_func=classification.wknn):
        # sped up version:
            # make n_sites the outer loop : collapse calls go from len(range(nsites)) * len(range(k)) to just len(range(nsites))
            # for each inst, calculate distances ONCE and predict for all k's in a single pass
        
        # set up parameter ranges
        # window coordinates
        max_pos = self.loader.dims[1]
        start_range = np.arange(0, max_pos - w_size, w_step)
        if start_range[-1] < max_pos - w_size:
            # add a tail window, if needed, to cover the entire sequence
            np.append(start_range, max_pos - w_size)
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

        for idx, (start, end) in enumerate(w_coords):
            print(f'Window {start} - {end}')
            window = self.loader.get_window(start, end, self.row_thresh, self.col_thresh)
            if len(window.cons_mat) == 0:
                continue
            selector = fsele.Selector(window.cons_mat, window.cons_tax)
            selector.select_taxons(minseqs = self.min_seqs)
            selector.generate_diff_tab()
            
            # flip loops (n_sites is the outer loop)
            for n_sites in site_range:
                # post collapse, generates the final data matrix and taxonomy table
                x,y, super_c = preproc.preprocess(selector, self.row_thresh, self.col_thresh, minseqs=min_seqs, nsites=n_sites)
                print(f'\t{n_sites} sites')
                loo_classif = pd.DataFrame(columns = y.columns)
                loo_classif['K'] = 0
                # iterate trough the data
                if x.shape[1] == 0:
                    continue
                
                idx_gen = loo_generator(x.shape[0])
                row_idx = 0
                n_seqs = x.shape[0]
                quintile = int(n_seqs / 5)
                for train_idx, test_idx in idx_gen:
                    if test_idx % quintile == 0:
                        print(f'\t\tCalibrated {test_idx} of {n_seqs} ({(test_idx/n_seqs) * 100:.2f}%)')
                    query = x[test_idx]
                    train_data = x[train_idx]
                    train_tax = y.iloc[train_idx]
                    results = classification.calibration_classify(query, k_range, train_data, train_tax, dist_mat, q_name=test_idx, mode=mode, support_func=support_func)
                    
                    for k, res in zip(k_range, results):
                        classif = classification.get_classif(res, mode)
                        classif['K'] = k
                        loo_classif.at[row_idx] = classif
                        row_idx += 1
                
                for k, subtab in loo_classif.groupby('K'):
                    calib_tab = build_cal_tab(subtab, y)
                    calib_tab['w_start'] = start
                    calib_tab['w_end'] = end
                    calib_tab['K'] = k
                    calib_tab['n_sites'] = n_sites
                    calibration_result = pd.concat([calibration_result, calib_tab], ignore_index=True)
        return calibration_result
    
    def loo_calibrate3(self, w_size, w_step, max_k, step_k, max_sites, step_sites, dist_mat):
        # sped up version:
            # make n_sites the outer loop : collapse calls go from len(range(nsites)) * len(range(k)) to just len(range(nsites))
            # for each inst, calculate distances ONCE and predict for all k's in a single pass
        
        # this version simultaneously calibrates for all 3 methods (majority, weighted + wknn, weighted + dwknn)
        # set up parameter ranges
        # window coordinates
        max_pos = self.loader.dims[1]
        start_range = np.arange(0, max_pos - w_size, w_step)
        if start_range[-1] < max_pos - w_size:
            # add a tail window, if needed, to cover the entire sequence
            np.append(start_range, max_pos - w_size)
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
                                                   'F1_score',
                                                   'mode']) # do all three modes in a single table
        # calibration_maj = calibration_result.copy()
        # calibration_wknn = calibration_result.copy()
        # calibration_dwknn = calibration_result.copy()

        for idx, (start, end) in enumerate(w_coords):
            print(f'Window {start} - {end}')
            window = self.loader.get_window(start, end, self.row_thresh, self.col_thresh)
            if len(window.cons_mat) == 0:
                continue
            selector = fsele.Selector(window.cons_mat, window.cons_tax)
            selector.select_taxons(minseqs = self.min_seqs)
            selector.generate_diff_tab()
            
            # flip loops (n_sites is the outer loop)
            for n_sites in site_range:
                # post collapse, generates the final data matrix and taxonomy table
                x,y, super_c = preproc.preprocess(selector, self.row_thresh, self.col_thresh, minseqs=min_seqs, nsites=n_sites)
                print(f'\t{n_sites} sites')
                loo_classif = pd.DataFrame(columns = y.columns)
                loo_classif['K'] = 0
                
                loo_maj = loo_classif.copy()
                loo_wknn = loo_classif.copy()
                loo_dwknn = loo_classif.copy()
                # iterate trough the data
                if x.shape[1] == 0:
                    continue
                
                idx_gen = loo_generator(x.shape[0])
                row_idx = 0
                n_seqs = x.shape[0]
                quintile = int(n_seqs / 5)
                for train_idx, test_idx in idx_gen:
                    if len(train_idx) == 0:
                        continue
                    if test_idx % quintile == 0:
                        print(f'\t\tCalibrated {test_idx} of {n_seqs} ({(test_idx/n_seqs) * 100:.2f}%)')
                    query = x[test_idx]
                    train_data = x[train_idx]
                    train_tax = y.iloc[train_idx]

                    results_maj = classification.calibration_classify(query, k_range, train_data, train_tax, dist_mat, q_name=test_idx, mode='majority', support_func=None)
                    results_wknn = classification.calibration_classify(query, k_range, train_data, train_tax, dist_mat, q_name=test_idx, mode='weighted', support_func=classification.wknn)
                    results_dwknn = classification.calibration_classify(query, k_range, train_data, train_tax, dist_mat, q_name=test_idx, mode='weighted', support_func=classification.dwknn)

                    for res_tab, loo_tab, mode in [(results_maj, loo_maj, 'majority'),
                                                     (results_wknn, loo_wknn, 'weighted'),
                                                     (results_dwknn, loo_dwknn, 'weighted')]:
                        for k, tab in zip(k_range, res_tab):
                            classif = classification.get_classif(tab, mode)
                            classif['K'] = k
                            loo_tab.at[row_idx] = classif
                            row_idx += 1
                
                # for loo_tab, result_tab in [(loo_maj, calibration_maj),
                #                             (loo_wknn, calibration_wknn),
                #                             (loo_dwknn, calibration_dwknn)]:
                for loo_tab, mode in [(loo_maj, 'maj'),
                                      (loo_wknn, 'wknn'),
                                      (loo_dwknn, 'dwknn')]:
                    for k, subtab in loo_tab.groupby('K'):
                        calib_tab = build_cal_tab(subtab, y)
                        calib_tab['w_start'] = start
                        calib_tab['w_end'] = end
                        calib_tab['K'] = k
                        calib_tab['n_sites'] = n_sites
                        calib_tab['mode'] = mode
                        # result_tab = pd.concat([result_tab, calib_tab], ignore_index=True)
                        calibration_result = pd.concat([calibration_result, calib_tab], ignore_index=True)
        
        # return calibration_maj, calibration_wknn, calibration_dwknn
        return calibration_result

#%%

# def loo_calibrate(self, w_size, w_step, max_k, step_k, max_sites, step_sites, dist_mat, mode='majority', support_func=classification.wknn):
w_size = 200
w_step = 15
max_k = 15
step_k = 2
max_sites = 30
step_sites = 5
dist_mat = cost_matrix.cost_matrix()
mode = 'weighted'
# support_func=classification.dwknn
row_thresh = 0.2
col_thresh = 0.2
min_seqs = 10

#%%
base_dir = '/home/hernan/PROYECTOS/Graboid/'
out_dir = '/home/hernan/PROYECTOS/Graboid/calib_test'
import time
def run(taxon, marker):
    in_dir = f'{base_dir}/{taxon}_{marker}/out_dir'
    out_dir = f'{base_dir}/{taxon}_{marker}/out_dir'
    tmp_dir = f'{base_dir}/{taxon}_{marker}/tmp_dir'
    warn_dir = f'{base_dir}/{taxon}_{marker}/warn_dir'

    t0 = time.time()
    cal = Calibrator(taxon, marker, in_dir, out_dir, tmp_dir, warn_dir)
    results = cal.loo_calibrate3(w_size, w_step, max_k, step_k, max_sites, step_sites, dist_mat)
    t1 = time.time()
    return t1 - t0, results

def run2():
    cal = Calibrator(taxon, marker, in_dir, out_dir, tmp_dir, warn_dir)
    results = cal.loo_calibrate2(w_size, w_step, max_k, step_k, max_sites, step_sites, dist_mat)
    return results
# params = run()
#%%
# TODO: delete this cell
if __name__ == '__main__':
    # sped up version:
        # make n_sites the outer loop : collapse calls go from len(range(nsites)) * len(range(k)) to just len(range(nsites))
        # for each inst, calculate distances ONCE and predict for all k's in a single pass
    taxon = 'nematoda'
    marker = '18s'
    in_dir = '/home/hernan/PROYECTOS/nematoda_18s/out_dir'
    out_dir = '/home/hernan/PROYECTOS/nematoda_18s/out_dir'
    tmp_dir = '/home/hernan/PROYECTOS/nematoda_18s/tmp_dir'
    warn_dir = '/home/hernan/PROYECTOS/nematoda_18s/warn_dir'

    loader = windows.WindowLoader(taxon, marker, in_dir, out_dir, tmp_dir, warn_dir)
    # set up parameter ranges
    # window coordinates
    max_pos = loader.dims[1]
    start_range = np.arange(0, max_pos - w_size, w_step)
    if start_range[-1] < max_pos - w_size:
        # add a tail window, if needed, to cover the entire sequence
        np.append(start_range, max_pos - w_size)
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
    calibration_maj = calibration_result.copy()
    calibration_wknn = calibration_result.copy()
    calibration_dwknn = calibration_result.copy()

    for idx, (start, end) in enumerate(w_coords[10:]):
        print(f'Window {start} - {end}')
        window = loader.get_window(start, end, row_thresh, col_thresh)
        if len(window.cons_mat) == 0:
            continue
        selector = fsele.Selector(window.cons_mat, window.cons_tax)
        selector.select_taxons(minseqs = min_seqs)
        selector.generate_diff_tab()
        
        # flip loops (n_sites is the outer loop)
        for n_sites in site_range:
            # post collapse, generates the final data matrix and taxonomy table
            x,y, super_c = preproc.preprocess(selector, row_thresh, col_thresh, minseqs=min_seqs, nsites=n_sites)
            print(f'\t{n_sites} sites')
            loo_classif = pd.DataFrame(columns = y.columns)
            loo_classif['K'] = 0
            
            loo_maj = loo_classif.copy()
            loo_wknn = loo_classif.copy()
            loo_dwknn = loo_classif.copy()
            # iterate trough the data
            if x.shape[1] == 0:
                continue
            
            idx_gen = loo_generator(x.shape[0])
            row_idx = 0
            n_seqs = x.shape[0]
            quintile = int(n_seqs / 5)
            for train_idx, test_idx in idx_gen:
                if test_idx % quintile == 0:
                    print(f'\t\tCalibrated {test_idx} of {n_seqs} ({(test_idx/n_seqs) * 100:.2f}%)')
                query = x[test_idx]
                train_data = x[train_idx]
                train_tax = y.iloc[train_idx]

                results_maj = classification.calibration_classify(query, k_range, train_data, train_tax, dist_mat, q_name=test_idx, mode='majority', support_func=None)
                results_wknn = classification.calibration_classify(query, k_range, train_data, train_tax, dist_mat, q_name=test_idx, mode='weighted', support_func=classification.wknn)
                results_dwknn = classification.calibration_classify(query, k_range, train_data, train_tax, dist_mat, q_name=test_idx, mode='weighted', support_func=classification.dwknn)

                for res_tab, loo_tab, mode in [(results_maj, loo_maj, 'majority'),
                                                 (results_wknn, loo_wknn, 'weighted'),
                                                 (results_dwknn, loo_dwknn, 'weighted')]:
                    for k, tab in zip(k_range, res_tab):
                        classif = classification.get_classif(tab, mode)
                        classif['K'] = k
                        loo_tab.at[row_idx] = classif
                        row_idx += 1
            
            for loo_tab, result_tab in [(loo_maj, calibration_maj),
                                        (loo_wknn, calibration_wknn),
                                        (loo_dwknn, calibration_dwknn)]:
                for k, subtab in loo_tab.groupby('K'):
                    calib_tab = build_cal_tab(subtab, y)
                    calib_tab['w_start'] = start
                    calib_tab['w_end'] = end
                    calib_tab['K'] = k
                    calib_tab['n_sites'] = n_sites
                    result_tab = pd.concat([result_tab, calib_tab], ignore_index=True)
        break
