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
import concurrent.futures
import json
import logging
import numba as nb
import numpy as np
import os
import pandas as pd
import time
from classification import classification
from classification import cost_matrix
from DATA import DATA
from preprocess import feature_selection as fsele
from preprocess import windows

#%% set
logger = logging.getLogger('Graboid.calibrator')
logger.setLevel(logging.DEBUG)

#%% functions
def make_dirs(base_dir):
    os.makedirs(f'{base_dir}/data', exist_ok=bool)
    os.makedirs(f'{base_dir}/warnings', exist_ok=bool)

def get_metrics(confusion, taxons):
    # get calibration metrics for a given confusion matrix
    # confusion: confusion matrix
    # taxons: taxon index (generated by build_confusion)

    results = []
    for idx, tax in enumerate(taxons):
        # get true/false positives/negatives for each taxon
        tp = confusion[idx, idx]
        tn = np.delete(np.delete(confusion, idx, axis=0), idx, axis=1).sum()
        fp = np.delete(confusion[idx], idx).sum()
        fn = np.delete(confusion[:, idx], idx).sum()
        
        # calculate metrics (accuracy, precision, recall and F1)
        acc = (tp + tn) / (tp + tn + fp + fn)
        prc = 0
        rec = 0
        f1 = 0
        if tp > 0:
            # if there are no positive values, prec, rec and f1 are 0 by default
            prc = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1 = (2 * prc * rec)/(prc + rec)
        
        results.append([tax, acc, prc, rec, f1])
    return results

def loo_generator(nrecs):
    # generates the indexes for the training dataset and the testing instance in leave-one-out calibration
    record_idxs = np.arange(nrecs)
    for idx in record_idxs:
        train_idx = np.delete(record_idxs, idx)
        test_idx = idx
        yield train_idx, test_idx

@nb.njit
def build_confusion(pred, real):
    # build the confusion matrix (for a given RANK)
    # pred: array of PREDICTED taxon for each training sample
    # pred: array of REAL taxon for each training sample

    # list & count taxes
    uniq_taxes = np.unique(real)
    n_taxes = len(uniq_taxes)
    
    confusion = np.zeros((n_taxes, n_taxes), dtype=np.int32)
    # rows: REAL taxons
    # columns: PREDICTED taxons
    
    for idx_0, tax_0 in enumerate(uniq_taxes):
        tax_pred = pred[real == tax_0]
        for idx_1, tax_1 in enumerate(uniq_taxes):
            pred_as = len(tax_pred[tax_pred == tax_1])
            confusion[idx_0, idx_1] = pred_as
    return confusion, uniq_taxes
    
def build_cal_tab(pred_tax, real_tax, n_ranks=6):
    # build the calibration table from the given results
    # n_ranks 6 by default (phylum, class, order, family, genus, species), could be modifyed to include less/more (should be done automatically)
    
    results = []
    # remove first(query name) and last (K) columns from predicted taxons
    pred_cropped = pred_tax[:,1:-1].T
    real_mat = real_tax.to_numpy().T
    ranks = real_tax.columns
    
    for rank, pred, real in zip(ranks, pred_cropped, real_mat):
        # build a confusion matrix and get results from there, update results table
        rank_confusion, taxons = build_confusion(pred, real)
        rank_metrics = get_metrics(rank_confusion, taxons)
        rank_metrics = np.insert(rank_metrics, 0, rank, axis=1)
        results.append(rank_metrics)
    return np.concatenate(results)

    for rank in np.arange(n_ranks):
        # build a confusion matrix and get results from there, update results table
        rank_confusion, taxons = build_confusion(pred_cropped[:,rank], real_tax.iloc[:,rank].to_numpy())
        rank_metrics = get_metrics(rank_confusion, taxons)
        rank_metrics = np.insert(rank_metrics, 0, rank, axis=1)
        results.append(rank_metrics)
    return np.concatenate(results)

#%% classes
class Calibrator:
    def __init__(self, out_dir, warn_dir, prefix='calibration'):
        self.out_dir = out_dir
        self.warn_dir = warn_dir
        
        # prepare out files
        self.out_file = self.out_dir + f'/{prefix}.report'
        self.classif_file = self.out_dir + f'/{prefix}.classif'
        self.meta_file = self.out_dir + f'/{prefix}.meta'
        
        self.selector = fsele.Selector(out_dir)
        self.loader = None
        
    @property
    def dist_mat(self):
        return self.__dist_mat
    @dist_mat.setter
    def dist_mat(self, mat_code):
        try:
            self.__dist_mat = cost_matrix(mat_code)
        except:
            raise
            
    def set_database(self, database):
        if not database in DATA.DBASES:
            print(f'Database {database} not found.')
            print('Current databases include:')
            for db, desc in DATA.DBASE_LIST.items():
                print(f'\tDatabase: {db} \t:\t{desc}')
            raise Exception('Database not found')
        self.db = database
        self.db_dir = DATA.DATAPATH + '/' + database
        # use meta file from database to locate necessary files
        with open(self.db_dir + '/meta.json', 'r') as meta_handle:
            db_meta = json.load(meta_handle)
        mat_file = db_meta['mat_file']
        tax_file = db_meta['tax_file']
        acc_file = db_meta['acc_file']
        order_file = db_meta['order_file']
        diff_file = db_meta['diff_file']
        
        # set the loader with the learning data
        self.loader = windows.WindowLoader('Graboid.calibrator.windowloader')
        self.loader.set_files(mat_file, acc_file, tax_file)
        # load information files
        self.selector.load_order_mat(order_file)
        self.selector.load_diff_tab(diff_file)
    
    def set_windows(self, size=np.inf, step=np.inf, starts=0, ends=np.inf):
        # this function establishes the windows to be used in the grid search
        # size and step establish the length and displacement rate of the sliding window
            # default values use the entire sequence (defined by w_start & w_end) in a single run
        # start and end define the scope(s) to analize
        # multiple values of starts & ends allow for calibration on multiple separated windows
            # default values use the entire sequence
        
        # prepare starts & ends
        starts = list(starts)
        ends = list(ends)
        if len(starts) != len(ends):
            raise Exception(f'Given starts and ends lengths do not match: {len(starts)} starts, {len(ends)} ends')
        # establish the scope
        max_pos = self.loader.dims[1]
        w_coords = []
        w_info = pd.DataFrame(columns='start end size step'.split())
        for w_idx, (start, end) in enumerate(zip(starts, ends)):
            w_start = max(0, start)
            w_end = min(end, max_pos)
            scope_len = w_end - w_start
            # define the windows
            w_size = min(size, scope_len)
            w_step = min(step, scope_len)
            start_range = np.arange(w_start, w_end, w_step)
            if start_range[-1] < max_pos - w_size:
                # add a tail window, if needed, to cover the entire sequence
                start_range = np.append(start_range, max_pos - w_size)
            end_range = start_range + w_size
            
            w_coords.append(pd.DataFrame({'start':start_range, 'end':end_range}, index = [w_idx for i in start_range]))
            w_info.at[f'w_{w_idx}'] = [w_start, w_end, w_size, w_step]
        self.w_coords = pd.concat(w_coords)
        self.w_info = w_info
    
    def grid_search(self,
                    max_k,
                    step_k,
                    max_n,
                    step_n,
                    min_seqs=10,
                    rank='genus',
                    row_thresh=0.2,
                    col_thresh=0.2,
                    min_k=1,
                    min_n=5,
                    threads=1,
                    keep_classif=False):
        
        # k & n ranges
        k_range = np.arange(min_k, max_k, step_k)
        n_range = np.arange(min_n, max_n, step_n)
        
        # begin calibration
        for idx, (start, end) in enumerate(self.w_coords.to_numpy()):
            t0 = time.time()
            print(f'Window {start} - {end} ({idx + 1} of {len(self.w_coords)})')
            # extract window and select atributes
            window = self.loader.get_window(start, end, row_thresh, col_thresh)
            if len(window.eff_mat) == 0:
                # no effective sequences in the window
                continue
            n_seqs = window.eff_mat.shape[0]
            if n_seqs < min_seqs:
                # not enough sequences passed the filter, skip iteration
                print(f'Window {start} - {end}. Not enoug sequences to perform calibration ({n_seqs}, min = {min_seqs}), skipping')
                continue
            
            n_sites = self.selector.get_sites(n_range, rank, window.cols)
            y = window.eff_tax
            # distance container, 3d array, paired distance matrix for every value of n
            dist_mat = np.zeros((n_seqs, n_seqs, len(n_range)), dtype=np.float32)
            # get paired distances
            t1 = time.time()
            logger.debug(f'prep time {t1 - t0}')
            for idx_0 in np.arange(n_seqs - 1):
                qry_seq = window.eff_mat[[idx_0]]
                idx_1 = idx_0 + 1
                ref_seqs = window.eff_mat[idx_1:]
                # persistent distance array, updates with each value of n
                dists = np.zeros((1, ref_seqs.shape[0]), dtype=np.float32)
                for n_idx, sites in enumerate(n_sites.values()):
                    sub_qry = qry_seq[:, sites]
                    sub_ref = ref_seqs[:, sites]
                    dists += classification.get_dists(sub_qry, sub_ref, self.cost_mat).reshape(1, -1)
                    dist_mat[idx_0, idx_1:, n_idx] = dists
                    dist_mat[idx_1:, idx_0, n_idx] = dists # is this necessary? (yes), allows sortying of distances in a single step
            # fill the diagonal values with infinite value, this ensures they are never amongst the k neighs
            for i in range(n_range): np.fill_diagonal(dist_mat[:,:,i], np.inf)
            t2 = time.time()
            logger.debug(f'dist calculation {t2 - t1}')
            # get ordered_neighbours and sorted distances
            neighbours = np.argsort(dist_mat, axis=1)
            ordered_dists = [dist_mat[np.tile(np.arange(n_seqs), (n_seqs, 1)).T, neighbours[...,n], n] for n in range(neighbours.shape[2])]
            
            guide = [(n, mode, classif) for mode, classif in classification.classif_funcs_nb.items() for n in range(len(n_range))]
            classif_report = []
            # use multiprocessing to speed up classification
            t3 = time.time()
            with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
                # since we're using numba functions, y must be cast as a numpy array
                future_classifs = {executor.submit(classifier, neighbours[...,n], ordered_dists[n], y.to_numpy(), k_range):(mode,n) for (n, mode, classifier) in guide}
                for future in concurrent.futures.as_completed(future_classifs):
                    pre_classif, columns = future.result()
                    mode, n = future_classifs[future]
                    classif = classification.get_classif(pre_classif, classification.classif_modes[mode])
                    mode_report = pd.DataFrame(classif, columns=columns)
                    mode_report['mode'] = mode
                    mode_report['n'] = n_range[n]
                    classif_report.append(mode_report)
            classif_report = pd.concat(classif_report)
            t4 = time.time()
            logger.debug(f'classification {t4 - t3}')
            # store intermediate classification results (if enabled)
            if keep_classif:
                classif_report['w_start'] = start
                classif_report['w_end'] = end
                classif_report.to_csv(self.classif_file, header=os.path.isfile(self.classif_file), index=False, mode='a')
            # get classification metrics
            t5 = time.time()
            for (k, n, mode), subtab in classif_report.groupby(['_k', 'n', 'mode']):
                for rk, rk_subtab in subtab.groupby('rk'):
                    pred = rk_subtab.tax.values
                    real = y.loc[rk_subtab.idx.values].iloc[:,int(rk)].values
                    confusion, taxons = build_confusion(pred, real)
                    metrics = get_metrics(confusion, taxons)
                    metrics_report = pd.DataFrame(metrics, columns=['Taxon', 'Accuracy', 'Precision', 'Recall', 'F1_score'])
                    metrics_report['w_start'] = start
                    metrics_report['w_end'] = end
                    metrics_report['rank'] = rk
                    metrics_report['n_sites'] = n
                    metrics_report['K'] = k
                    metrics_report['mode'] = classification.classif_longnames[mode]
                    
                    metrics_report.to_csv(self.out_file, header=os.path.isfile(self.classif_file), index=False, mode='a')
            t6 = time.time()
            logger.debug(f'metric calculation {t6 - t5}')
            
            # register report metadata
            meta = {'k':k_range,
                    'n':n_range,
                    'db': self.db,
                    'windows':self.w_info.T.to_dict()}
            with open(self.meta_file, 'w') as meta_handle:
                json.dump(meta, meta_handle)
