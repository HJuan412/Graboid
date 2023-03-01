#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 11:06:10 2021

@author: hernan
"""

#%% libraries
import concurrent.futures
import json
import logging
import numba as nb
import numpy as np
import os
import pandas as pd
import time

from calibration import reporter as rp
from ..classification import classification
from ..classification import cost_matrix
from ..DATA import DATA
from ..preprocess import feature_selection as fsele
from ..preprocess import windows

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

def get_mem_magnitude(size):
    order = np.log2(size)
    # change order of magnitude at 750 b, KiB, MiB
    if order < 9.584962500721156:
        return size, 'b'
    if order < 19.584962500721158:
        return size / 1024, 'KiB'
    if order < 29.584962500721158:
        return size / 1048576, 'MiB'
    return size / 1073741824, 'GiB'

def get_classif_mem(classif_dir):
    file_list = os.listdir(classif_dir)
    mem = 0
    for file in file_list:
        mem += os.path.getsize(classif_dir + '/' + file)
    return get_mem_magnitude(mem)

def build_report(classification, taxonomy_matrix, w_start, w_end, out_file=None, log=False):
    # setup report logger
    logname = ''
    if log:
        logname = 'Graboid.calibrator.report'
    report_logger = logging.getLogger(logname)
    # classification is a pandas dataframe containing the classification results, index must be a multiindex with levels ['_k', 'n', 'mode']
    # taxonomy matrix is a numpy array with rows equal to the real taxonomies for the 
    
    # build rows & param guides
    # param guides keys are the ranks visited during calibration
    # each value is a list of tuples containing the parameter combination and the corresponding rows
    # this is used to retrieve the correct rows from each confusion matrix
    # looks like -> param_guides[rk] = [(params0, rows0), (param1, rows1),...]
    t01 = time.time()
    param_guides = {}
    for rk, rk_subtab in classification.groupby('rk'):
        param_guides[rk] = []
        aux_tab = pd.Series(index=rk_subtab.index, data=np.arange(len(rk_subtab)))
        for params, aux_subtab in aux_tab.groupby(level=[0,1,2]):
            param_guides[rk].append((params, aux_subtab.values))
    t02 = time.time()
    report_logger.debug(f'Report prep time {t02 - t01:.3f}')
    # get rank, index, predicted and real data from the classification table
    # index contains the query sequence position in the taxonomy matrix
    rk_data = classification.rk.astype(int).to_numpy()
    idx_data = classification.idx.astype(int).to_numpy()
    pred_data = classification.tax.astype(int).to_numpy()
    real_data = taxonomy_matrix[idx_data, rk_data]
    t03 = time.time()
    report_logger.debug(f'Report data gathered {t03 - t02:.3f}')
    # begin construction
    # report columns: Taxon Rank K n_sites mode Accuracy Precision Recall F1_score
    pre_report = [] # stores completed subreports
    # generate report by taxonomic rank
    for rk in np.unique(rk_data):
        t04 = time.time()
        # access the guide for the current rank
        guide = param_guides[rk]
        # locate rows in classification belonging to the current rank
        rk_idx = np.argwhere(rk_data == rk).flatten()
        # get predicted and actual values
        pred_submat = pred_data[rk_idx]
        real_submat = real_data[rk_idx]
        # get a list of unique taxons in the current rank
        rk_tax = np.unique(taxonomy_matrix[:,rk])
        mem, units = get_mem_magnitude(len(rk_idx) * 4)
        report_logger.debug(f'Rank {rk} confusion matrix of shape ({len(rk_idx)}, 4) and size {mem:.3f} {units}')
        # build confusion matrix per taxon in rank, generate metrics and move on
        for idx, tax in enumerate(rk_tax):
            tax_true = real_submat == tax
            tax_pred = pred_submat == tax
            
            tp = tax_true & tax_pred # true positives
            tn = ~tax_true & ~tax_pred # true negatives
            fn = tax_true & ~tax_pred # false negatives
            fp = ~tax_true & tax_pred # false positives
            confusion = np.array([tp, tn, fn, fp]).T
        
            # get the metrics for each parameter combination
            for params, rows in guide:
                # count occurrences of each case (true/false positive/negative) and calculate metrics
                sum_confusion = confusion[rows].sum(axis=0)
                # clip function used in metrics calculation to handle cases in which divisor is 0
                acc = (sum_confusion[0] + sum_confusion[1]) / (sum_confusion.sum(axis=0))
                prc = sum_confusion[0] / np.clip((sum_confusion[0] + sum_confusion[3]), a_min=np.e**-7, a_max=None)
                rec = sum_confusion[0] / np.clip((sum_confusion[0] + sum_confusion[2]), a_min=np.e**-7, a_max=None)
                f1 = (2 * prc * rec)/np.clip((prc + rec), np.e**-7, a_max=None)
                # build subreport, add rank and parameter data
                pre_subreport = [tax, rk, w_start, w_end, params[0], params[1], params[2], acc, prc, rec, f1]
                pre_report.append(pre_subreport)
        t05 = time.time()
        report_logger.debug(f'Rank {rk} calibration {t05 - t04:.3f}')
    t07 = time.time()
    report_logger.debug(f'Calibration complete {t07 - t03:.3f}')
    report = pd.DataFrame(pre_report, columns = 'Taxon Rank w_start w_end K n_sites mode Accuracy Precision Recall F1_score'.split())
    if out_file is None:
        return report
    report.to_csv(out_file, header = not os.path.isfile(out_file), index = False, mode = 'a')
    return
#%% classes
class Calibrator:
    def __init__(self, out_dir, warn_dir, prefix='calibration'):
        self.out_dir = out_dir
        self.classif_dir = out_dir + '/classification'
        self.warn_dir = warn_dir
        
        # make a directory to store classification reports
        os.mkdir(self.classif_dir)
        # prepare out files
        self.report_file = self.out_dir + f'/{prefix}.report'
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
            self.__dist_mat = cost_matrix.get_matrix(mat_code)
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
        self.guide_file = db_meta['guide_file'] # this isn't used in the calibration process, used in result visualization
        
        # set the loader with the learning data
        self.loader = windows.WindowLoader('Graboid.calibrator.windowloader')
        self.loader.set_files(mat_file, acc_file, tax_file)
        self.max_pos = self.loader.dims[1]
        # load information files
        self.selector.load_order_mat(order_file)
        self.selector.load_diff_tab(diff_file)
        
        logger.info(f'Set database: {database}')
    
    def set_windows(self, size=np.inf, step=np.inf, starts=[0], ends=[np.inf]):
        # this function establishes the windows to be used in the grid search
        # size and step establish the length and displacement rate of the sliding window
            # default values use the entire sequence (defined by w_start & w_end) in a single run
        # start and end define the scope(s) to analize
        # multiple values of starts & ends allow for calibration on multiple separated windows
            # default values use the entire sequence
        
        starts = list(starts)
        ends = list(ends)
        if len(starts) != len(ends):
            raise Exception(f'Given starts and ends lengths do not match: {len(starts)} starts, {len(ends)} ends')
        raw_coords = np.array([starts, ends]).T
        # clip coordinates outside of boundaries and detect the ones that are flipped
        clipped = np.clip(raw_coords, 0, self.max_pos).astype(int)
        flipped = clipped[:,0] >= clipped[:,1]
        for flp in raw_coords[flipped]:
            logger.warning(f'Column {flp} is not valid')
        windows = clipped[~flipped]
        # establish the scope
        w_tab = []
        w_info = pd.DataFrame(columns='start end size step'.split())
        for w_idx, (w_start, w_end) in enumerate(windows):
            scope_len = w_end - w_start
            if scope_len < size:
                # do a single window
                w_coords = np.array([[w_start, w_end]])
                w_size = scope_len
            else:
                w_step = min(step, size)
                w_size = size
                start_range = np.arange(w_start, w_end, w_step)
                end_range = start_range + w_size
                w_coords = np.array([start_range, end_range]).T
                # clip windows
                w_coords = w_coords[end_range <= w_end]
                if w_coords[-1, 1].T > w_end:
                    # add a tail window, if needed, to cover the entire sequence
                    w_coords = np.append(w_coords, [[w_end - size, w_end]], axis=0)
            
            w_tab.append(pd.DataFrame(w_coords, columns = 'start end'.split(), index = [w_idx for i in w_coords]))
            w_info.at[f'w_{w_idx}'] = [w_start, w_end, w_size, w_step]
        self.w_coords = pd.concat(w_tab)
        self.w_info = w_info
        logger.info(f'Set {len(w_coords)} windows of size {size} and step {step}')
    
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
                    keep_classif=False,
                    log_report=False):
        
        # k & n ranges
        k_range = np.arange(min_k, max_k, step_k)
        n_range = np.arange(min_n, max_n, step_n)
        
        # register report metadata
        meta = {'k':k_range.tolist(),
                'n':n_range.tolist(),
                'db':self.db,
                'guide': self.guide_file,
                'windows':self.w_info.T.to_dict()}
        with open(self.meta_file, 'w') as meta_handle:
            json.dump(meta, meta_handle)
            
        # begin calibration
        logger.info('Began calibration')
        t00 = time.time()
        for idx, (start, end) in enumerate(self.w_coords.to_numpy()):
            t0 = time.time()
            print(f'Window {start} - {end} ({idx + 1} of {len(self.w_coords)})')
            # extract window and select atributes
            window = self.loader.get_window(start, end, row_thresh, col_thresh)
            if len(window.eff_mat) == 0:
                # no effective sequences in the window
                continue
            n_seqs = window.n_seqs
            if n_seqs < min_seqs:
                # not enough sequences passed the filter, skip iteration
                logger.info(f'Window {start} - {end}. Not enoug sequences to perform calibration ({n_seqs}, min = {min_seqs}), skipping')
                continue
            
            n_sites = self.selector.get_sites(n_range, rank, window.cols)
            y = window.eff_tax
            # distance container, 3d array, paired distance matrix for every value of n
            dist_mat = np.zeros((n_seqs, n_seqs, len(n_range)), dtype=np.float32)
            # get paired distances
            t1 = time.time()
            logger.debug(f'prep time {t1 - t0:.3f}')
            for idx_0 in np.arange(n_seqs - 1):
                qry_seq = window.eff_mat[[idx_0]]
                idx_1 = idx_0 + 1
                ref_seqs = window.eff_mat[idx_1:]
                # persistent distance array, updates with each value of n
                dists = np.zeros((1, ref_seqs.shape[0]), dtype=np.float32)
                for n_idx, n in enumerate(n_range):
                    try:
                        sites = n_sites[n]
                        sub_qry = qry_seq[:, sites]
                        sub_ref = ref_seqs[:, sites]
                        dists += classification.get_dists(sub_qry, sub_ref, self.dist_mat).reshape(1, -1)
                    except KeyError:
                        # no new sites for n
                        pass
                    dist_mat[idx_0, idx_1:, n_idx] = dists
                    dist_mat[idx_1:, idx_0, n_idx] = dists # is this necessary? (yes), allows sortying of distances in a single step
            # fill the diagonal values with infinite value, this ensures they are never amongst the k neighs
            for i in range(len(n_range)): np.fill_diagonal(dist_mat[:,:,i], np.inf)
            t2 = time.time()
            logger.debug(f'dist calculation {t2 - t1:.3f}')
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
            logger.debug(f'classification {t4 - t3:.3f}')
            # store intermediate classification results (if enabled)
            if keep_classif:
                classif_file = self.classif_dir + f'/{start}-{end}_{n_seqs}.classif'
                classif_report.to_csv(classif_file, index=False)
                # store table with real values as well
                uniq_idxs = classif_report.idx.sort_values().unique()
                y.loc[uniq_idxs].to_csv(self.classif_dir + f'/{start}-{end}_{n_seqs}.real')
            # get classification metrics
            t5 = time.time()
            classification_table = classif_report.set_index(['_k', 'n', 'mode'])
            tax_matrix = y.loc[uniq_idxs].to_numpy()
            build_report(classification_table, tax_matrix, start, end, self.report_file, log_report)
            #
            t6 = time.time()
            logger.debug(f'metric calculation {t6 - t5:.2f}')
            logger.info(f'Window {start} - {end} ({n_seqs} effective sequences) Calibrated in {t6 - t0:.2f} seconds')
        elapsed = time.time() - t00
        logger.info(f'Finished calibration in {elapsed:.2f} seconds')
        logger.info(f'Stored calibration report to {self.report_file}')
        if keep_classif:
            mem, unit = get_classif_mem(self.classif_dir)
            logger.info(f'Stored classification results to {self.classif_dir}, using {mem:.2f} {unit}')
        else:
            os.rmdir(self.classif_dir)
    
    def build_summaries(self):
        # run this function to build the summary files after running the grid search
        print('Building summary files')
        cal_report = pd.read_csv(self.report_file)
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1_score']:
            print(f'Building {metric} summary')
            score_file = f'{self.out_dir}/{metric}_score.summ'
            param_file = f'{self.out_dir}/{metric}_param.summ'
            rk_tab, param_tab = rp.make_summ_tab(cal_report, metric)
            rk_tab.to_csv(score_file)
            param_tab.to_csv(param_file)
            logger.info(f'Stored score summary to {score_file}')
            logger.info(f'Stored param summary to {score_file}')
