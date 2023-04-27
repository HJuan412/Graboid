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
import numpy as np
import os
import pandas as pd
import sys
import time
# Graboid libraries
sys.path.append("..") # use this to allow importing from a sibling package
from calibration import cal_dists
from calibration import cal_neighsort
from calibration import cal_classify
from calibration import cal_metrics
from calibration import cal_report
from calibration import cal_plot
from DATA import DATA
from preprocess import feature_selection as fsele
from preprocess import windows as wn

#%% set logger
logger = logging.getLogger('Graboid.calibrator')
logger.setLevel(logging.DEBUG)

#%% functions
def collapse_windows(windows, matrix, tax_tab, row_thresh=0.1, col_thresh=0.1, min_seqs=50, threads=1):
    # collapse the selected windows in the given matrix, apply corresponding filters
    # return:
        # win_indexes : sorted array of collapsed windows indexes
        # win_list : list of Window instances, sorted
        # rej_indexes : sorted array of rejected windows indexes
        # rej_list : list of rejection messages, sorted
    collapsed_windows = {}
    rejected_windows = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        future_windows = {executor.submit(wn.Window, matrix, tax_tab, win[0], win[1], row_thresh, col_thresh, min_seqs):idx for idx, win in enumerate(windows)}
        for future in concurrent.futures.as_completed(future_windows):
            ft_idx = future_windows[future]
            try:
                collapsed_windows[ft_idx] = future.result()
                print(f'Collapsed window {ft_idx}')
            except Exception as excp:
                rejected_windows[ft_idx] = str(excp)
                print(f'Rejected window {ft_idx}')
                continue
    
    # translate the collapsed_windows and rejected_windows dicts into ordered lists
    win_indexes = np.sort(list(collapsed_windows.keys()))
    win_list = [collapsed_windows[idx] for idx in win_indexes]
    
    rej_indexes = np.sort(list(rejected_windows.keys()))
    rej_list = [rejected_windows[idx] for idx in win_indexes]
    
    return win_indexes, win_list, rej_indexes, rej_list

def select_sites(win_list, tax_ext, rank, min_n, max_n, step_n):
    # get the arrays of selected sites for each collapsed window
    window_sites = []
    for win in win_list:
        win_tax = tax_ext.loc[win.taxonomy][[rank]] # trick: if the taxonomy table passed to get_sorted_sites has a single rank column, entropy difference is calculated for said column
        sorted_sites = fsele.get_sorted_sites(win.window, win_tax) # remember that you can use return_general, return_entropy and return_difference to get more information
        window_sites.append(fsele.get_nsites(sorted_sites[0], min_n, max_n, step_n))
    return window_sites

#%% classes
class Calibrator:
    def __init__(self, out_dir=None):
        self.save = False # indicates if a save location is set, if True, calibration reports are stored to files
        if not out_dir is None:
            self.set_outdir(out_dir)
    
    @property
    def window_len(self):
        if hasattr(self, 'windows'):
            return self.windows[:,1] - self.windows[:,0]
        return 0
    
    @property
    def n_windows(self):
        if hasattr(self, 'windows'):
            return len(self.windows)
        return 0
    
    def set_outdir(self, out_dir):
        self.out_dir = out_dir
        self.reports_dir = out_dir + '/reports'
        self.classif_dir = out_dir + '/classification'
        self.warn_dir = out_dir + '/warnings'
        
        try:
            os.mkdir(out_dir)
        except FileExistsError:
            raise Exception(f'Specified output directory "{out_dir}" already exists. Pick a different name or set argmuent "clear" as True')
        self.save = True
        # make a directory to store classification reports
        os.makedirs(self.reports_dir)
        os.makedirs(self.classif_dir)
        os.makedirs(self.warn_dir)
        
    def set_database(self, database):
        self.db = database
        try:
            self.db_dir = DATA.get_database(database)
        except Exception:
            raise
        # use meta file from database to locate necessary files
        with open(self.db_dir + '/meta.json', 'r') as meta_handle:
            db_meta = json.load(meta_handle)
        
        # load taxonomy guides
        self.guide = pd.read_csv(db_meta['guide_file'], index_col=0)
        self.tax_ext = pd.read_csv(db_meta['expguide_file'], index_col=0)
        self.ranks = self.tax_ext.columns.tolist()
        
        # load matrix & accession codes
        map_npz = np.load(db_meta['mat_file'])
        self.matrix = map_npz['matrix']
        self.max_pos = self.matrix.shape[1]
        with open(db_meta['acc_file'], 'r') as handle:
            self.accs = handle.read().splitlines()
        
        # build extended taxonomy
        tax_tab = pd.read_csv(db_meta['tax_file'], index_col=0).loc[self.accs]
        # the tax_tab attribute is the extended taxonomy for each record
        self.tax_tab = self.tax_ext.loc[tax_tab.TaxID.values]
        self.tax_tab.index = tax_tab.index
        
        logger.info(f'Set database: {database}')
        
    def set_sliding_windows(self, size, step):
        if size >= self.max_pos:
            raise Exception(f'Given window size: {size} is equal or greater than the total length of the alignment {self.max_pos}, please use a smaller window size.')
        
        # adjust window size to get uniform distribution (avoid having to use a "tail" window)
        last_position = self.max_pos - size
        n_windows = int(np.ceil(last_position / step))
        w_start = np.linspace(0, last_position, n_windows, dtype=int)
        self.windows = np.array([w_start, w_start + size]).T
        self.w_type = 'sliding'
        logger.info(f'Set {n_windows} windows of size {size} at intervals of {w_start[1] - w_start[0]}')
    
    def set_custom_windows(self, starts, ends):
        # check that given values are valid: same length, starts < ends, within alignment bounds
        try:
            raw_coords = np.array([starts, ends], dtype=np.int).T
            raw_coords = np.reshape(raw_coords, (-1, 2))
        except ValueError:
            raise Exception(f'Given starts and ends lengths do not match: {len(starts)} starts, {len(ends)} ends')
        invalid = raw_coords[:, 0] >= raw_coords[:, 1]
        if invalid.sum() > 0:
            raise Exception(f'At least one pair of coordinates is invalid: {[list(i) for i in raw_coords[invalid]]}')
        out_of_bounds = ((raw_coords < 0) | (raw_coords >= self.max_pos))
        out_of_bounds = out_of_bounds[:,0] | out_of_bounds[:,1]
        if out_of_bounds.sum() > 0:
            raise Exception(f'At least one pair of coordinates is out of bounds [0 {self.max_pos}]: {[list(i) for i in raw_coords[out_of_bounds]]}')
        self.windows = raw_coords
        self.w_type = 'custom'
        logger.info(f'Set {raw_coords.shape[0]} custom windows at positions:')
        for coor_idx, coords in raw_coords:
            logger.info(f'\tWindow {coor_idx}: [{coords[0]} - {coords[1]}] (length {coords[1] - coords[0]})')
    
    # report functions
    def report_windows(self, win_indexes, win_list, rej_indexes, rej_list):
        if not self.save:
            return
        with open(self.reports_dir + '/windows.report', 'w') as handle:
            if len(win_list) > 0:
                handle.write('Collapsed windows:\n\n')
                for w_idx, win in zip(win_indexes, win_list):
                    handle.write(f'Window {w_idx} {self.windows[w_idx]}: collapsed into matrix of shape {win.window.shape}\n')
                handle.write('\n')
            if len(rej_list) > 0:
                handle.write('Rejected windows:\n\n')
                for r_idx, rej in zip(rej_indexes, rej_list):
                    handle.write(f'Window {r_idx} {self.windows[r_idx]}: ' + rej)
    
    def report_sites(self, window_sites, win_list, win_indexes, n_range):
        if not self.save:
            return
        with open(self.reports_dir + '/sites.report', 'w') as handle:
            handle.write('Selected sites:\n')
            for w_idx, win, win_sites in zip(win_indexes, win_list, window_sites):
                handle.write(f'Sites for window {w_idx} {self.windows[w_idx]}:\n')
                for n, n_sites in zip(n_range, win_sites):
                    handle.write(f'\t{n} sites: {n_sites}\n')
    
    def report_metrics(self, report, params, metric):
        # report already has tax_names as the second level of the multiindex
        if not self.save:
            return
        # translate report tax names
        tax_names = [i for i in map(lambda x : x.upper(), report.index.get_level_values(1).tolist())]
        # may need to change type of column headers because csv doesn't like tuples
        # report has a multiindex (rank, tax_name)
        report.to_csv(self.reports_dir + '/{metric}_calibration.csv')
        
        # build params dictionary
        windows = {}
        merged_params = np.concatenate(params, 0)
        for w_idx, (win, param_row) in enumerate(zip(report.columns.tolist(), merged_params.T)):
            win_keys = {i for i in set(param_row) if isinstance(i, tuple)}
            win_keys.add(np.nan)
            win_dict = dict.fromkeys(win_keys, [])
            for tax_idx, param_combo in zip(tax_names, param_row):
                win_dict[param_combo].append(tax_idx)
            windows[win] = win_dict
        with open(self.reports_dir + '/{metric}_params.json', 'w') as handle:
            # windows dictionary has structure:
                # windows = {(w0_start, w0_end) : {(n0, k0, m0) : [TAX_NAME0, ..., TAX_NAMEn]}} # TAX_NAMEs are not separated by rank. This json dictionary will be used to locate the best param combination for specified taxa in the classification step
            json.dump(windows, handle)
    
    def grid_search(self,
                    max_n,
                    step_n,
                    max_k,
                    step_k,
                    cost_mat,
                    row_thresh=0.1,
                    col_thresh=0.1,
                    min_seqs=50,
                    rank='genus',
                    metric='f1',
                    min_n=5,
                    min_k=3,
                    criterion='orbit',
                    threads=1):
        
        # prepare n, k ranges
        n_range = np.arange(min_n, max_n, step_n)
        k_range = np.arange(min_k, max_k, step_k)
        # log calibration parameters
        logger.info('Calibration report:')
        logger.info('=' * 10 + '\nCalibration parameters:')
        logger.info('Database: {self.db}')
        logger.info(f'N sites: {" ".join(n_range)}')
        logger.info(f'K neighbours: {" ".join(k_range)}')
        logger.info(f'Max unknowns per sequence: {row_thresh * 100}%')
        logger.info(f'Max unknowns per site: {col_thresh * 100}%')
        logger.info(f'Min non-redundant sequences: {min_seqs}')
        logger.info(f'Rank used for site selection: {rank}')
        logger.info(f'Choose by metric: {metric}')
        logger.info(f'Classification criterion: {criterion}')
        logger.info(f'Threads: {threads}')
        
        print('Beginning calibration...')
        t0 = time.time()
        
        # collapse windows
        print('Collapsing windows...')
        win_indexes, win_list, rej_indexes, rej_list = collapse_windows(self.windows, self.matrix, self.tax_tab, row_thresh, col_thresh, min_seqs, threads)
        self.report_windows(win_indexes, win_list, rej_indexes, rej_list)
        
        t1 = time.time()
        print(f'Collapsed windows in {t1 - t0:.3f} seconds')
        # abort calibration if no collapsed windows are generated
        if len(win_list) == 0:
            logger.info('No windows passed the collapsing filters. Ending calibration')
            return
        
        # select sites
        print('Selecting informative sites...')
        window_sites = select_sites(win_list, self.tax_ext, rank, min_n, max_n, step_n)
        self.report_sites(window_sites, win_list, win_indexes, n_range)
        t2 = time.time()
        print(f'Sitee selection finished  in {t2 - t1:.3f} seconds')
        
        # calculate distances
        print('Calculating paired distances...')
        all_win_dists = cal_dists.get_all_distances(win_list, window_sites, cost_mat, threads, win_indexes)
        t3 = time.time()
        print(f'Distance calculation finished in {t3 - t2:.3f} seconds')

        # sort neighbours
        print('Sorting neighbours...')
        sorted_win_neighbours = cal_neighsort.sort_neighbours(win_list, all_win_dists, threads, win_indexes)
        t4 = time.time()
        print(f'Done in {t4 - t3:.3f} seconds')
        
        # classify
        print('Classifying...')
        win_classifs = cal_classify.classify_windows(win_list, sorted_win_neighbours, self.tax_ext, min_k, max_k, step_k, criterion, threads, win_indexes)
        t5 = time.time()
        print(f'Done in {t5 - t4:.3f} seconds')
        
        # get metrics
        print('Calculating metrics...')
        metrics = cal_metrics.get_metrics(win_list, win_classifs, self.tax_ext, threads)
        t6 = time.time()
        print(f'Done in {t6 - t5:.3f} seconds')
        
        # report
        print('Building report...')
        acc_report, acc_params = cal_report.build_report(win_list, metrics, 'acc', self.tax_ext, self.guide, n_range, k_range)
        prc_report, prc_params = cal_report.build_report(win_list, metrics, 'prc', self.tax_ext, self.guide, n_range, k_range)
        rec_report, rec_params = cal_report.build_report(win_list, metrics, 'rec', self.tax_ext, self.guide, n_range, k_range)
        f1_report, f1_params = cal_report.build_report(win_list, metrics, 'f1', self.tax_ext, self.guide, n_range, k_range)
        self.report_metrics(acc_report, acc_params, 'acc')
        self.report_metrics(prc_report, prc_params, 'prc')
        self.report_metrics(rec_report, rec_params, 'rec')
        self.report_metrics(f1_report, f1_params, 'f1')
        t7 = time.time()
        print(f'Done in {t7 - t6:.3f} seconds')
        
        # # plot results
        print('Plotting results...')
        cal_plot.plot_results(acc_report, acc_params, metric, self.plot_prefix, self.ranks) # TODO: define plot_prefix
        cal_plot.plot_results(prc_report, prc_params, metric, self.plot_prefix, self.ranks) # TODO: define plot_prefix
        cal_plot.plot_results(rec_report, rec_params, metric, self.plot_prefix, self.ranks) # TODO: define plot_prefix
        cal_plot.plot_results(f1_report, f1_params, metric, self.plot_prefix, self.ranks) # TODO: define plot_prefix
        t8 = time.time()
        print(f'Done in {t8 - t7:.3f} seconds')
        print(f'Finished in {t7 - t0:.3f} seconds')
        return
