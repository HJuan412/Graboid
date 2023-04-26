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
    def __init__(self, out_dir, warn_dir, prefix='calibration'):
        self.out_dir = out_dir
        self.classif_dir = out_dir + '/classification'
        self.warn_dir = warn_dir
        
        # make a directory to store classification reports
        os.makedirs(self.classif_dir, exist_ok=True)
        # prepare out files
        self.report_file = self.out_dir + f'/{prefix}.report'
        self.classif_file = self.out_dir + f'/{prefix}.classif'
        self.meta_file = self.out_dir + f'/{prefix}.meta'
    
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
    
    def set_database(self, database):
        self.db = database
        try:
            self.db_dir = DATA.get_database(database)
        except Exception as excp:
            raise excp
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
    
    def grid_search(self,
                    max_n,
                    step_n,
                    max_k,
                    step_k,
                    cost_mat,
                    row_thresh=0.2,
                    col_thresh=0.2,
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
        # log window collapses
        logger.info('=' * 10 + 'Collapsed windows:')
        for w_idx, win in zip(win_indexes, win_list):
            logger.info(f'Window {w_idx} {self.windows[w_idx]}: collapsed into matrix of shape {win.window.shape}')

        logger.info('=' * 10 + 'Rejected windows:')
        for r_idx, rej in zip(rej_indexes, rej_list):
            logger.info(f'Window {r_idx} {self.windows[r_idx]}: ' + rej)
        t1 = time.time()
        print(f'Collapsed windows in {t1 - t0:.3f} seconds')
        # abort calibration if no collapsed windows are generated
        if len(win_list) == 0:
            logger.info('No windows passed the collapsing filters. Ending calibration')
            return
        
        # select sites
        print('Selecting informative sites...')
        window_sites = select_sites(win_list, self.tax_ext, rank, min_n, max_n, step_n)
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
        # window_taxes = [self.tax_ext.loc[win.taxonomy].to_numpy() for win in win_list]
        # with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        #     future_metrics = [future for future in executor.map(cal_metrics.get_metricas, win_classifs, window_taxes)]
        metrics = cal_metrics.get_metrics(win_list, win_classifs, self.tax_ext, threads)
        t6 = time.time()
        print(f'Done in {t6 - t5:.3f} seconds')
        
        # report
        print('Building report...')
        # met_codes = {'acc':0, 'prc':1, 'rec':2, 'f1':3}
        # pre_report, params = cal_report.build_prereport(metrics, met_codes[metric], self.tax_ext)
        # # process report
        # pre_report.columns = [f'W{w_idx} [{win.start} - {win.end}]' for w_idx, win in zip(win_indexes, win_list)]
        # index_datum = self.guide.loc[pre_report.index.get_level_values(1)]
        # pre_report.index = pd.MultiIndex.from_arrays([index_datum.Rank, index_datum.SciName])
        
        # params = cal_report.translate_params(params, n_range, k_range)
        report, params = cal_report.build_report(win_list, metrics, metric, self.tax_ext, self.guide, n_range, k_range, win_indexes)
        t7 = time.time()
        print(f'Done in {t7 - t6:.3f} seconds')
        
        # # plot results
        print('Plotting results...')
        cal_plot.plot_results(report, params, metric, self.plot_prefix, self.ranks) # TODO: define plot_prefix
        t8 = time.time()
        print(f'Done in {t8 - t7:.3f} seconds')
        print(f'Finished in {t7 - t0:.3f} seconds')
        return report, params
