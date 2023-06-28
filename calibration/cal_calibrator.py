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
import re
import pickle
import sys
import time
# Graboid libraries
sys.path.append("..") # use this to allow importing from a sibling package
# from calibration import cal_dists # TODO: delete cal_distances
from calibration import cal_classify
from calibration import cal_dists
from calibration import cal_metrics
from calibration import cal_neighsort
from calibration import cal_report
from calibration import cal_plot
from calibration import cal_preprocess

from DATA import DATA

#%% set logger
logger = logging.getLogger('Graboid.calibrator')
logger.setLevel(logging.DEBUG)

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
        self.plots_dir = out_dir + '/plots'
        self.classif_dir = out_dir + '/classification'
        self.warn_dir = out_dir + '/warnings'
        
        try:
            os.mkdir(out_dir)
        except FileExistsError:
            raise Exception(f'Specified output directory "{out_dir}" already exists. Pick a different name or set argmuent "clear" as True')
        self.save = True
        # make a directory to store classification reports
        os.makedirs(self.reports_dir)
        os.makedirs(self.plots_dir)
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
        self.custom = False
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
        self.custom = True
        logger.info(f'Set {raw_coords.shape[0]} custom windows at positions:')
        for coor_idx, coords in enumerate(raw_coords):
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
        report_cp = report.copy()
        report_cp.columns = pd.MultiIndex.from_tuples(report_cp.columns, names=['w_start', 'w_end'])
        report_cp.to_csv(self.reports_dir + f'/{metric}_calibration.csv') # remember to load using index=[0,1] and header=[0,1]
        
        # build params dictionary
        windows = {}
        merged_params = np.concatenate(params, 0)
        for w_idx, (win, param_row) in enumerate(zip(report.columns.tolist(), merged_params.T)):
            combos = [combo for combo in map(lambda x : x if isinstance(x, tuple) else 0, param_row)]
            win_dict = {wk:{rk:[] for rk in self.ranks} for wk in set(combos)}
            # for tax_name, param_combo in zip(tax_names, combos):
            #     win_dict[param_combo].append(tax_name)
            for tax_data, param_combo in zip(report.index.values, combos):
                # tax data is a tuple with the current taxa's rank and name
                win_dict[param_combo][tax_data[0]] = tax_data[1].upper()
            windows[win] = win_dict

        with open(self.reports_dir + f'/{metric}_params.pickle', 'wb') as handle:
            # windows dictionary has structure:
                # windows = {(w0_start, w0_end) :
                                # {(n0, k0, m0) :
                                    # {rk0:[TAX_NAME0, ..., TAX_NAMEn]}}} # This json dictionary will be used to locate the best param combination for specified taxa in the classification step
            pickle.dump(windows, handle)
    
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
                    min_n=5,
                    min_k=3,
                    criterion='orbit',
                    collapse_hm=True,
                    threads=1):
        
        # prepare n, k ranges
        n_range = np.arange(min_n, max_n, step_n)
        k_range = np.arange(min_k, max_k, step_k)
        # log calibration parameters
        logger.info('Calibration report:')
        logger.info('=' * 10 + '\nCalibration parameters:')
        logger.info(f'Database: {self.db}')
        logger.info(f'N sites: {n_range}')
        logger.info(f'K neighbours: {k_range}')
        logger.info(f'Max unknowns per sequence: {row_thresh * 100}%')
        logger.info(f'Max unknowns per site: {col_thresh * 100}%')
        logger.info(f'Min non-redundant sequences: {min_seqs}')
        logger.info(f'Rank used for site selection: {rank}')
        logger.info(f'Classification criterion: {criterion}')
        logger.info(f'Threads: {threads}')
        
        print('Beginning calibration...')
        t0 = time.time()
        
        # collapse windows
        print('Collapsing windows...')
        win_indexes, win_list, rej_indexes, rej_list = cal_preprocess.collapse_windows(self.windows, self.matrix, self.tax_tab, row_thresh, col_thresh, min_seqs, threads)
        self.report_windows(win_indexes, win_list, rej_indexes, rej_list)
        
        t1 = time.time()
        print(f'Collapsed {len(win_list)} of {len(self.windows)} windows in {t1 - t0:.3f} seconds')
        # abort calibration if no collapsed windows are generated
        if len(win_list) == 0:
            logger.info('No windows passed the collapsing filters. Ending calibration')
            return
        
        # select sites
        print('Selecting informative sites...')
        windows_sites = cal_preprocess.select_sites(win_list, self.tax_ext, rank, min_n, max_n, step_n)
        self.report_sites(windows_sites, win_list, win_indexes, n_range)
        t2 = time.time()
        print(f'Site selection finished  in {t2 - t1:.3f} seconds')
        
        # calculate distances
        print('Calculating paired distances...')
        all_distances = []
        # TODO: parallel process
        for window, win_sites in zip(win_list, windows_sites):
            all_distances.append(cal_dists.get_distances(window, win_sites, cost_mat))
        t3 = time.time()
        print(f'Distance calculation finished in {t3 - t2:.3f} seconds')

        # sort neighbours
        print('Sorting neighbours...')
        window_packages = []
        for distances in all_distances:
            # sort distances and compress them into orbitals
            sorted_distances, sorted_indexes, compressed = cal_neighsort.sort_compress(distances)
            # build classification packages
            window_packages.append(cal_neighsort.build_packages(compressed, sorted_indexes, n_range, k_range, criterion))
        t4 = time.time()
        print(f'Sorted neighbours in {t4 - t3:.3f} seconds')
        
        # classify
        print('Classifying...')
        # get supports
        for win_idx, win_package in enumerate(window_packages):
            # get packages for a single window
            for (n, k), package in win_package.items():
                # get individual packages
                id_array, data_array = cal_classify.classify_V(package, self.tax_tab)
                predicted_u, real_u_support, predicted_w, real_w_support, predicted_d, real_d_support = cal_classify.get_supports(id_array, data_array, self.tax_tab) # TODO: verify that this work, self.tax_tab may need adjustments
                # save classification results
                np.savez(self.classif_dir + f'/{win_idx}_{n}_{k}.npz',
                         predicted_u = predicted_u,
                         predicted_w = predicted_w,
                         predicted_d = predicted_d,
                         real_u_support = real_u_support,
                         real_w_support = real_w_support,
                         real_d_support = real_d_support)
        t5 = time.time()
        print(f'Finished classifications in {t5 - t4:.3f} seconds')
        
        # get metrics
        print('Calculating metrics...')
        for res_file in os.listdir(self.clasi_dir):
            metrics, cross_entropy = cal_metrics.get_metrics0(self.clasif_dir + '/' +res_file, self.tax_tab)
            np.savez(self.reports_dir + '/' + re.sub('.npz', '_metrics.npz'), metrics = metrics, cross_entropy = cross_entropy)
        t6 = time.time()
        print(f'Done in {t6 - t5:.3f} seconds')
        
        # report
        print('Building report...')
        # acc_report, acc_params = cal_report.build_report(win_list, metrics, 'acc', self.tax_ext, self.guide, n_range, k_range)
        # prc_report, prc_params = cal_report.build_report(win_list, metrics, 'prc', self.tax_ext, self.guide, n_range, k_range)
        # rec_report, rec_params = cal_report.build_report(win_list, metrics, 'rec', self.tax_ext, self.guide, n_range, k_range)
        # f1_report, f1_params = cal_report.build_report(win_list, metrics, 'f1', self.tax_ext, self.guide, n_range, k_range)
        # self.report_metrics(acc_report, acc_params, 'acc')
        # self.report_metrics(prc_report, prc_params, 'prc')
        # self.report_metrics(rec_report, rec_params, 'rec')
        # self.report_metrics(f1_report, f1_params, 'f1')
        t7 = time.time()
        print(f'Done in {t7 - t6:.3f} seconds')
        
        # # plot results
        print('Plotting results...')
        if self.save:
            # lin_codes = self.guide.set_index('SciName')['LinCode'] # use this to add lineage codes to calibration heatmaps
            # with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
            #     for mt_report, mt_params, mt in zip((acc_report, prc_report, rec_report, f1_report),
            #                                         (acc_params, prc_params, rec_params, f1_params),
            #                                         ('acc', 'prc', 'rec', 'f1')):
            #         executor.submit(cal_plot.plot_results, mt_report, mt_params, mt, self.plots_dir, self.ranks, lin_codes, collapse_hm, self.custom)
            t8 = time.time()
            print(f'Done in {t8 - t7:.3f} seconds')
        print(f'Finished in {t8 - t0:.3f} seconds')
        return
