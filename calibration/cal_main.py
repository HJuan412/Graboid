#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 11:06:10 2021

@author: hernan
"""

#%% libraries
import concurrent.futures
import datetime
import json
import logging
import numpy as np
import os
import pandas as pd
import re
import sys
import time

# Graboid libraries
sys.path.append("..") # use this to allow importing from a sibling package
from calibration import cal_classify
from calibration import cal_dists
from calibration import cal_metrics
from calibration import cal_report
from calibration import cal_plot
from calibration import cal_preprocess
from DATA import DATA

#%% set logger
logger = logging.getLogger('Graboid.calibrator')
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
logger.addHandler(sh)

#%% functions
def report_params(date, database, n_range, k_range, criterion, row_thresh, col_thresh, min_seqs, rank, threads, report_file):
    sep = '#' * 40 + '\n'
    with open(report_file, 'w') as handle:
        handle.write('Grid search report\n')
        handle.write(sep + '\n')
        handle.write('Parameters\n')
        handle.write(sep)
        handle.write(f'Date: {date}\n')
        handle.write(f'Database: {database}\n')
        handle.write(f'n sites: {n_range}\n')
        handle.write(f'k neighbours: {k_range}\n')
        handle.write(f'Classification criterion: {criterion}\n')
        handle.write(f'Max unknowns per sequence: {row_thresh * 100} %\n')
        handle.write(f'Max unknowns per site: {col_thresh * 100} %\n')
        handle.write(f'Min non-redundant sequences: {min_seqs}\n')
        handle.write(f'Rank used for site selection: {rank}\n')
        handle.write(f'Threads: {threads}\n')
        handle.write('\n')

def report_windows(win_indexes, win_list, rej_indexes, rej_list, report_file):
    sep = '#' * 40 + '\n'
    collapsed_tab = []
    for win_idx, win in zip(win_indexes, win_list):
        collapsed_tab.append([win_idx, win.start, win.end, win.window.shape[1], win.window.shape[0]])
    collapsed_tab = pd.DataFrame(collapsed_tab, columns='Window Start End Length Effective_sequences'.split()).set_index('Window', drop=True)
    
    rejected_tab = []
    for rej_idx, rej in zip(rej_indexes, rej_list):
        rejected_tab.append([f'Window {rej_idx}', rej])
    rejected_tab = pd.DataFrame(rejected_tab)
    
    with open(report_file, 'a') as handle:
        handle.write('Windows\n')
        handle.write(sep)
        handle.write('Collapsed windows:\n')
        handle.write(repr(collapsed_tab))
        handle.write('\n')
        handle.write('Rejected windows:\n')
        rejected_tab.to_csv(handle, sep='\t', header=None, index=None)
        handle.write('\n')

def report_sites_ext(win_indexes, win_list, windows_sites, n_range, ext_site_report):
    site_tabs = []
    for win_idx, win, win_sites in zip(win_indexes, win_list, windows_sites):
        total_sites = np.sum([len(n) for n in win_sites])
        site_array = np.full((len(n_range), total_sites), np.nan)
        n_idx = 0
        for n, n_sites in enumerate(win_sites):
            n_end = n_idx + len(n_sites)
            site_array[n:, n_idx:n_end] = win.cols[n_sites] + win.start
            n_idx = n_end
        site_tabs.append(pd.DataFrame(site_array, index = pd.MultiIndex.from_product(([win_idx], n_range), names = ['Window', 'n'])))
    
    site_report = pd.concat(site_tabs).fillna(-1).astype(np.int16)
    site_report.to_csv(ext_site_report, sep='\t')
    
def report_sites(win_indexes, windows_sites, n_range, ext_site_report, report_file):
    sep = '#' * 40 + '\n'
    sites_tab = []
    for win, sites in zip(win_indexes, windows_sites):
        site_counts = np.cumsum([len(n) for n in sites]).tolist()
        sites_tab.append([win] + site_counts)
    sites_tab = pd.DataFrame(sites_tab, columns=['Window'] + [n for n in n_range]).set_index('Window', drop=True)
    with open(report_file, 'a') as handle:
        handle.write('Sites\n')
        handle.write(sep)
        handle.write(f'Extended site report: {ext_site_report}\n')
        handle.write(repr(sites_tab))

def report_taxa(windows, win_indexes, guide, guide_ext, report_file):
    # Count the number of instances of each taxon per window
    
    # get taxids for each window
    merged_guides = pd.concat([guide_ext.loc[win.taxonomy] for win in windows])
    uniq_idxs = []
    for rk, row in merged_guides.T.iterrows():
        uniq_idxs.append(row.dropna().unique())
    count_tab = pd.DataFrame(0, index = np.concatenate(uniq_idxs), columns=pd.Series(win_indexes, name = 'window'))
    
    # for count instances of each taxon (high level taxa include representatives of child taxa)
    for win_idx, win in zip(win_indexes, windows):
        win_tax = guide_ext.loc[win.taxonomy]
        for rk, row in win_tax.T.iterrows():
            counts = row.value_counts()
            count_tab.loc[counts.index, win_idx] = counts.values
    
    # update headers
    count_tab['Rank'] = guide.loc[count_tab.index, 'Rank']
    count_tab['Taxon'] = guide.loc[count_tab.index, 'SciName']
    count_tab = count_tab.set_index(['Rank', 'Taxon'])
    
    with open(report_file, 'w') as handle:
        handle.write('# Effective sequences for each taxa found in collapsed calibration windows:\n')
        for win_idx, window in zip(win_indexes, windows):
            handle.write(f'# Window {win_idx}: [{window.start} - {window.end}]\n')
    # save report
    count_tab.to_csv(report_file, sep='\t', mode='a')

def classify(win_distances, win_list, win_indexes, taxonomy, n_range, k_range, out_dir, criterion='orbit', threads=1):
    """Direct support calculation and classification for each calibration window"""
    # win_distances: list of 3d-numpy arrays of shape (#seqs, #seqs, len(n_range))
    # win_list: list of Window objects
    # win_indexes: array of selected window indexes
    # taxonomy: pandas dataframe containing the training set's extended taxonomy
    # n_range, k_range: ranges of n and k values
    # out_dir: classification directory
    # criterion: orbit/neigh
    
    # build list of inputs for parallel classification jobs
    inputs = []
    for distances, window, window_idx in zip(win_distances, win_list, win_indexes):
        win_tax = taxonomy.loc[window.taxonomy].to_numpy()
        for n, n_dists in zip(n_range, distances):
            inputs.append([n_dists, win_tax, window_idx, n])
    
    # parallel classification, one job per window*n, each job classifies for all the k values
    n_cells = 0
    total_cells = len(win_list)*len(n_range)*len(k_range)
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(cal_classify.classify, dists, w_tax, n, k_range, out_dir, w_idx, criterion) for (dists, w_tax, w_idx, n) in inputs]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            n_cells += len(k_range)
            print(f'Classified {n_cells} of {total_cells} cells')

def get_metrics(win_list, win_indexes, classif_dir, out_dir, taxonomy):
    """Calculate calibration metrics for each cell in the grid"""
    # win_list: list of Window objects
    # win_indexes: array of selected window indexes
    # classif_dir: directory containing classification results
    # out_dir: metrics directory
    # taxonomy: pandas dataframe containing the training set's extended taxonomy
    
    win_array = np.array(win_list)
    for res_file in os.listdir(classif_dir):
        classif_results = np.load(classif_dir + '/' +res_file)
        window = win_array[win_indexes == classif_results['params'][0]][0]
        win_tax = taxonomy.loc[window.taxonomy].to_numpy()
        aprf_metrics, cross_entropy = cal_metrics.get_metrics(classif_results, win_tax)
        np.savez(out_dir + '/' + re.sub('.npz', '_metrics.npz', res_file), metrics = aprf_metrics, cross_entropy = cross_entropy, params = classif_results['params'])
#%% classes
class Calibrator:
    def __init__(self, out_dir=None):
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
        self.out_dir = re.sub('/$', '', out_dir)
        self.tmp_dir = out_dir + '/tmp'
        self.metrics_dir = self.tmp_dir + '/metrics'
        self.classif_dir = self.tmp_dir + '/classification'
        self.plots_dir = out_dir + '/plots'
        # self.reports_dir = out_dir + '/reports'
        # self.warn_dir = out_dir + '/warnings'
        
        try:
            os.mkdir(out_dir)
        except FileExistsError:
            raise Exception(f'Specified output directory "{out_dir}" already exists')
        # make a directory to store classification reports
        os.makedirs(self.metrics_dir)
        os.makedirs(self.classif_dir)
        os.makedirs(self.plots_dir)
        # build a file handler
        fh = logging.FileHandler(self.out_dir + '/calibration.log')
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        
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
        self.guide.loc[-2] = 'undetermined'
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
        n_range = np.arange(min_n, max_n + 1, step_n)
        k_range = np.arange(min_k, max_k + 1, step_k)
        
        # initialize grid search report report
        date = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        report_file = self.out_dir + '/GS_report.txt'
        ext_site_report = self.out_dir + '/sites_report.csv'
        tax_report = self.out_dir + '/taxa_report.csv'
        report_params(date, self.db, n_range, k_range, criterion, row_thresh, col_thresh, min_seqs, rank, threads, report_file)
        
        logger.info('Beginning calibration...')
        t_collapse_0 = time.time()
        
        # collapse windows
        logger.info('Collapsing windows...')
        win_indexes, win_list, rej_indexes, rej_list = cal_preprocess.collapse_windows(self.windows, self.matrix, self.tax_tab, row_thresh, col_thresh, min_seqs, threads)
        report_windows(win_indexes, win_list, rej_indexes, rej_list, report_file)
        report_taxa(win_list, win_indexes, self.guide, self.tax_ext, tax_report)
        t_collapse_1 = time.time()
        logger.info(f'Collapsed {len(win_list)} of {len(self.windows)} windows in {t_collapse_1 - t_collapse_0:.3f} seconds')
        
        # abort calibration if no collapsed windows are generated
        if len(win_list) == 0:
            logger.info('No windows passed the collapsing filters. Ending calibration')
            return
        
        # select sites
        logger.info('Selecting informative sites...')
        t_sselection_0 = time.time()
        windows_sites = cal_preprocess.select_sites(win_list, self.tax_ext, rank, min_n, max_n, step_n)
        report_sites_ext(win_indexes, win_list, windows_sites, n_range, ext_site_report)
        report_sites(win_indexes, windows_sites, n_range, ext_site_report, report_file)
        t_sselection_1 = time.time()
        logger.info(f'Site selection finished in {t_sselection_1 - t_sselection_0:.3f} seconds')
        
        # calculate distances
        logger.info('Calculating paired distances...')
        t_distance_0 = time.time()
        all_distances = [] # contains one 3d array per window. Arrays have shape (n, #seqs, #seqs), contain paired distances for every level of n
        for window, win_sites in zip(win_list, windows_sites):
            all_distances.append(cal_dists.get_distances(window, win_sites, cost_mat))
        t_distance_1 = time.time()
        logger.info(f'Distance calculation finished in {t_distance_1 - t_distance_0:.3f} seconds')
        
        # classify
        logger.info('Classifying...')
        t_classification_0 = time.time()
        classify(all_distances, win_list, win_indexes, self.tax_ext, n_range, k_range, self.classif_dir, criterion, threads)
        t_classification_1 = time.time()
        logger.info(f'Finished classifications in {t_classification_1 - t_classification_0:.3f} seconds')
        
        # get metrics
        logger.info('Calculating metrics...')
        t_metrics_0 = time.time()
        get_metrics(win_list, win_indexes, self.classif_dir, self.metrics_dir, self.tax_ext)
        CE_full_report, CE_counts = cal_metrics.CE_full_report(win_list, win_indexes, self.classif_dir, self.tax_ext, self.guide)
        acc_full_report = cal_metrics.aprf_full_report(self.metrics_dir, 'a', self.guide)
        prc_full_report = cal_metrics.aprf_full_report(self.metrics_dir, 'p', self.guide)
        rec_full_report = cal_metrics.aprf_full_report(self.metrics_dir, 'r', self.guide)
        f1_full_report = cal_metrics.aprf_full_report(self.metrics_dir, 'f', self.guide)
        # save full_reports
        CE_full_report.to_csv(self.out_dir + '/full_report__cross_entropy.csv')
        CE_counts.to_csv(self.out_dir + '/full_report__counts.csv')
        acc_full_report.to_csv(self.out_dir + '/full_report_accuracy.csv')
        prc_full_report.to_csv(self.out_dir + '/full_report_precision.csv')
        rec_full_report.to_csv(self.out_dir + '/full_report_recall.csv')
        f1_full_report.to_csv(self.out_dir + '/full_report_f1.csv')
        t_metrics_1 = time.time()
        logger.info(f'Calculated metrics in {t_metrics_1 - t_metrics_0:.3f} seconds')
        
        # report
        logger.info('Building report...')
        t_report_0 = time.time()
        ce_file, acc_file, prc_file, rec_file, f1_file = cal_report.build_reports(win_indexes, self.metrics_dir, self.out_dir, self.ranks, self.guide)
        t_report_1 = time.time()
        logger.info(f'Finished building reports in {t_report_1 - t_report_0:.3f} seconds')
        
        # # plot results
        logger.info('Plotting results...')
        t_plots_0 = time.time()
        # plot aprf
        for report_file, metric in zip((acc_file, prc_file, rec_file, f1_file),
                                       ('Accuracy', 'Precision', 'Recall', 'F1 score')):
            cal_plot.plot_aprf(report_file, metric, self.windows, out_dir = self.plots_dir)
        # plot ce
        cal_plot.plot_CE_results(ce_file, self.windows, out_dir = self.plots_dir)
        t_plots_1 = time.time()
        logger.info(f'Finished plotting in {t_plots_1 - t_plots_0:.3f} seconds')
        return
