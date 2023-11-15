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
import shutil
import sys
import time

# Graboid libraries
sys.path.append("..") # use this to allow importing from a sibling package
from calibration import cal_classify
from calibration import cal_dists
from calibration import cal_metrics
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
def set_options(on=True):
    # Set display options to show all rows and columns (used for report construction)
    options = ['max_rows', 'max_columns', 'expand_frame_repr']
    values = [None, None, False]
    if on:
        for opt, val in zip(options, values):
            pd.set_option(f'display.{opt}', val)
    else:
        for opt in options:
            pd.reset_option(f'display.{opt}')

def header(text):
    text = f'### {text} '
    text += '#' * (80 - len(text))
    return '\n' + text + '\n'

def print_table(tab):
    set_options()
    tab = repr(tab)
    set_options(False)
    return tab + '\n'

def count_taxa(windows, win_indexes, guide, guide_ext):
    # Count the number of instances of each taxon per window
    
    # get taxids for each window
    merged_guides = pd.concat([guide_ext.loc[win.taxonomy] for win in windows])
    uniq_idxs = [] # contains the unique KNOWN taxa for each rank (sorted by rank)
    for rk, row in merged_guides.T.iterrows():
        uniq_idxs.append(row.dropna().unique())
    count_tab = pd.DataFrame(0, index = np.concatenate(uniq_idxs), columns=pd.Series(win_indexes, name = 'window'))
    
    # for count instances of each taxon (high level taxa include representatives of child taxa)
    for win_idx, win in zip(win_indexes, windows):
        win_tax = guide_ext.loc[win.taxonomy]
        for rk, row in win_tax.T.iterrows():
            counts = row.value_counts()
            count_tab.loc[counts.index, win_idx] = counts.values
    
    # update index
    index = pd.MultiIndex.from_frame(guide.loc[count_tab.index, ['Rank', 'SciName']].rename(columns={'SciName':'Taxon'}))
    count_tab.index = index
    return count_tab, merged_guides

def build_CE_summary(ce_table, out_file=None):
    ce_summary = []
    for rk, rk_subtab in ce_table.groupby(level=0, sort=False):
        mean_vals = rk_subtab.mean(0)
        mean_vals.name = rk
        ce_summary.append(mean_vals)
    ce_summary = pd.DataFrame(ce_summary)
    if not out_file is None:
        ce_summary.to_csv(out_file, sep='\t')
    return ce_summary

def build_APRF_summary(table, out_file=None):
    """Get best score for a given metric, for each window"""
    # prepare report(get best metric for each taxon for each window)
    summary = pd.DataFrame(index=table.index, columns = pd.MultiIndex.from_product((table.columns.levels[0], ('n', 'k', 'Method', 'Score'))))
    for win, win_tab in table.T.groupby(level=0, sort=False):
        scores = win_tab.max(0)
        locs = np.argmax(win_tab.values, 0)[scores >= 0]
        scores = scores.loc[scores >= 0]
        params = win_tab.index[locs].to_frame().values
        summary.loc[scores.index, (win, 'Score')] = scores
        summary.loc[scores.index, [(win, 'n'), (win, 'k'), (win, 'Method')]] = params[:,1:]
    if not out_file is None:
        summary.to_csv(out_file, sep='\t')
    return summary

def final_recount(report, taxa_count, out_dir=None):
    """Count absences, losses, failures and wins"""
    # for each taxon:
        # abs_report: lists windows in which the taxon is absent
        # lose_report: lists windows in which the taxon yields a score of 0
        # fail_report: lists taxa that are absent or yield 0 in all windows
        # win_report: lists windows in which the taxon yields a score over 0
        
    # get absents (taxa that are missing in one or more windows)
    abs_scores = report.droplevel(0, axis=1)['Score'].isna()
    abs_scores = abs_scores.loc[abs_scores.sum(1) > 0]
    x_array = np.full(abs_scores.shape, np.nan, dtype=object)
    x_array[abs_scores.to_numpy()] = 'X'
    abs_report = pd.DataFrame(x_array, abs_scores.index, columns=report.columns.levels[0])
    abs_report['Total_absences'] = abs_scores.sum(1)
    
    # get loses (taxa that have score 0 in one or more windows)
    lose_scores = report.droplevel(0, axis=1)['Score'] == 0
    lose_scores = lose_scores.loc[lose_scores.sum(1) > 0]
    # locate taxa that have single representatives (can only result in loses)
    single_taxa = (taxa_count.loc[lose_scores.index] == 1).to_numpy()
    x_array = np.full(lose_scores.shape, np.nan, dtype=object)
    x_array[lose_scores.to_numpy()] = 'X'
    x_array[lose_scores.to_numpy() & single_taxa] = 'S'
    lose_report = pd.DataFrame(x_array, lose_scores.index, columns=report.columns.levels[0])
    lose_report['Total_losses'] = lose_scores.sum(1)
    
    # get failures (taxa that have only 0 or null values)
    lose_scores = report.droplevel(0, axis=1)['Score'] == 0
    abs_scores = report.droplevel(0, axis=1)['Score'].isna()
    fail_scores = lose_scores | abs_scores
    fail_scores = fail_scores.loc[fail_scores.sum(1) == fail_scores.shape[1]]
    fail_report = fail_scores.index.to_frame().reset_index(drop=True)
    
    # get successes (taxa that have valid score over 0 in one or more windows)
    win_scores = report.droplevel(0, axis=1)['Score'] > 0
    win_scores = win_scores.loc[win_scores.sum(1) > 0]
    x_array = np.full(win_scores.shape, np.nan, dtype=object)
    x_array[win_scores.to_numpy()] = 'X'
    win_report = pd.DataFrame(x_array, win_scores.index, columns=report.columns.levels[0])
    win_report['Total_successes'] = win_scores.sum(1)
    
    if not out_dir is None:
        with open(out_dir + '/recount_absences.csv', 'w') as handle:
            handle.write('# Absent taxa per window\n')
            abs_report.to_csv(handle, mode='a')
        with open(out_dir + '/recount_loses.csv', 'w') as handle:
            handle.write('# X: Taxa yielded a score of 0 in window\n')
            handle.write('# S: Taxa has a single representative in window')
            handle.write('# NOTICE: taxa with a single representative can\'t be correctly classified during calibration\n')
            lose_report.to_csv(handle, mode='a')
        with open(out_dir + '/recount_failures.csv', 'w') as handle:
            handle.write('# Taxa that are missing or yield a score of 0 in all windows\n')
            fail_report.to_csv(handle, mode='a')
        with open(out_dir + '/recount_wins.csv', 'w') as handle:
            handle.write('# Taxa that yield a score greater than 0 in at least one window\n')
            win_report.to_csv(handle, mode='a')
    return abs_report, lose_report, fail_report, win_report

# grid search functions
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

def save_classifications(cls_dir, out_dir):
    """Save classification results to a single npz file"""
    
    cls_files = os.listdir(cls_dir)
    cls_tabs = {}
    
    # extract classification results for the three methods from each file
    for file in cls_files:
        npz = np.load(cls_dir + '/' + file)
        for idx, mth in enumerate('uwd'):
            key = re.sub('.npz', f'_{idx}', file)
            cls_tabs[key] = npz[f'predicted_{mth}']
    
    np.savez(out_dir + '/classifs.npz', **cls_tabs)
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
        db_meta = DATA.DBASE_INFO[database]
        
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
        if step > size:
            raise Exception(f'Window displacement rate ({step}) must be lower or equal to the window size ({size})')
        
        # adjust window size to get uniform distribution (avoid having to use a "tail" window)
        last_position = self.max_pos - size
        n_windows = int(np.ceil(last_position / step))
        w_start = np.linspace(0, last_position, n_windows, dtype=int)
        self.windows = np.array([w_start, w_start + size]).T
        self.custom = False
        logger.info(f'Set {n_windows} windows of size {size} at intervals of {w_start[1] - w_start[0]}')
    
    def set_custom_windows(self, starts, ends):
        # ensure arguments are lists
        if isinstance(starts, int):
            starts = [starts]
        if isinstance(ends, int):
            ends = [ends]
        
        # ensure start and end lengths match
        if len(starts) != len(ends):
            raise Exception(f'Error: start and end coordinates do not match ({len(starts)} starts, {len(ends)} ends)')
        
        raw_coords = np.array([starts, ends], dtype=np.int).T
        
        # ensure all coordinate pairs are valid
        invalid = raw_coords[:, 0] >= raw_coords[:, 1]
        if invalid.sum() > 0:
            raise Exception(f'At least one pair of coordinates is invalid: {[list(i) for i in raw_coords[invalid]]}')
        
        # ensure all cordinates are within bounds
        out_of_bounds = ((raw_coords < 0) | (raw_coords >= self.max_pos))
        out_of_bounds = out_of_bounds[:,0] | out_of_bounds[:,1]
        if out_of_bounds.sum() > 0:
            raise Exception(f'At least one pair of coordinates is out of bounds [0 {self.max_pos}]: {[list(i) for i in raw_coords[out_of_bounds]]}')
        
        # set custom window values
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
                    threads=1,
                    clear_tmp=True):
        
        t0 = time.time()
        # prepare n, k ranges
        n_range = np.arange(min_n, max_n + 1, step_n)
        k_range = np.arange(min_k, max_k + 1, step_k)
        
        # initialize grid search report report
        reporter = RunReporter(self.out_dir, self.db)
        
        logger.info('Beginning calibration...')
        t_collapse_0 = time.time()
        
        # collapse windows
        logger.info('Collapsing windows...')
        win_indexes, win_list, rej_indexes, rej_list = cal_preprocess.collapse_windows(self.windows, self.matrix, self.tax_tab, row_thresh, col_thresh, min_seqs, threads)
        taxa_counts, merged_guides = count_taxa(win_list, win_indexes, self.guide, self.tax_ext)  # keep the number of taxa per window

        # build windows tab
        win_tab = pd.DataFrame(self.windows, columns = ['Start', 'End'])
        win_tab.index.name = 'Window'
        win_tab.to_csv(self.out_dir + '/windows.csv')
        # store window taxonomies (used for building confusion matrices later)
        win_taxa = {}
        for win_idx, window in zip(win_indexes, win_list):
            win_taxa[str(win_idx)] = self.tax_ext.loc[window.taxonomy].fillna(-1).to_numpy().astype(int)
        np.savez(self.out_dir + '/win_taxa.npz', **win_taxa)
        t_collapse_1 = time.time()
        logger.info(f'Collapsed {len(win_list)} of {len(self.windows)} windows in {t_collapse_1 - t_collapse_0:.2f} seconds')
        
        # abort calibration if no collapsed windows are generated
        if len(win_list) == 0:
            logger.info('No windows passed the collapsing filters. Ending calibration')
            return
        
        # save parameters, used for reporting calibration metrics for classification parameters
        np.savez(self.out_dir + '/params.npz', n = n_range, k = k_range, windows = np.array(win_idx))
        # select sites
        logger.info('Selecting informative sites...')
        t_sselection_0 = time.time()
        windows_sites = cal_preprocess.select_sites(win_list, self.tax_ext, rank, min_n, max_n, step_n)
        t_sselection_1 = time.time()
        logger.info(f'Site selection finished in {t_sselection_1 - t_sselection_0:.2f} seconds')
        
        # report parameters
        reporter.build_report(n_range,
                              k_range,
                              criterion,
                              row_thresh,
                              col_thresh,
                              min_seqs,
                              rank,
                              threads,
                              win_indexes,
                              win_list,
                              rej_indexes,
                              rej_list,
                              windows_sites,
                              taxa_counts,
                              merged_guides)
        
        # calculate distances
        logger.info('Calculating paired distances...')
        t_distance_0 = time.time()
        all_distances = [] # contains one 3d array per window. Arrays have shape (n, #seqs, #seqs), contain paired distances for every level of n
        for window, win_sites in zip(win_list, windows_sites):
            all_distances.append(cal_dists.get_distances(window, win_sites, cost_mat))
        t_distance_1 = time.time()
        logger.info(f'Distance calculation finished in {t_distance_1 - t_distance_0:.2f} seconds')
        
        # classify
        logger.info('Classifying...')
        t_classification_0 = time.time()
        classify(all_distances, win_list, win_indexes, self.tax_ext, n_range, k_range, self.classif_dir, criterion, threads)
        save_classifications(self.classif_dir, self.out_dir)
        t_classification_1 = time.time()
        logger.info(f'Finished classifications in {t_classification_1 - t_classification_0:.2f} seconds')
        
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
        CE_full_report.to_csv(self.out_dir + '/report__cross_entropy.csv') # double underscore so CE doesn't mix with APRF
        acc_full_report.to_csv(self.out_dir + '/report_accuracy.csv')
        prc_full_report.to_csv(self.out_dir + '/report_precision.csv')
        rec_full_report.to_csv(self.out_dir + '/report_recall.csv')
        f1_full_report.to_csv(self.out_dir + '/report_f1.csv')
        t_metrics_1 = time.time()
        logger.info(f'Calculated metrics in {t_metrics_1 - t_metrics_0:.2f} seconds')
        
        # metric summaries
        logger.info('Building metric summaries...')
        t_report_0 = time.time()
        # ce_file, acc_file, prc_file, rec_file, f1_file = cal_report.build_reports(win_indexes, self.metrics_dir, self.out_dir, self.ranks, self.guide)
        summ_acc = build_APRF_summary(acc_full_report, out_file = self.out_dir + '/summary_accuracy.csv')
        summ_prc = build_APRF_summary(prc_full_report, out_file = self.out_dir + '/summary_precision.csv')
        summ_rec = build_APRF_summary(rec_full_report, out_file = self.out_dir + '/summary_recall.csv')
        summ_f1 = build_APRF_summary(f1_full_report, out_file = self.out_dir + '/summary_f1.csv')
        summ_ce = build_CE_summary(CE_full_report, out_file = self.out_dir + '/summary__cross_entropy.csv')
        final_recount(summ_f1, taxa_counts, out_dir = self.out_dir)
        t_report_1 = time.time()
        logger.info(f'Finished building reports in {t_report_1 - t_report_0:.2f} seconds')
        
        # # plot results
        logger.info('Plotting results...')
        t_plots_0 = time.time()
        # plot aprf
        lincode_guide = self.guide.set_index('SciName', drop=False)
        lincode_guide = lincode_guide.LinCode + ' ' +lincode_guide.SciName
        
        for report_tab, metric in zip((summ_acc, summ_prc, summ_rec, summ_f1),
                                       ('Accuracy', 'Precision', 'Recall', 'F1_score')):
            cal_plot.plot_APRF(report_tab, metric, win_tab, self.plots_dir, lincode_guide, collapse_hm, custom=True)
        
        # plot ce
        cal_plot.plot_CE(summ_ce, win_tab, self.plots_dir)
        t_plots_1 = time.time()
        logger.info(f'Finished plotting in {t_plots_1 - t_plots_0:.2f} seconds')
        
        # clear temporal files
        if clear_tmp:
            logger.info('Removing temporal files')
            shutil.rmtree(self.tmp_dir)
        t1 = time.time()
        logger.info(f'Finished calibration in {t1 - t0:.2f} seconds')
        return

class RunReporter:
    def __init__(self,
                 out_dir,
                 db):
        
        self.out_dir = out_dir
        self.report = self.out_dir + '/GS_report.txt'
        self.sites_report = self.out_dir + '/sites_report.csv'
        self.tax_report = self.out_dir + '/taxa_report.csv'
        self.date = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        self.db = db
        
    def build_windows_report(self, win_idxs, win_list, rej_idxs, rej_list):
        collapsed_tab = []
        for win_idx, win in zip(win_idxs, win_list):
            collapsed_tab.append([win_idx, win.start, win.end, win.window.shape[1], win.window.shape[0]])
        self.collapsed_tab = pd.DataFrame(collapsed_tab, columns='Window Start End Length Effective_sequences'.split()).set_index('Window', drop=True)
        
        rejected_tab = []
        for rej_idx, rej in zip(rej_idxs, rej_list):
            rejected_tab.append([rej_idx, rej])
        self.rejected_tab = pd.DataFrame(rejected_tab, columns='Window Reason'.split())
    
    def build_sites_report(self, n_range, win_idxs, win_list, win_sites):
        sites_summary = []
        sites_report = []
        for win_idx, win, win_sites in zip(win_idxs, win_list, win_sites):
            site_counts = np.cumsum([len(n) for n in win_sites]).tolist()
            sites_summary.append([win_idx] + site_counts)
            total_sites = site_counts[-1]
            site_array = np.full((len(n_range), total_sites), -1, dtype=np.int16)
            n_idx = 0
            for n, n_sites in enumerate(win_sites):
                n_end = n_idx + len(n_sites)
                site_array[n:, n_idx:n_end] = win.cols[n_sites] + win.start
                n_idx = n_end
            win_sites_report = pd.DataFrame(site_array, index = pd.MultiIndex.from_product(([win_idx], n_range), names = ['Window', 'n']))
            win_sites_report['Total_sites'] = site_counts
            sites_report.append(win_sites_report)
            
        self.sites_summary = pd.DataFrame(sites_summary, columns=['Window'] + [n for n in n_range]).set_index('Window', drop=True)
        with open(self.sites_report, 'w') as handle:
            handle.write('# Selected site indexes for each value of n\n')
            handle.write('# Null values = -1')
        sites_report.to_csv(self.sites_report, sep='\t', mode='a')
        
    def build_tax_report(self, count_tab, merged_guides, win_idxs, win_list):
        with open(self.tax_report, 'w') as handle:
            handle.write('# Effective sequences for each taxa found in collapsed calibration windows:\n')
            for win_idx, window in zip(win_idxs, win_list):
                handle.write(f'# Window {win_idx}: [{window.start} - {window.end}]\n')
        # save report
        count_tab.to_csv(self.tax_report, sep='\t', mode='a')
        
        # summary tax report, write n tax per rank for each window
        tax_summary = pd.DataFrame(index = count_tab.columns, columns = merged_guides.columns)
        for rk, rk_tab in self.count_tab.groupby(level=0):
            tax_summary.loc[:, rk] = (rk_tab > 0).sum(0)
        self.tax_summary = tax_summary
    
    def build_report(self,
                     n_range,
                     k_range,
                     criterion,
                     row_thresh,
                     col_thresh,
                     min_seqs,
                     rank,
                     threads,
                     win_idxs,
                     win_list,
                     rej_idxs,
                     rej_list,
                     win_sites,
                     count_tab,
                     merged_guides):
        self.build_windows_report(win_idxs, win_list, rej_idxs, rej_list)
        self.build_sites_report(n_range, win_idxs, win_list, win_sites)
        self.build_tax_report(count_tab, merged_guides, win_idxs, win_list)
        
        lines = ['Graboid Calibration report',
                 'Date: ' + self.date, '',
                 header('Parameters'),
                 f'Database: {self.db}',
                 f'n sites: {n_range}',
                 f'k neighbours: {k_range}',
                 f'Classification criterion: {criterion}',
                 f'Max unknowns per sequence: {row_thresh * 100} %',
                 f'Max unknowns per site: {col_thresh * 100} %',
                 f'Min non-redundant sequences: {min_seqs}',
                 f'Rank used for site selection: {rank}',
                 f'Threads: {threads}', '',
                 header('Windows'),
                 'Collapsed windows:',
                 print_table(self.collapsed_tab),
                 'Rejected windows:',
                 print_table(self.rejected_tab),
                 header('Taxa'),
                 'Taxa per rank per window:',
                 print_table(self.tax_summary),
                 f'Extended taxonomy report: {self.tax_report}',
                 header('Sites'),
                 'Selected sites per n per window:',
                 print_table(self.sites_summary),
                 f'Extended sites report: {self.sites_report}']
        with open(self.report, 'w') as handle:
            handle.write('\n'.join(lines))