#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 09:59:03 2022

@author: hernan
Director for the classification of sequences of unknown taxonomic origin
"""
#%%
from classification import classification
from classification import cost_matrix
import logging
from mapping import director as mpdir
from mapping import matrix
from preprocess import feature_selection as fsele
from preprocess import windows
import numpy as np
import pandas as pd
#%% variables
mode_dict = {'m':'majority',
             'w':'wknn',
             'd':'dwknn'}

#%% set logger
logger = logging.getLogger('Graboid.Classifier')
logger.setLevel(logging.INFO)

#%%
def tr_report(report, query_names, rank_names, taxon_names):
    q_dict = {idx:acc for idx, acc in enumerate(query_names)}
    rk_dict = {idx:rk for idx, rk in enumerate(rank_names)}
    tax_dict = {taxid:tax for tax, taxid in taxon_names.taxID.iteritems()}
    mode_dict = {'m':'majority',
                 'w':'wknn',
                 'd':'dwknn'}
    report['idx'].replace(q_dict, inplace=True)
    report['rank'].replace(rk_dict, inplace=True)
    report['taxon'].replace(tax_dict, inplace=True)
    report['mode'].replace(mode_dict, inplace=True)

def community_report(report, rank, by='count'):
    # generates a report of the estimate composition of the comunity based on the individual reads classification
    final_report = []
    rk_tab = report.loc[rank]
    max_value = int(rk_tab[by].max())
    taxa_idx = {tx:idx for idx, tx in enumerate(rk_tab.taxon.unique())}
    n_tax = len(taxa_idx)
    n_k = len(rk_tab.unique())
    value_matrix = np.zeros((n_tax, max_value, n_k), dtype=np.int32)
    
    for k_idx, (k, k_subtab) in enumerate(rk_tab.groupby('K')):
        for tax, tax_subtab in k_subtab.groupby('taxon'):
            tax_idx = taxa_idx[tax]
            # Count number of occurrences for each level of support for the current tax
            values = tax_subtab[by].value_counts().sort_index()
            values.set_axis(values.index.astype(int))
            values = values.reset_index()
            values.rename({'index':'support'})
            values['K'] = k
            values['tax'] = tax
            final_report.append(values)
            # update value matrix
            value_matrix[tax_idx, values.support.values, k_idx] = values[by].values
    return
#%%
class Director:
    def __init__(self, out_dir, tmp_dir, warn_dir):
        self.out_dir = out_dir
        self.tmp_dir = tmp_dir
        self.warn_dir = warn_dir
        
        self.taxa = {}
        self.loader = windows.WindowLoader()
        self.selector = fsele.Selector()
        self.mapper = mpdir.Director(tmp_dir, warn_dir)
    
    @property
    def ref_mat(self):
        return self.loader.matrix
    @property
    def ref_bounds(self):
        return self.loader.bounds
    
    def set_ref_data(self, mat_file, acc_file, tax_file):
        self.loader.set_files(mat_file, acc_file, tax_file)
        self.get_overlap()
    
    def set_taxguide(self, guide_file):
        self.taxguide = pd.read_csv(guide_file, index_col=0)
    
    def set_order(self, order_file):
        self.selector.load_order_mat(order_file)
    
    def set_db(self, db_dir):
        self.mapper.set_blastdb(db_dir)
    
    def set_dist_mat(self, mat_code):
        matrix = cost_matrix.get_matrix(mat_code)
        if matrix is None:
            print('Could not set distance matrix, invalid matrix code')
            return
        self.cost_mat = matrix
        
    def set_query(self, fasta_file, query_name=None, threads=1):
        # load query files
        # if query is already mapped, load map, else map query
        query_mat, query_acc = self.mapper.get_files(fasta_file, query_name)
        try:
            query_data = np.load(query_mat)
            self.query_mat = query_data['matrix']
            self.query_bounds = query_data['bounds']
            with open(query_acc, 'r') as acc_handle:
                self.query_accs = acc_handle.read().splitlines()
        except FileNotFoundError:
            self.query_mat, self.query_bounds, self.query_accs = self.mapper.direct(fasta_file, threads=threads, keep=True)
        
        self.get_overlap()
    
    def get_overlap(self):
        # method called by set_query and set_ref_data, only completes when both are set
        try:
            overlap_low = max(self.query_bounds[0], self.ref_bounds[0])
            overlap_high = min(self.query_bounds[1], self.ref_bounds[1])
            if overlap_high > overlap_low:
                self.overlap = [overlap_low, overlap_high]
        except TypeError:
            return
    
    def classify(self, w_start, w_end, k, n, mode='mwd', site_rank='genus', out_path=None):
        # check query data and reference data
        try:
            self.query_mat
        except AttributeError:
            print('There is no query file set')
            return
        try:
            self.ref_mat
        except AttributeError:
            print('There is no reference matrix set')
        # check that given coordinates are within valid space
        try:
            if self.overlap is None:
                print('Error: No valid overlap between query and reference sequences')
                return
        except AttributeError:
            print('No overlap set between query and reference sequences')
            return
        # check valid overlap
        try:
            self.overlap
        except AttributeError:
            print(f'No overlap found between query bounds {self.query_bounds} and reference bounds {self.ref_bounds}')
            return
        
        window = [np.max(w_start, self.overlap[0]), np.min(w_end, self.overlap[1])]
        window_length = window[1] - window[0]
        if window_length < 0:
            print(f'Error: No overlap bewteen given coordinates {w_start} - {w_end} and the valid overlap {self.overlap[0]} - {self.overlap[1]}')
            return
        if window[0] > w_start or window[1] < w_end:
            print('Cropped given window from {w_start} - {w_end} to {window[0]} - {window[1]} to fit valid overlap')
        
        # account offset for query and reference
        ref_offset = window[0] - self.ref_bounds[0]
        query_offset = self.ref_bounds[0] - self.query_bounds[0]
        
        # select sites
        ref_sites = self.selector.select_sites(ref_offset, window_length + ref_offset, n, site_rank)
        query_sites = ref_sites + query_offset
        
        # collect sequence and taxonomic data
        ref_window = self.loader.get_window(window[0], window[1], row_thresh=0, col_thresh=1)
        ref_data = ref_window.eff_mat[:, ref_sites]
        query_data = self.query_mat[:, query_sites]
        ref_tax = ref_window.eff_tax
        
        # get_distances (prev_disances isn't used, kept for compatiblity with calibration)
        report, prev_distances = classification.classify(query_data, ref_data, self.dist_mat, ref_tax, k, mode)
        
        report[f'n per taxon ({site_rank})'] = n
        report['total n'] = len(ref_sites)
        report['start'] = window[0]
        report['end'] = window[1]
        
        sites_report = [window[0], window[1]] + list(ref_sites - ref_offset)
        
        report = tr_report(report, self.query_accs, self.ranks, self.taxguide)
        report.reset_index(drop=True, inplace=True)
        
        if out_path is None:
            return report, sites_report
        report.to_csv(f'{self.out_dir}/{out_path}.csv')
        sites_report.to_csv(f'{self.out_dir}/{out_path}.sites')

#%% Deprecated
def get_best_params(subreport, metric='F1_score'):
    # from a subsection of a calibration report, get the average score for the given metric for each represented combination of parameters
    params = []
    for w_start, win_tab in subreport.groupby('w_start'):
        for k, k_tab in win_tab.groupby('K'):
            for n, n_tab in k_tab.groupby('n_sites'):
                params.append(pd.Series({'w_start':w_start,
                                         'w_end':win_tab.iloc[0].w_end,
                                         'K':k,
                                         'n':n,
                                         metric:n_tab[metric].mean()}))
    param_tab = pd.DataFrame(params)
    return param_tab.sort_values(metric, ascending=False)

def get_valid_windows(windows, overlap, crop=True):
    # without = windows fully outside the overlapped reference
    without = np.logical_or(windows[:,1] < overlap[0], windows[:, 0] > overlap[1])
    partial = np.invert(without)
    within = np.logical_and(windows[:,0] > overlap[0], windows[:,1] < overlap[1])
    
    invalid = windows[without]
    valid = windows[partial]
    if len(invalid) > 0:
        print(f'The following windows:\n{invalid}\nDo nor overlap with coordinates {overlap}. Will be discarded')
    if crop:
        valid[:,0] = [np.max(v, overlap[0]) for v in valid[:,0]]
        valid[:,1] = [np.min(v, overlap[1]) for v in valid[:,1]]
    else:
        valid = windows[within]
    
    if np.sum(partial) != np.sum(within):
        print(f'The valid windows:\n{windows[partial]}\nWere truncated to:\n{valid}\nTo fit within coordinates {overlap}')
    return valid

class DirectorOLD:
    def __init__(self, out_dir, tmp_dir, warn_dir):
        self.out_dir = out_dir
        self.tmp_dir = tmp_dir
        self.warn_dir = warn_dir
        self.mat_file = None
        self.acc_file = None
        self.tax_file = None
        self.taxguide_file = None
        self.report = None
        self.taxa = []
        self.windows = None
        self.query_blast = None
        self.query_mat = None
        self.result = None
        self.mapper = mpdir.Director(tmp_dir, warn_dir)
        self.loader = windows.WindowLoader()
        self.selector = fsele.Selector()
    
    @property
    def ref_mat(self):
        return self.loader.matrix
    @property
    def ref_bounds(self):
        return self.loader.bounds
    @property
    def ranks(self):
        if self.loader.tax_tab is None:
            return []
        cols = self.loader.tax_tab.columns.tolist()
        return [rank for rank in cols if len(rank.split('_')) == 1]
    
    def set_reference(self, mat_file, acc_file, tax_file, taxguide_file, order_file):
        self.mat_file = mat_file
        self.acc_file = acc_file
        self.tax_file = tax_file
        self.taxguide_file = taxguide_file
        
        self.loader.set_files(mat_file, acc_file, tax_file)
        self.taxguide = pd.read_csv(taxguide_file, index_col=0)
        # this the matrix file containing the ordered sites for each taxon
        self.selector.load_order_mat(order_file)
    
    def set_db(self, db_dir):
        self.mapper.set_blastdb(db_dir)
        
    def set_report(self, report_file):
        report = pd.read_csf(report_file)
        self.w_len = report['w_end'].iloc[0] - report['w_start'].iloc[1]
        self.w_step = report['w_start'].iloc[1] - report['w_start'].iloc[0]
        self.report = report
    
    # def set_taxa(self, taxa):
    #     taxonomies, missing = get_taxonomy(taxa, self.taxguide)
    #     if len(missing) > 0:
    #         print(f'Given taxa are not present in the reference dataset:\n\
    #               {", ".join(missing)}')
    #     for tax in taxa:
    #         self.taxa.append(taxa)
    
    def check_report(self):
        if self.report is None:
            print('No calibration report set')
            return True
    
    def check_query_map(self):
        if self.query_blast is None:
            print('No query map is set')
            return True
    
    def check_ref(self):
        if self.ref_mat is None:
            print('No reference matrix is set')
            return True
        
    def check_query(self):
        if self.query_mat is None:
            print('No query file is set')
            return True
    
    def map_query(self, fasta_file, threads=1):
        self.query_mat, self.query_bounds, self.query_accs = self.mapper.direct(fasta_file, threads=threads, keep=True)
        self.fasta_file = fasta_file
        self.query_blast = self.mapper.blast_report
        self.query_mat_file = self.mapper.mat_file
        self.query_acc_file = self.mapper.acc_file
        
    def get_windows(self, metric='F1_score', min_overlap=0.9):
        if self.check_report() or self.check_query_map():
            print('Aborting')
            return
        
        min_cov = self.w_len * min_overlap
        max_ncov = self.w_len - min_cov
        
        windows = {}
        # read query_blast
        blast_tab = matrix.read_blast(self.query_blast)
        guide = matrix.make_guide(blast_tab.qseqid)
        # for each match, determine the first and last windows it overlaps with
        # low & high --> match bounds
        # n_low & n_high --> index of highest and lowest windows
        # max_ncov & min_cov --> max non covered space & min covered space
        # need to find n_low so that low - step * n_low <= max_ncov
        # need to find n_high so that high - step * n_high >= min_cov
        
        # lowest window above coverage:
            # n_low = ceil(abs(max_ncov - low) / step)
        # highest window above coverage:
            # n_cov = floor(abs(min_cov - high) / step)
        
        min_window = np.ceil(abs(max_ncov - blast_tab.sstart.values)/self.w_step)
        max_window = np.floor(abs(min_cov - blast_tab.send.values)/self.w_step)
        
        # get windows covered by each sequence and select the best one from the report
        for acc, idx0, idx1 in guide:
            seq_windows = np.concatenate([np.arange(min_win,max_win+1) for min_win, max_win in zip(min_window[idx0:idx1], max_window[idx0:idx1])])
            seq_windows *= self.w_step
            
            sub_report = self.report.loc[self.report.w_start.isin(seq_windows)]
            params = get_best_params(sub_report, metric)
            windows[acc] = params.iloc[0]
        self.windows = pd.DataFrame.from_dict(windows, orient='index')
    
    def hint_params(self, w_start, w_end, metric='F1_score'):
        if self.check_report():
            print('Aborting')
            return
        
        if w_end - w_start > self.w_len * 1.5:
            print(f'Warning: The provided window length {w_end - w_start} is more than 1.5 times the window length used in the calibration {self.w_len}\n\
                  Parameter hints may not be reliable. Recommend performing a calibration step for the desired window')
        sub_report = self.report.loc[(self.report.w_start >= w_start) & (self.report.w_end <= w_end)].sort_values(metric, ascending=False)
        
        taxguide = self.taxguide.set_index('taxID')
        
        missing = []
        params = []
        
        if len(self.taxa) > 0:
            # check for taxa not in window
            # if a given taxon is not in the window, search the lowest rank in its taxonomy found in the window
            taxes = []
            for tax, taxonomy in self.taxa.items():
                if taxonomy[0] in sub_report.taxon.values:
                    taxes.append(taxonomy[0])
                else:
                    for taxid in taxonomy:
                        if taxid in sub_report.taxon.values:
                            taxes.append(taxid)
                            missing.append((taxonomy[0], taxid))
                            break
            # get the best match for each taxon
            for tax in taxes:
                tax_report = sub_report.loc[sub_report.taxon == tax]
                param = get_best_params(tax_report)
                param = param.iloc[0]
                param['taxID'] = tax
                param['taxon'] = taxguide.loc[tax, 'SciName']
                param['rank'] = taxguide.loc[tax, 'rank']
                params.append(param)
        else:
            # no taxes are set, get the generic best for each window
            param = get_best_params(sub_report, metric)
            for win, win_param in param.groupby('w_start'):
                params.append(win_param.iloc[0])
        self.params = pd.concat(params)
    
    def get_overlap(self):
        overlap_low = max(self.query_bounds[0], self.ref_bounds[0])
        overlap_high = max(self.query_bounds[1], self.ref_bounds[1])
        overlap_len = max(0, overlap_high - overlap_low)
        
        if overlap_len == 0:
            print(f'Reference matrix coverage ({self.ref_bounds[0]} - {self.ref_bounds[1]}) does not overlap with the query matrix coverage ({self.query_bounds[0]} - {self.query_bounds[1]})')
            return False, None
        print(f'Reference and query matrices overlap in coordinates {overlap_low} - {overlap_high}')
        return True, [overlap_low, overlap_high]
    
    def set_dist_mat(self, identity=False, transition=1, transversion=2):
        if identity:
            self.dist_mat = cost_matrix.id_matrix()
        else:
            self.dist_mat = cost_matrix.cost_matrix(transition, transversion)
    
    def classify(self, w_start, w_end, k, n, mode='mwd', crop=True, site_rank='genus', out_path=None):
        if self.check_ref() or self.check_query():
            print('Aborting')
            return
        # set parameters
        w_start = list(w_start)
        if len(w_start) > 1:
            # multiple windows, w_end is the window length, get the end coordinates for each one
            w_end = [start + w_end for start in w_start]
        else:
            # single window, w_end is the ending coordinate
            w_end = [w_end]
        windows = np.array([w_start, w_end]).T
        k_range = list(k)
        n_range = list(n)
        
        # check valid windows
        check, overlap = self.get_overlap()
        if not check:
            print('Aborting')
            return
        windows = get_valid_windows(windows, overlap, crop)
        
        # account offset for query and reference
        query_offset = self.query_bounds[0] - self.ref_bounds
        
        final_report = []
        sites_report = []
        for coords in windows:
            # select sites
            n_sites = self.selector.get_sites(n_range, coords[0], coords[1], site_rank)
            # select data windows
            ref_window = self.loader.get_window(coords[0], coords[1], row_thresh=0, col_thresh=1)
            ref_offset = coords[0]
            
            prev_distances = np.zeros((self.query_mat.shape[0], ref_window.shape[0]), dtype = np.int8)
            for n, sites in n_sites.items():
                query_data = self.query_mat[:, sites - query_offset]
                
                ref_data = ref_window.eff_mat[:, sites - ref_offset]
                ref_tax = ref_window.eff_tax
                # get_distances                
                classifs, prev_distances = classification.classify(query_data, ref_data, ref_tax, self.dist_mat, k_range, mode, prev_distances)
                results = classification.get_classification(classifs)
                
                report = classification.parse_report(results)
                report['n'] = n
                report['start'] = coords[0]
                report['end'] = coords[1]
                final_report.append(report)
                
                sites_report.append(pd.Series({'w start':coords[0], 'w end':coords[1], 'n':sites}))
                
        final_report = pd.concat(final_report)
        final_report = tr_report(final_report, self.query_accs, self.ranks, self.taxguide)
        # TODO: incorporate cluster distances comparisons
        final_report.reset_index(drop=True, inplace=True)
        sites_report = pd.concat(sites_report)
        
        if out_path is None:
            return final_report, sites_report
        final_report.to_csv(f'{self.out_dir}/{out_path}.csv')
        sites_report.to_csv(f'{self.out_dir}/{out_path}.sites')
