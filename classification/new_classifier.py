#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:25:00 2023

@author: hernan

Classifier class, handles steps: database loading, query blasting, custom calibration, parameter selection, classification and reporting
"""

#%% libraries
from datetime import datetime
import json
import logging
import numpy as np
import os
import pandas as pd
import shutil
# Graboid libraries
from calibration import cal_calibrator
from classification import cost_matrix
from classification import cls_classify
from classification import cls_distance
from classification import cls_neighbours
from classification import cls_preprocess
from classification import cls_report

from DATA import DATA
from mapping import director as mp

#%% set logger
logger = logging.getLogger('Graboid.Classification')
logger.setLevel(logging.DEBUG)

#%% functions
def map_query(out_dir, warn_dir, fasta_file, db_dir, evalue=0.005, dropoff=0.05, min_height=0.1, min_width=2, threads=1):
    map_director = mp.Director(out_dir, warn_dir, logger)
    map_director.direct(fasta_file = fasta_file,
                        db_dir = db_dir,
                        evalue = evalue,
                        dropoff = dropoff,
                        min_height = min_height,
                        min_width = min_width,
                        threads = threads,
                        keep = True)
    map_file = os.path.abspath(map_director.mat_file)
    acc_file = os.path.abspath(map_director.acc_file)
    blast_report = os.path.abspath(map_director.blast_report)
    acc_list = map_director.accs
    bounds = map_director.bounds
    matrix = map_director.matrix
    coverage = map_director.coverage
    mesas = map_director.mesas
    return map_file, acc_file, blast_report, acc_list, bounds, matrix, coverage, mesas

def get_mesas_overlap(ref_mesas, qry_mesas, min_width=10):
    # array of columns [ol_start, ol_end, ol_width, ref_cov, qry_cov]
    mesas_overlap = []
    for q_mesa in qry_mesas:
        # get reference mesas that overlap with the current query mesa: (ref mesa start < qry mesa end) & (ref mesa end > qry mesa start)
        overlap_idxs = (ref_mesas[:,0] <= q_mesa[1]) & (ref_mesas[:,1] >= q_mesa[0])
        overlapping_mesas = np.clip(ref_mesas[overlap_idxs, :2], q_mesa[0], q_mesa[1])
        overlapping_widths = overlapping_mesas[:,1] - overlapping_mesas[:,0]
        # build overlapping matrix for current q_mesa
        q_overlap = np.zeros((len(overlapping_mesas), 5))
        q_overlap[:, :2] = overlapping_mesas
        q_overlap[:, 2] = overlapping_widths
        q_overlap[:, 3] = ref_mesas[overlap_idxs, 3]
        q_overlap[:, 4] = q_mesa[3]
        
        mesas_overlap.append(q_overlap[q_overlap[:,2] >= min_width]) # append only overlaps over the specified minimum width
    return np.concatenate(mesas_overlap, 0)

def select_window_params(window_dict, rep_column):
    # select parameters of taxa with values in rep_column above 0 (null or 0 value are worthless)
    # for win, win_dict in params.items():
    #     for tax in taxa:
    #         for combo, tx in win_dict.items():
    #             if tax in tx:
    #                 tax_params[win].update({tax:combo})
    #                 params_per_win[win].add(combo)
    return

## parameter selection
def select_params_per_window(params, report, **kwargs):
    report_taxa = report.index.get_level_values(1)
    tax_idx = [tx.upper() for tx in report.index.get_level_values(1)]
    # specify rank and/or taxa
    if 'rank' in kwargs.keys():
        rank = kwargs['rank'].lower()
        report_ranks = report.index.get_level_values(0)
        if not rank in report_ranks:
            raise Exception(f'Specified rank {rank} not found among: {" ".join(report_ranks)}')
        report_taxa = report.loc[rank].index.values
        tax_idx = [tx.upper() for tx in report_taxa]
        
    if 'taxa' in kwargs.keys():
        taxa = list(kwargs['taxa'])
        upper_taxa = set([upp for upp in map(lambda x : x.upper, taxa)])
        upper_rep_taxa = set([upp for upp in map(lambda x : x.upper, report_taxa)])
        tax_idx = set(upper_rep_taxa).intersection(upper_taxa)
        if len(tax_idx) == 0:
            raise Exception('None of the given taxa: {' '.join(taxa)} found in the database')
    
    report_cp = report.droplevel(level=0)
    report_cp.index = [tx.upper() for tx in report_cp.index]
    report_cp = report_cp.loc[tax_idx]
    # takes a parameters dictionary and a taxa list
    params_per_win = {win:set() for win in params.keys()}
    tax_params = {win:{} for win in params.keys()}
    
    for win, rep_column in report_cp.T.iterrows():
        win_params, win_tax_params = select_window_params(params[win], rep_column)
        params_per_win[win] = win_params
        tax_params[win] = win_tax_params

    return params_per_win, tax_params

def get_param_tab(keys, ranks):
    tuples = [k for k in keys if isinstance(k, tuple)]
    param_tab = pd.DataFrame(tuples, columns='n k m'.split())
    param_tab = pd.compat([param_tab, pd.DataFrame(data=False, index=param_tab.index, columns=ranks, dtype=bool)], axis=1)
    param_tab = param_tab.set_index(['n', 'k', 'm'])
    param_tab = param_tab.sort_index(level=[0,1])
    return param_tab

def get_params_ce(report, ranks):
    # select the best parameter combinations for each rank using the cross entropy metric
    # returns:
        # pandas dataframe with columns: rank, window, w_start, w_end, n, k, method, score
    
    selected_params = []
    for rk in ranks:
        # get the best (minimum) score for rk, retrieve parameter combinations that yield it
        min_ce = report[rk].min()
        params_subtab = report.loc[report[rk] == min_ce, ['window', 'w_start', 'w_end', 'n', 'k', 'method', rk]].copy()
        params_subtab.rename(columns={rk:'score'}, inplace=True)
        params_subtab['rank'] = rk
        selected_params.append(params_subtab)
    
    selected_params = pd.concat(selected_params).reset_index(drop=True)
    
    # filter params
    # the basal rank will usually score 0 loss for all param combinations, select only combinations that yield good scores in lower ranks
    score0_tab = selected_params.loc[selected_params.score == 0].reset_index().set_index(['window', 'n', 'k', 'method']) # all combinations with 0 entropy
    next_best_tab = selected_params.loc[selected_params.score > 0] # all parameter combinations with cross entropy greater than 0
    
    filtered_idxs = []
    for params, params_subtab in next_best_tab.groupby(['window', 'n', 'k', 'method']):
        try:
            filtered_idxs.append(score0_tab.loc[params, 'index'])
        except KeyError:
            continue
        
    best_params = pd.concat((selected_params.loc[filtered_idxs], next_best_tab))['rank', 'window', 'w_start', 'w_end', 'n', 'k', 'method', 'score'] # reorganize columns
    return best_params, {} # empty dict used for compatibility with get_params_met

def get_params_met(taxa, report):
    # select the best parameter combinations for each tax in taxa using the given metrics report
    # returns:
        # table with columns: taxon, window, w_start, w_end, n, k, method, score
        # warning dictionary with tax:warning key:value
    
    
    best_params = []
    warnings = {}
    
    for tax in taxa:
        # locate occurrences of tax in the report. Generate a warning if tax is absent or its best score is 0
        tax_subtab = report.loc[report.taxon == tax]
        if tax_subtab.shape[0] == 0:
            warnings[tax] = f'Taxon {tax} not found in the given report'
            continue
        best_score = tax_subtab.score.max()
        if best_score == 0:
            warnings[tax] = f'Taxon {tax} had a null score. Cannot be detected in the current window.'
            continue
        best_params.append(tax_subtab.loc[tax_subtab.score == best_score, ['taxon', 'window', 'w_start', 'w_end', 'n', 'k', 'method', 'score']])
    
    best_params = pd.concat(best_params)
    return best_params, warnings

#%% classes
class Classifier:
    def __init__(self, out_dir, mat_code='s1v2', overwrite=False):
        self.db = None
        self.last_calibration = None
        self.query_file = None
        self.query_map_file = None
        self.query_acc_file = None
        self.set_cost_matrix(mat_code)
        self.set_outdir(out_dir, overwrite)
        self.update_meta()
    
    @property
    def meta(self):
        return {'db':self.db,
                'query_file':self.query_file,
                'query_map_file':self.query_map_file,
                'query_acc_file':self.query_acc_file,
                'last_calibration':self.last_calibration,
                'cost_matrix':self.mat_code}
        
    def update_meta(self):
        with open(self.out_dir + '/meta.json', 'w') as handle:
            json.dump(self.meta, handle)
    
    def set_cost_matrix(self, mat_code):
        self.mat_code = mat_code
        self.cost_matrix = cost_matrix.get_matrix(mat_code)
        
    def set_database(self, database):
        # verify that given database is valid and there isn't another database already set
        if not self.db is None and self.db != database:
            raise Exception(f'Working directory {self.out_dir} already has a graboid database set: {self.db}. To use a different database, select a different working directory or overwrite the current one.')
        try:
            self.db_dir = DATA.get_database(database)
        except Exception as excp:
            raise excp
        # log database and update meta
        if self.db is None:
            # only update db attribute and meta when the database is set for the first time
            self.db = database
            self.update_meta()
        # use meta file from database to locate necessary files
        with open(self.db_dir + '/meta.json', 'r') as meta_handle:
            db_meta = json.load(meta_handle)
        
        # get database reference sequence
        self.db_reffile = db_meta['reference']
        self.db_refpath = db_meta['ref_file']
        self.db_refdir = '/'.join(self.db_refpath.split('/')[:-1]) # TODO: include the refdir location in the database meta file
        # load taxonomy guides
        self.guide = pd.read_csv(db_meta['guide_file'], index_col=0)
        self.tax_ext = pd.read_csv(db_meta['expguide_file'], index_col=0)
        self.ranks = self.tax_ext.columns.tolist()
        
        # load matrix & accession codes
        map_npz = np.load(db_meta['mat_file'])
        self.ref_matrix = map_npz['matrix']
        self.ref_mesas = map_npz['mesas']
        self.ref_coverage = map_npz['coverage']
        self.max_pos = self.ref_matrix.shape[1]
        with open(db_meta['acc_file'], 'r') as handle:
            self.ref_accs = handle.read().splitlines()
        
        # build extended taxonomy
        tax_tab = pd.read_csv(db_meta['tax_file'], index_col=0).loc[self.ref_accs]
        # the tax_tab attribute is the extended taxonomy for each record
        self.tax_tab = self.tax_ext.loc[tax_tab.TaxID.values]
        self.tax_tab.index = tax_tab.index
    
    # load and map query file
    def set_query(self, query_file, evalue=0.005, dropoff=0.05, min_height=0.1, min_width=2, threads=1):
        if self.db is None:
            raise Exception('You must set a Graboid databse before loading a query file.')
        if not self.query_file is None and query_file != self.query_file:
            # a query is already set, raise a warning
            raise Warning(f'Attempted to set {query_file} as query over existing one {self.query_file}. To use a different query, use a different working directory or overwrite the current one')
        # map query to the same reference sequence of the database
        map_file, acc_file, blast_report, acc_list, bounds, matrix, coverage, mesas = map_query(self.query_dir,
                                                                                                self.warn_dir,
                                                                                                query_file,
                                                                                                self.db_refdir,
                                                                                                evalue,
                                                                                                dropoff,
                                                                                                min_height,
                                                                                                min_width,
                                                                                                threads)
        self.query_file = query_file
        self.query_map_file = map_file
        self.query_acc_file = acc_file
        self.query_blast_report = blast_report
        self.query_accs = acc_list
        self.query_bounds = bounds
        self.query_matrix = matrix
        self.query_coverage = coverage
        self.query_mesas = mesas
        self.update_meta()
        
    def set_outdir(self, out_dir, overwrite=False):
        self.out_dir = out_dir
        self.calibration_dir = out_dir + '/calibration'
        self.classif_dir = out_dir + '/classification'
        self.query_dir = out_dir + '/query'
        self.tmp_dir = out_dir + '/tmp'
        self.warn_dir = out_dir + '/warnings'
        
        if overwrite:
            try:
                shutil.rmtree(out_dir)
            except FileNotFoundError:
                pass
        if os.path.isdir(out_dir):
            # out dir already exists
            try:
                with open(out_dir + '/meta.json', 'r') as handle:
                    meta = json.load(handle)
                    self.db = meta['db']
                    self.set_database(meta['db'])
                    self.last_calibration = meta['last_calibration']
                    self.query_file = meta['query_file']
                    self.query_map_file = meta['query_map_file']
                    self.query_acc_file = meta['query_acc_file']
                    self.load_query()
            except FileNotFoundError:
                raise Exception('Specified output directory exists but cannot be verified as a Graboid classification directory. Recommend overwrtiting it or using a different name')
                # TODO: maybe include the option of verifying if it is a classif dir with a damaged/mising meta file
        else:
            os.mkdir(out_dir)
            # make a directory to store classification reports
            os.makedirs(self.calibration_dir)
            os.makedirs(self.classif_dir)
            os.makedirs(self.query_dir)
            os.makedirs(self.tmp_dir)
            os.makedirs(self.warn_dir)
    
    def load_query(self):
        if self.query_file is None:
            return
        query_npz = np.load(self.query_map_file)
        self.query_bounds = query_npz['bounds']
        self.query_matrix = query_npz['matrix']
        self.query_coverage = query_npz['coverage']
        self.query_mesas = query_npz['mesas']
        with open(self.query_acc_file, 'r') as handle:
            self.query_accs = handle.read().splitlines()
        
    # locate overlapping regions between reference and query maps
    def get_overlaps(self, min_width=10):
        self.overlapps = get_mesas_overlap(self.ref_mesas, self.query_mesas, min_width)
    
    # custom calibrate
    def custom_calibrate(self,
                         max_n,
                         step_n,
                         max_k,
                         step_k,
                         row_thresh,
                         col_thresh,
                         min_seqs,
                         rank,
                         min_n,
                         min_k,
                         criterion,
                         threads=1,
                         **kwargs):
        """Perform custom calibration for the reference database, parameters
        are the same as those used to direct the grid search.
        If the user provides w_starts and w_ends coordinates as kwargs, use
        those. Otherwise, use the selected overlaps.
        Calibration results are stored to a subfolder inside the calibration
        directory in the working dir, by default named using datetime, unless
        the user provides cal_dir kwarg as an alternative name.
        Updates last_calibration parameter with the output directory"""
        # set calibrator
        calibrator = cal_calibrator.Calibrator()
        calibrator.set_database(self.db)
        
        if 'w_starts' in kwargs.keys() and 'w_ends' in kwargs.keys():
            calibrator.set_custom_windows(kwargs['w_starts'], kwargs['w_ends'])
        elif hasattr(self, 'overlapps'):
            calibrator.set_custom_windows(self.overlapps[:,0], self.overlapps[:,1])
        else:
            raise Exception('Missing parameters to set calibration windows. Run the get_overlapps method to get overlapping sections of query and reference data or provide matching sets of custom start and end positions')
        # set calibration directory
        try:
            cal_dir = kwargs['cal_dir']
        except KeyError:
            cal_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
        calibrator.set_outdir(self.calibration_dir + '/' + cal_dir)
        
        calibrator.grid_search(max_n,
                               step_n,
                               max_k,
                               step_k,
                               self.cost_matrix,
                               row_thresh,
                               col_thresh,
                               min_seqs,
                               rank,
                               min_n,
                               min_k,
                               criterion,
                               collapse_hm=True,
                               threads=threads)
        self.last_calibration = calibrator.out_dir
        self.update_meta()
    
    # select parameters
    # def select_params(self, metric, cal_dir=None, **kwargs):
    #     # specify taxa
    #     tax_idxs = self.guide.index.to_numpy()
    #     use_all = False
    #     if 'rank' in kwargs.keys():
    #         rank = kwargs['rank'].lower()
    #         if not rank in self.ranks:
    #             raise Exception(f'Specified rank {rank} not found among: {" ".join(self.ranks)}')
    #     if 'taxa' in kwargs.keys():
    #         taxa = list(kwargs['taxa'])
    #         upper_taxa = set([upp for upp in map(lambda x : x.upper, taxa)])
    #         upper_guide = [upp for upp in map(lambda x : x.upper, self.guide.SciName.tolist())]
    #         rev_guide = self.guide.reset_index()
    #         rev_guide.index = upper_guide
    #         tax_intersect = set(upper_guide).intersection(upper_taxa)
    #         if len(tax_intersect) == 0:
    #             raise Exception('None of the given taxa: {' '.join(taxa)} found in the database')
    #         tax_idxs = rev_guide.loc[tax_intersect.TaxID].to_numpy()
    #     else:
    #         # no rank or taxa list was provided, use all parameter combination winners
    #         use_all = True
    #     # locate calibration directory
    #     if cal_dir is None:
    #         cal_dir = self.last_calibration
    #     elif not cal_dir in os.listdir(self.calibration_dir):
    #         raise Exception(f'Specified calibration directory not found in {self.calibration_dir}')
        
    #     # open calibration and params reports
    #     cal_report = pd.read_csv(cal_dir + f'/calibration_{metric}.csv', index_col=[0,1], header=[0,1]).fillna(-1)
    #     cal_report.columns = pd.MultiIndex.from_arrays([cal_report.columns.get_level_values(0).astype(int), cal_report.columns.get_level_values(1).astype(int)]) # calibration report headers are read as string tuples, must convert them back to strings of ints
    #     with open(cal_dir + f'/params_{metric}.pickle', 'rb') as handle:
    #         params_dict = pickle.load(handle)
        
    #     # verify that current database is the same as the calibration report's
    #     if self.db != params_dict['db']:
    #         self.set_database(params_dict['db'])
        
    #     win_params = {col:[] for col in cal_report.columns}
    #     # select parameters combinations for each window
    #     for win, win_col in cal_report.T.iterrows():
    #         accepted = win_col.loc[win_col > 0].index
    #         rejected = win_col.loc[win_col <= 0].index.get_level_values(1)
    #     return
    
    
    # classify using different parameter combinations, register which parameter 
    def classify(self,
                 w_start,
                 w_end,
                 n,
                 k,
                 rank,
                 row_thresh,
                 col_thresh,
                 min_seqs,
                 criterion='orbit',
                 method='wknn',
                 save=True,
                 save_dir=''):
        method = method.lower()
        methods = {'unweighted':cls_classify.unweighted,
                   'wknn':cls_classify.wknn,
                   'dwknn':cls_classify.dwknn,
                   'rara':cls_classify.wknn_rara}
        
        # verify that method is valid
        try:
            classif_method = methods[method.lower()]
        except KeyError:
            raise Exception(f'Invalid method: {method}. Must be one of: "unweighted", "wknn", "dwknn"')
        # collapse reference and query matrices for the given window coordinates w_start & w_end, selecting the n most informative sites for the given rank
        ref_window, qry_window, qry_branches, win_tax, sites = cls_preprocess.collapse(self,
                                                                                       w_start = w_start,
                                                                                       w_end = w_end,
                                                                                       n = n,
                                                                                       rank = rank,
                                                                                       row_thresh = row_thresh,
                                                                                       col_thresh = col_thresh,
                                                                                       min_seqs = min_seqs)
        q_seqs = qry_window.shape[0]
        seqs_per_branch = np.array([len(br) for br in qry_branches])
        ref_mat = ref_window.window[:, sites]
        window_tax = self.tax_ext.loc[win_tax.index]
        
        # calculate distances
        distances = cls_distance.get_distances(qry_window, ref_mat, self.cost_matrix)
        
        # sort distances and get k nearest neighbours
        sorted_idxs = np.argsort(distances, axis=1)
        sorted_dists = np.sort(distances, axis=1)
        compressed = [np.unique(dist, return_index=True, return_counts = True) for dist in sorted_dists] # for each qry_sequence, get distance groups, as well as the index where each group begins and the count for each group
        # get k nearest orbital or orbital containing the kth neighbour
        if criterion == 'orbit':
            k_dists, k_positions, k_counts = cls_neighbours.get_k_nearest_orbit_V(compressed, k)
        else:
            k_dists, k_positions, k_counts = cls_neighbours.get_k_nearest_neigh_V(compressed, k)
        
        # assign classifications
        # clasif_id is a 2d array containing columns: query_idx, rank_idx, tax_id
        # classif_data is a 2d array containing columns: total_neighbours, mean_distances, std_distances, total_support and softmax_support
        classif_id, classif_data = cls_classify.classify_V(k_dists, k_positions, k_counts, sorted_idxs, window_tax, classif_method)
        
        #build reports
        pre_report = cls_report.build_prereport_V(classif_id, classif_data, seqs_per_branch)
        report = cls_report.build_report(pre_report, self.guide, q_seqs, seqs_per_branch) # TODO: remember to remove LinCode columns
        characterization = cls_report.characterize_sample(report)
        designation = cls_report.designate_branch_seqs(qry_branches, self.query_accs)
        
        # replace tax codes in pre report for their real names
        for rep in pre_report.values():
            rep['tax'] = self.guide.loc[rep.tax.values, 'SciName'].values
        
        if save:
            # save results to files
            # generate output directory
            out_dir = self.classif_dir + '/' + datetime.now().strftime("%Y%m%d_%H%M%S") if save_dir == '' else save_dir
            os.mkdir(out_dir)
            # save pre reports
            for rk, rk_prereport in pre_report.items():
                rk_prereport.to_csv(out_dir + f'/pre_report_{rk}.csv')
            report.to_csv(out_dir + '/report.csv')
            characterization.to_csv(out_dir + '/sample_characterization.csv')
            designation.to_csv(out_dir + '/sequence_designation.csv')
        else:
            return pre_report, report, characterization, designation
