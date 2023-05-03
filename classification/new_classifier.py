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
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import re
# Graboid libraries
from calibration import cal_calibrator as ccb
from classification import cost_matrix
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

def plot_ref_v_qry(ref_coverage, ref_mesas, qry_coverage, qry_mesas, overlapps, figsize=(12,7)):
    x = np.arange(len(ref_coverage)) # set x axis
    # plot ref
    fig, ax_ref = plt.subplots(figsize = figsize)
    ax_ref.plot(x, ref_coverage, label='Reference coverage')
    for r_mesa in ref_mesas.astype(int):
        mesa_array = np.full(len(ref_coverage), np.nan)
        # columns in the mesa arrays are [mesa start, mesa end, mesa width, mesa average height]
        mesa_array[r_mesa[0]:r_mesa[1]] = r_mesa[3]
        ax_ref.plot(x, mesa_array, c='r')
    
    # plot qry
    ax_qry = ax_ref.twinx()
    ax_qry.plot(x, qry_coverage, c='tab:orange')
    for q_mesa in qry_mesas.astype(int):
        mesa_array = np.full(len(ref_coverage), np.nan)
        # columns in the mesa arrays are [mesa start, mesa end, mesa width, mesa average height]
        mesa_array[q_mesa[0]:q_mesa[1]] = q_mesa[3]
        ax_qry.plot(x, mesa_array, c='g')
    
    # # plot overlaps
    # # vertical lines indicating overlapps between query and reference mesas
    for ol in overlapps:
        ol_height = max(ol[4], (ol[3] / ref_coverage.max()) * qry_coverage.max())
        
        ax_qry.plot(ol[[0,0]], [0, ol_height], linestyle=':', linewidth=1.5, c='k')
        ax_qry.plot(ol[[1,1]], [0, ol_height], linestyle=':', linewidth=1.5, c='k')
    # ol_x = overlapps[:, [0,0,1,1]].flatten() # each overlap takes two times the start coordinate and two times the end coordinate in the x axis (this is so they can be plotted as vertical lines)
    # ol_rheight = (overlapps[:, 3] / ref_coverage.max()) * qry_coverage.max() # transform the reference mesa height to the scale in the query axis
    # ol_y = np.array([overlapps[:,4], ol_rheight, overlapps[:,4], ol_rheight]).T.flatten() # get the ref and qry height of each overlape TWICE and interloped so we can plot th evertical lines at both ends of the overlap
    # ax_qry.plot(ol_x, ol_y, c='k')
    
    ax_ref.plot([0], [0], c='r', label='Reference mesas')
    ax_ref.plot([0], [0], c='tab:orange', label='Query coverage')
    ax_ref.plot([0], [0], c='g', label='Query mesas')
    ax_ref.plot([0], [0], linestyle=':', c='k', label='Overlapps')
    ax_ref.legend()
    # TODO: fix issues with mesa calculations
    # only filter out overlap coordinates when they appear too closely together (<= 20 sites) in the x axis
    ol_coords = np.unique(overlapps[:,:2])
    ol_coor_diffs = np.diff(ol_coords)
    selected_ol_coords = ol_coords[np.insert(ol_coor_diffs > 20, 0, True)]
    ax_qry.set_xticks(selected_ol_coords)
    ax_ref.set_xticklabels(selected_ol_coords.astype(int), rotation=70)
    
    ax_ref.set_xlabel('Coordinates')
    ax_ref.set_ylabel('Reference coverage')
    ax_qry.set_ylabel('Query coverage')
    
    # TODO: save plot

def select_window_params(window_dict, rep_column):
    # select parameters of taxa with values in rep_column above 0 (null or 0 value are worthless)
    # for win, win_dict in params.items():
    #     for tax in taxa:
    #         for combo, tx in win_dict.items():
    #             if tax in tx:
    #                 tax_params[win].update({tax:combo})
    #                 params_per_win[win].add(combo)
    return

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

# parameter selection
def get_param_tab(keys, ranks):
    tuples = [k for k in keys if isinstance(k, tuple)]
    param_tab = pd.DataFrame(tuples, columns='n k m'.split())
    param_tab = pd.compat([param_tab, pd.DataFrame(data=False, index=param_tab.index, columns=ranks, dtype=bool)], axis=1)
    param_tab = param_tab.set_index(['n', 'k', 'm'])
    param_tab = param_tab.sort_index(level=[0,1])
    return param_tab

#%% classes
class Classifier:
    def __init__(self, out_dir):
        self.db = None
        self.last_calibration = None
        self.set_outdir(out_dir)
        self.update_meta()
    
    def update_meta(self):
        with open(self.out_dir + '/meta.json', 'w') as handle:
            json.dump({'db':self.db, 'last_calibration':self.last_calibration}, handle)
    
    def read_meta(self):
        self.set_database(self.meta['db'])
        self.last_calibration = self.meta['last_calibration']
    
    def set_database(self, database):
        self.db = database
        try:
            self.db_dir = DATA.get_database(database)
        except Exception as excp:
            raise excp
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
        
        logger.info(f'Set database: {database}')
        self.update_meta()
    
    def set_outdir(self, out_dir):
        self.out_dir = out_dir
        self.calibration_dir = out_dir + '/calibration'
        self.classif_dir = out_dir + '/classification'
        self.tmp_dir = out_dir + '/tmp'
        self.warn_dir = out_dir + '/warnings'
        
        if os.path.isdir(out_dir):
            # out dir already exists
            try:
                with open(out_dir + '/meta.json', 'r') as handle:
                    self.meta = json.load(handle)
                    self.read_meta()
            except FileNotFoundError:
                raise Exception('Specified output directory exists but cannot be verified as a Graboid classification directory. Recommend overwrtiting it or using a different name')
                # TODO: maybe include the option of verifying if it is a classif dir with a damaged/mising meta file
        else:
            os.mkdir(out_dir)
            # make a directory to store classification reports
            os.makedirs(self.calibration_dir)
            os.makedirs(self.classif_dir)
            os.makedirs(self.tmp_dir)
            os.makedirs(self.warn_dir)
            self.meta = {}
        
    # load and map query file
    def set_query(self, query_file, evalue=0.005, dropoff=0.05, min_height=0.1, min_width=2, threads=1):
        if not hasattr(self, 'db'):
            raise Exception('No database set. Aborting')
        # map query to the same reference sequence of the database
        map_file, acc_file, blast_report, acc_list, bounds, matrix, coverage, mesas = map_query(self.tmp_dir,
                                                                                                self.warn_dir,
                                                                                                query_file,
                                                                                                self.db_refdir,
                                                                                                evalue,
                                                                                                dropoff,
                                                                                                min_height,
                                                                                                min_width,
                                                                                                threads)
        self.query_map_file = map_file
        self.query_acc_file = acc_file
        self.query_blast_report = blast_report
        self.query_accs = acc_list
        self.query_bounds = bounds
        self.query_matrix = matrix
        self.query_coverage = coverage
        self.query_mesas = mesas
    
    # locate overlapping regions between reference and query maps
    def get_overlaps(self, min_width=10):
        self.overlapps = get_mesas_overlap(self.ref_mesas, self.query_mesas, min_width)
    
    # custom calibrate
    def custom_calibrate(self,
                         max_n,
                         step_n,
                         max_k,
                         step_k,
                         mat_code,
                         row_thresh,
                         col_thresh,
                         min_seqs,
                         rank,
                         metric,
                         min_n,
                         min_k,
                         criterion,
                         threads=1,
                         **kwargs):
        # set calibrator
        calibrator = ccb.Calibrator()
        calibrator.set_database(self.db)
        
        if 'w_starts' in kwargs.keys() and 'w_ends' in kwargs.keys():
            calibrator.set_custom_windows(kwargs['w_starts'], kwargs['w_ends'])
        elif hasattr(self, 'overlapps'):
            calibrator.set_custom_windows(self.overlapps[:,0], self.overlapps[:,1])
        else:
            raise Exception('Missing parameters to set calibration windows. Run the get_overlapps method to get overlapping sections of query and reference data or provide matching sets of custom start and end positions')
        try:
            calibrator.set_outdir(self.calibration_dir + '/' + kwargs['cal_dir'])
        except KeyError:
            cal_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
            calibrator.set_outdir(self.calibration_dir + '/' + cal_dir)
        
        cost_mat = cost_matrix.get_matrix(mat_code)
        calibrator.grid_search(max_n,
                               step_n,
                               max_k,
                               step_k,
                               cost_mat,
                               row_thresh,
                               col_thresh,
                               min_seqs,
                               rank,
                               metric,
                               min_n,
                               min_k,
                               criterion,
                               threads)
        self.last_calibration = calibrator.out_dir
        
    # select parameters
    def select_params(self, metric, cal_dir=None, **kwargs):
        # specify taxa
        tax_idxs = self.guide.index.to_numpy()
        use_all = False
        if 'rank' in kwargs.keys():
            rank = kwargs['rank'].lower()
            if not rank in self.ranks:
                raise Exception(f'Specified rank {rank} not found among: {" ".join(self.ranks)}')
        if 'taxa' in kwargs.keys():
            taxa = list(kwargs['taxa'])
            upper_taxa = set([upp for upp in map(lambda x : x.upper, taxa)])
            upper_guide = [upp for upp in map(lambda x : x.upper, self.guide.SciName.tolist())]
            rev_guide = self.guide.reset_index()
            rev_guide.index = upper_guide
            tax_intersect = set(upper_guide).intersection(upper_taxa)
            if len(tax_intersect) == 0:
                raise Exception('None of the given taxa: {' '.join(taxa)} found in the database')
            tax_idxs = rev_guide.loc[tax_intersect.TaxID].to_numpy()
        else:
            # no rank or taxa list was provided, use all parameter combination winners
            use_all = True
        # locate calibration directory
        if cal_dir is None:
            cal_dir = self.last_calibration
        elif not cal_dir in os.listdir(self.calibration_dir):
            raise Exception(f'Specified calibration directory not found in {self.calibration_dir}')
        
        # open calibration and params reports
        cal_report = pd.read_csv(cal_dir + f'/calibration_{metric}.csv', index_col=[0,1], header=[0,1]).fillna(-1)
        cal_report.columns = pd.MultiIndex.from_arrays([cal_report.columns.get_level_values(0).astype(int), cal_report.columns.get_level_values(1).astype(int)]) # calibration report headers are read as string tuples, must convert them back to strings of ints
        with open(cal_dir + f'/params_{metric}.pickle', 'rb') as handle:
            params_dict = pickle.load(handle)
        
        # verify that current database is the same as the calibration report's
        if self.db != params_dict['db']:
            self.set_database(params_dict['db'])
        
        win_params = {col:[] for col in cal_report.columns}
        # select parameters combinations for each window
        for win, win_col in cal_report.T.iterrows():
            accepted = win_col.loc[win_col > 0].index
            rejected = win_col.loc[win_col <= 0].index.get_level_values(1)
        return
    
    # classify
    # classify using different parameter combinations, register which parameter 
    
    # report

#%%
# speed up distance calculation
# for each site in both the query and reference map, build dictionaries with key:values site_value:seq_indexes
# get distances between query_dict keys and ref_dict keys (at most #values ** 2)
# build distances array