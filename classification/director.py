#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 09:59:03 2022

@author: hernan
Director for the classification of sequences of unknown taxonomic origin
"""
#%%
import json
import logging
import numpy as np
import os
import pandas as pd
import re

from calibration import calibrator as clb
from calibration import reporter
from classification import classification
from classification import cost_matrix
from DATA import DATA
from mapping import director as mp
from preprocess import feature_selection as fsele
from preprocess import windows

#%% variables
mode_dict = {'m':'majority',
             'w':'wknn',
             'd':'dwknn'}

#%% set logger
cls_logger = logging.getLogger('Graboid.Classifier')
cls_logger.setLevel(logging.INFO)

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
        self.selector = fsele.Selector(tmp_dir)
    
    @property
    def ref_mat(self):
        return self.loader.matrix
    @property
    def ref_bounds(self):
        return self.loader.bounds
    @property
    def ref_cov(self):
        return self.loader.coverage
    @property
    def ref_mesas(self):
        return self.loader.mesas
    @property
    def ref_shape(self):
        return self.loader.dims
    @property
    def ref_tax(self):
        return self.loader.tax_tab
    @property
    def order_tab(self):
        return self.selector.order_tab
    @property
    def order_bounds(self):
        return self.selector.order_bounds
    @property
    def order_tax(self):
        return self.selector.order_tax
    @property
    def cost_mat(self):
        return self.__cost_mat
    @cost_mat.setter
    def cost_mat(self, mat_code):
        try:
            self.__cost_mat = cost_matrix.get_matrix(mat_code)
        except:
            raise
    
    def load_diff_tab(self, file):
        self.diff_tab = pd.read_csv(file, index_col = [0, 1])
    
    def set_train_data(self, meta):
        # locate the training files (matrix, accession list, taxonomy table, information scores) needed for classification
        mat_file = meta['mat_file']
        tax_file = meta['tax_file']
        acc_file = meta['acc_file']
        guide_file = meta['guide_file']
        order_file = meta['order_file']
        diff_file = meta['diff_file']
        
        # set the loader with the learning data
        self.loader.set_files(mat_file, acc_file, tax_file)
        # load the taxguide
        self.taxguide = pd.read_csv(guide_file, index_col=0)
        # load information files
        self.selector.load_order_mat(order_file)
        self.selector.load_diff_tab(diff_file)
    
    def set_query(self, map_file, acc_file):
        # load query files
        query_data = np.load(map_file)
        self.query_mat = query_data['matrix']
        self.query_bounds = query_data['bounds']
        self.query_cov = query_data['coverage']
        self.query_mesas = query_data['mesas']
        with open(acc_file, 'r') as acc_handle:
            self.query_accs = acc_handle.read().splitlines()
    
    def get_overlap(self):
        # retunrs overlaps array [overlap start, overlap end, query avg coverage, ref avg coverage]
        overlaps = []
        for qry in self.query_mesas:
            # get reference mesas that begin BEFORE qry ends and end AFTER qry starts
            ol_refs = self.ref_mesas[(self.ref_mesas[:,0] < qry[1]) & (self.ref_mesas[:,1] > qry[0])]
            for ref in ol_refs:
                overlaps.append([max(ref[0], qry[0]), min(ref[1], qry[1]), qry[3], ref[3]])
        self.overlaps = np.array(overlaps)
    
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

def map_query(out_dir, warn_dir, fasta_file, db_dir, evalue=0.005, dropoff=0.05, min_height=0.1, min_width=2, threads=1, logger=cls_logger):
    map_director = mp.Director(out_dir, warn_dir, logger)
    map_director.direct(fasta_file = fasta_file,
                        db_dir = db_dir,
                        evalue = evalue,
                        dropoff = dropoff,
                        min_height = min_height,
                        min_width = min_width,
                        threads = threads,
                        keep = False)
    map_file = os.path.abspath(map_director.mat_file)
    acc_file = os.path.abspath(map_director.acc_file)
    return map_file, acc_file
    
#%% main body
def main0(database, fasta_file, work_dir, overwrite=True, calibrate='yes', evalue=0.005, dropoff=0.05, min_height=0.1, min_width=2, dist_mat=None, max_k=15, step_k=2, max_n=30, step_n=5, threads=1, logger=cls_logger, *taxa):
    # locate database
    if not database in DATA.DBASES:
        print(f'Database {database} not found.')
        print('Current databases include:')
        for db, desc in DATA.DBASE_LIST.items():
            print(f'\tDatabase: {db} \t:\t{desc}')
        return
    db_dir = DATA.DATAPATH + '/' + database
    # load database meta
    with open(db_dir + '/meta.json', 'r') as meta_handle:
        db_meta = json.load(meta_handle)
    guide_tab = pd.read_csv(db_meta['guide_file'], index_col=0)

    # verify fasta
    # TODO: verify that fasta_file is a dasta file
    fasta = re.sub('.*/', '', fasta_file)
    
    # make directories
    map_dir = work_dir + '/maps'
    cal_dir = work_dir + '/calibrations'
    res_dir = work_dir + '/results'
    os.makedirs(map_dir, exist_ok = True)
    os.makedirs(cal_dir, exist_ok = True)
    os.makedirs(res_dir, exist_ok = True)
    
    # load local meta
    try:
        with open(work_dir + '/meta.json', 'r') as meta_handle:
            loc_meta = json.load(meta_handle)
    except FileNotFoundError:
        loc_meta = {'maps':{database:{fasta:{'map':None, 'acc':None}}},
                    'cal':{database:{fasta:{'report':None, 'meta':None, 'summ':None}}}}
    
    # build map
    make_map = overwrite
    try:
        map_file, acc_file = loc_meta['maps'][database][fasta]['map'], loc_meta['maps'][database][fasta]['acc']
        logger.info(f'Map file of fasta file {fasta_file} for database {database} located at {map_file}')
    except KeyError:
        make_map = True
    
    if make_map:
        # map of fasta file doesn't exist create it
        logger.info(f'Building a map for fasta file {fasta_file} versus database {database}')
        map_file, acc_file = map_query(map_dir, map_dir, fasta_file, db_meta['ref_dir'], evalue, dropoff, min_height, min_width, threads, logger)
        loc_meta['maps'][database][fasta]['map'] = map_file
        loc_meta['maps'][database][fasta]['acc'] = acc_file
    
    classifier = Director(res_dir, res_dir, res_dir)
    classifier.set_train_data(db_meta)
    classifier.set_query(map_file, acc_file)
    classifier.get_overlap()
    
    # build calibration
    make_cal = False
    try:
        cal_file, cal_meta, summ_files = loc_meta['cal'][database][fasta]['report'], loc_meta['cal'][database][fasta]['meta'], loc_meta['cal'][database][fasta]['summ']
        with open(cal_meta, 'r') as cal_meta_handle:
            cal_meta_data = json.load(cal_meta_handle)
        rk_dict = {}
        rk_dict = {rk:idx for idx, rk in enumerate(cal_meta_data['ranks'])}
        make_cal = calibrate == 'overwrite' # calibration files located but user wants to overwrite
        # summ files is a dicionary of the form {metric:[score file, param file]}
    except:
        make_cal = calibrate != 'no' # dont make calibration if user set as no
    
    if make_cal:
        calibrator = clb.Calibrator(cal_dir, cal_dir)
        calibrator.set_database(database)
        calibrator.set_windows(starts = classifier.overlaps[:,0], ends = classifier.overlaps[:,1])
        calibrator.dist_mat = dist_mat
        calibrator.grid_search(max_k, step_k, max_n, step_n)
        cal_tab = calibrator.report_file
        cal_meta = calibrator.meta_file
        summ_files = calibrator.build_summaries()
        loc_meta['cal'][database][fasta]['report'] = cal_tab
        loc_meta['cal'][database][fasta]['meta'] = cal_meta
        loc_meta['cal'][database][fasta]['summ'] = summ_files
        rk_dict = {rk:idx for idx, rk in enumerate(calibrator.ranks)}
        
    # get summary (or not)
    score_tab = pd.read_csv(summ_files['F1_score'][0], index_col=[0,1], header=[0,1])
    param_tab = pd.read_csv(summ_files['F1_score'][1], index_col=[0,1], header=[0,1])
    
    params, metrics = reporter.get_params(classifier.query_mesas, score_tab, param_tab, rk_dict, guide_tab, rank=None, *taxa)
    # classify
    return params, metrics

def main(work_dir, fasta_file, database, overwrite_map=False, calibration='yes', evalue=0.005, dropoff=0.05, min_height=0.1, min_width=2, dist_mat=None, max_k=15, step_k=2, max_n=30, step_n=5, threads=1, logger=cls_logger):
    # generate dirs
    tmp_dir = work_dir + '/tmp'
    warn_dir = work_dir + '/warning'
    cal_dir = work_dir + '/calibration'
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(warn_dir, exist_ok=True)
    # locate database
    if not database in DATA.DBASES:
        print(f'Database {database} not found.')
        print('Current databases include:')
        for db, desc in DATA.DBASE_LIST.items():
            print(f'\tDatabase: {db} \t:\t{desc}')
        return
    db_dir = DATA.DATAPATH + '/' + database
    # load meta_file
    with open(db_dir + '/meta.json', 'r') as meta_handle:
        meta = json.load(meta_handle)
    # map fasta file
    # see if fasta file is already mapped against the database, if it is, skip mapping, unless...
    # if overwrite_map is set as True, generate (evein if it is present)
    fasta = re.sub('.*/', '', fasta_file)
    make_map = overwrite_map
    try:
        prev_map = DATA.MAPS[database][fasta]
        # use existing map of fasta file
        map_file, acc_file = prev_map['map'], prev_map['acc']
        logger.info(f'Map file of fasta file {fasta_file} for database {database} located at {map_file}')
    except KeyError:
        make_map = True
    
    if make_map:
        # map of fasta file doesn't exist create it
        logger.info(f'Building a map for fasta file {fasta_file} versus database {database}')
        map_file, acc_file = map_query(work_dir, warn_dir, fasta_file, meta['ref_dir'], evalue, dropoff, min_height, min_width, threads, logger)
        # update map record
        DATA.add_map(database, fasta, map_file, acc_file, evalue, dropoff, min_height, min_width)
        
    classifier = Director(work_dir, tmp_dir, warn_dir)
    classifier.set_train_data(meta)
    classifier.set_query(map_file, acc_file)
    classifier.get_overlap()
    
    # calibration options: no, yes, overwrite
    make_calibrate = calibration == 'yes'
    if os.path.isdir(cal_dir):
        if calibration == 'overwrite':
            print(f'Removing existing calibration for database {database}...')
            # shutil.rmtree(cal_dir)
            make_calibrate = True
        
        
    os.mkdir(cal_dir)
    
    try:
        # calibration found, check if set to overwrite
        cal_tab = DATA.MAPS[database][fasta]['cal']
        cal_meta = DATA.MAPS[database][fasta]['cal_meta']
        make_calibrate = calibration == 'overwrite'
    except KeyError:
        # calibration not found, verify that one should be made
        make_calibrate = (calibration == 'yes') | (calibration == 'overwrite')
    if make_calibrate:
        calibrator = clb.Calibrator(work_dir, warn_dir)
        calibrator.set_database(database)
        calibrator.set_windows(starts = classifier.overlaps[:,0], ends = classifier.overlaps[:,1])
        calibrator.dist_mat = dist_mat
        calibrator.grid_search(max_k, step_k, max_n, step_n)
        cal_tab = calibrator.report_file
        cal_meta = calibrator.meta_file
        DATA.add_calibration(database, fasta, os.path.abspath(cal_tab), os.path.abspath(cal_meta))
        calibrator.build_summaries()
        
    if calibration == 'yes' or calibration == 'overwrite':
        # load the custom calibration and get optimum parameters from there
        pass
    else:
        # load generic calibration and get optimum parameters from there
        pass
    # designate classsification params
    # classifier.set_dist_mat(dist_mat)
    # # classify
    # return

if __name__ == '__main__':
    pass