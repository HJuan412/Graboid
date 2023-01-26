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
from DATA import DATA
from glob import glob
import logging
from mapping import blast
from mapping import director as mp
import numpy as np
from preprocess import feature_selection as fsele
from preprocess import windows
import pandas as pd
import re
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
        self.mapper = mp.Director(tmp_dir, warn_dir)
    
    @property
    def ref_mat(self):
        return self.loader.matrix
    @property
    def ref_bounds(self):
        return self.loader.bounds
    
    def set_train_data(self, data_dir):
        # locate the training files (matrix, accession list, taxonomy table, information scores) needed for classification
        mat_file = glob(data_dir + '/*__map.npz')[0]
        tax_file = glob(data_dir + '/*.tax')[0]
        acc_file = glob(data_dir + '/*__map.accs')[0]
        guide_file = glob(data_dir + '/*.taxguide')[0]
        order_file = data_dir + '/order.npz'
        diff_file = data_dir + 'diff.csv'
        
        # set the loader with the learning data
        self.loader.set_files(mat_file, acc_file, tax_file)
        # load the taxguide
        self.taxguide = pd.read_csv(guide_file, index_col=0)
        # load information files
        self.selector.load_order_mat(order_file)
        self.selector.load_diff_tab(diff_file)
    
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

#%% main body
def main0(work_dir, fasta_file, database, overwrite_map=False, evalue=0.005, dropoff=0.05, min_height=0.1, min_width=2, dist_mat=None, threads=1):
    # generate dirs
    tmp_dir = work_dir + '/tmp'
    warn_dir = work_dir + '/warning'
    # locate database
    if not database in DATA.DBASES:
        print(f'Database {database} not found.')
        print('Current databases include:')
        for db, desc in DATA.DBASE_LIST.items():
            print(f'\tDatabase: {db} \t:\t{desc}')
        raise Exception('Database not found')
    db_dir = DATA.DATAPATH + '/' + database
    
    # map fasta file
    # see if fasta file is already present, if it is, skip mapping
    fasta = re.sub('.*/', '', fasta_file)
    try:
        DATA.MAPS[database][fasta]
        map_exists = True
    except KeyError:
        map_exists = False
    
    if map_exists and not overwrite_map:
        # use existing map of fasta file
        map_file = DATA.MAPS[database][fasta]['map']
        acc_file = DATA.MAPS[database][fasta]['acc']
    else:
        # map of fasta file doesn't exist create it
        map_director = mp.Director(work_dir, warn_dir)
        map_director.direct(fasta_file = fasta_file,
                            db_dir = db_dir,
                            evalue = evalue,
                            dropoff=dropoff,
                            min_height=min_height,
                            min_width=min_width,
                            threads = threads)
        map_file = map_director.map_file
        acc_file = map_director.acc_file
        # update map record
        new_maps = DATA.MAPS.copy()
        new_maps.update({fasta:{'map':map_file, 'acc':acc_file}})
        DATA.update_maps(new_maps)
    
    classifier = Director(work_dir, tmp_dir, warn_dir)
    classifier.set_train_data(db_dir)
    # designate classsification params
    classifier.set_dist_mat(dist_mat)
    # classify
    return
def main(w_start=0, w_end=-1, k=1, n=0, cl_mode='knn', rank=None, out_file='', query_file='', query_name='', ref_dir='', dist_mat=None, out_dir='', tmp_dir='', warn_dir='', threads=1):
    # main classification function: requires
    # query data (map)
    # reference data (map)
    # k
    # n
    # weight method (knn, wknn, dwknn)
    # dist matrix (id, k2p)
    # out file
    
    classifier = Director(out_dir, tmp_dir, warn_dir)
    # get reference data
    try:
        classifier.set_train_data(ref_dir)
        # TODO: method should return True if successful, False if not, use that instead of try, except
    except KeyError:
        print('No db directory found in')
        return
    # handle query data

    classifier.set_query(query_file, query_name, threads)
    classifier.set_dist_mat(dist_mat)
    classifier.classify(w_start, w_end, k, n, cl_mode, rank, out_file)
    
    return

if __name__ == '__main__':
    pass