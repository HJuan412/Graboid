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
import shutil
# Graboid libraries
from calibration import cal_calibrator as ccb
from classification import cost_matrix
from DATA import DATA
from mapping import director as mp
from preprocess import feature_selection as fsele
from preprocess import sequence_collapse as sq
from preprocess import windows as wn

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

## classification fucntions
def collapse(classifier, w_start, w_end, n, rank='genus', row_thresh=0.1, col_thresh=0.1, min_seqs=50):
    """Collapse reference and query windows"""
    # collapse reference window, use the classifier's extended taxonomy table
    ref_window = wn.Window(classifier.ref_matrix,
                           classifier.tax_tab,
                           w_start,
                           w_end,
                           row_thresh=row_thresh,
                           col_thresh=col_thresh,
                           min_seqs=min_seqs)
    print('Collapsed reference window...')
    # build the collapsed reference window's taxonomy
    win_tax = classifier.tax_ext.loc[ref_window.taxonomy][[rank]] # trick: if the taxonomy table passed to get_sorted_sites has a single rank column, entropy difference is calculated for said column
    # sort informative sites
    sorted_sites = fsele.get_sorted_sites(ref_window.window, win_tax) # remember that you can use return_general, return_entropy and return_difference to get more information
    sites = np.concatenate(fsele.get_nsites(sorted_sites[0], n = n))
    # collapse query window, use only the selected columns to speed up collapsing time
    qry_cols = np.arange(w_start, w_end)[sites] # get selected column indexes
    qry_matrix = classifier.query_matrix[:, qry_cols]
    qry_branches = sq.collapse_window(qry_matrix)
    qry_window = qry_matrix[[br[0] for br in qry_branches]]
    print('Collapsed query window...')
    return ref_window, qry_window, qry_branches, win_tax, sites

# distance calculation
def combine(window):
    """Creates a dictionary for each site (column) in the window, grouping all
    the sequences (rows) sharing the same base. Reduces the amount of operations
    needed for distance calculation"""
    combined = []
    for col in window.T:
        col_vals = np.unique(col)
        col_combined = {val:np.argwhere(col==val).flatten() for val in col_vals}
        combined.append(col_combined)
    return combined

def get_distances(qry_window, ref_window, cost_mat):
    """Generates a distance matrix of shape (# qry seqs, # ref seqs)"""
    # combine query and reference sequences to (greatly) speed up calculation
    qry_combined = combine(qry_window)
    ref_combined = combine(ref_window)
    
    dist_array = np.zeros((qry_window.shape[0], ref_window.shape[0]))
    
    # calculate the distances for each site
    for site_q, site_r in zip(qry_combined, ref_combined):
        # sequences sharing values at each site are grouped, at most 5*5 operations are needed per site
        for val_q, idxs_q in site_q.items():
            for val_r, idxs_r in site_r.items():
                dist = cost_mat[val_q, val_r]
                # update distances
                for q in idxs_q: dist_array[q, idxs_r] += dist
    return dist_array

# get nearest neighbours
# k_nearest is a list containing information about a (collapsed) query sequence's neighbours
# each element is a 3 element tuple containing:
    # an array with the distances to the k nearest orbitals (or up to the orbital containing the kth neighbour)
    # a 2d array containing the start and end positions of each included orbital, shape is (# orbitals, 2)
    # an array with the number of neighbours contained in each orbital
def get_k_nearest_orbit(compressed, k):
    k_nearest = []
    for seq in compressed:
        k_dists = seq[0][:k]
        k_indexes = seq[1][:k+1]
        k_positions = np.array([k_indexes[:-1], k_indexes[1:]]).T
        k_counts = seq[2][:k]
        k_nearest.append((k_dists, k_positions, k_counts))
    return k_nearest

def get_k_nearest_neigh(compressed, k):
    k_nearest = []
    for seq in compressed:
        summed = np.cumsum(seq[2])
        break_orb = np.argmax(summed >= k) + 1
        k_dists = seq[0][:break_orb]
        k_indexes = seq[1][:break_orb+1]
        k_positions = np.array([k_indexes[:-1], k_indexes[1:]]).T
        k_counts = seq[2][:break_orb]
        k_nearest.append((k_dists, k_positions, k_counts))
    return k_nearest

# classify functions
def unweighted(dists):
    return np.ones(len(dists))

def wknn(dists):
    d1 = dists[0]
    dk = dists[-1]
    if d1 == dk:
        return np.array([1])
    return (dk - dists) / (dk - d1)

def wknn_rara(dists):
    # min distance is always the origin, max distance is the furthest neighbour + 1, questionable utility 
    d1 = 0
    dk = dists[-1] + 1
    if d1 == dk:
        return np.array([1])
    return (dk - dists) / (dk - d1)

def dwknn(dists):
    d1 = dists[0]
    dk = dists[-1]
    penal = (dk + d1) / (dk + dists)
    return wknn(dists) * penal

def get_tax_supports(taxa, supports):
    """Retrieves an array of taxa with their corresponding supports.
    Calculate total support for each taxon, filter out taxa with no support.
    Sort taxa by decreasing order of support"""
    uniq_taxa = np.unique(taxa)
    supports = np.array([supports[taxa == u_tax].sum() for u_tax in uniq_taxa])
    uniq_taxa = uniq_taxa[supports > 0]
    supports = supports[supports > 0]
    order = np.argsort(supports)[::-1]
    # normalize supports
    norm_supports = np.exp(supports)[order]
    norm_supports = norm_supports / norm_supports.sum()
    return uniq_taxa[order], supports[order], norm_supports

def classify(neighbours, neigh_idxs, tax_tab, weight_func):
    """Take the output list of one of the get_k_nearest... functions, the array
    of sorted neighbour distances indexes, the extended taxonomy table for the
    reference window, and one of the weighting functions."""
    # each rank's classifications are stored in a different list
    # each list contains a tuple containing an array of taxIDs and another with their corresponding total supports
    # classifications are sorted by decresing order of support
    rank_classifs = {rk:[] for rk in tax_tab.columns}
    for seq, ngh_idxs in zip(neighbours, neigh_idxs):
        supports = weight_func(seq[0]) # calculate support for each orbital
        support_array = np.concatenate([[supp]*count for supp, count in zip(supports, seq[2])]) # extend supports to fit the number of neighbours per orbital
        # retrieve the neighbouring taxa
        all_neighs = ngh_idxs[seq[1][0,0]: seq[1][-1,-1]]
        neigh_taxa = tax_tab.iloc[all_neighs]
        # get total tax supports for each rank
        for rk, col in neigh_taxa.T.iterrows():
            valid_taxa = ~np.isnan(col.values)
            rank_classifs[rk].append(get_tax_supports(col.values[valid_taxa], support_array[valid_taxa]))
    return rank_classifs

# reporter functions
def build_prereport(classifications, branch_counts):
    # for each rank, generates a dataframe with columns taxa, supports, normalized supports and sequences in branch, with indexes corresponding to the classified query sequence (indexes repeat themselves)
    rk_tabs = {}
    for rk, seqs in classifications.items():
        rk_array = []
        for idx, seq in enumerate(seqs):
            seq_array = np.array((np.full(seq[0].shape, idx), seq[0], seq[1], seq[2], np.full(seq[0].shape, branch_counts[idx]))).T
            rk_array.append(seq_array)
        rk_array = np.concatenate(rk_array)
        rk_tab = pd.DataFrame({'seq':rk_array[:,0].astype(np.int32),
                               'tax':rk_array[:,1].astype(np.int32),
                               'support':rk_array[:,2].astype(np.float64),
                               'norm_support':rk_array[:,3].astype(np.float64),
                               'n_seqs':rk_array[:,4].astype(np.int32)})
        rk_tabs[rk] = rk_tab.set_index('seq')
    return rk_tabs

def build_report(pre_report, guide, q_seqs, seqs_per_branch):
    # guide is the classifier's (not extended) taxonomy guide
    # q_seqs is the number of query sequences
    # seqs_per_branch is the array containing the number of sequences collapsed into each q_seq
    # extract the winning classification for each sequence in the prereport
    guide = guide.copy()
    guide.loc[-1, 'SciName'] = 'Undefined'
    
    abv_reports = []
    for rk, rk_prereport in pre_report.items():
        # for each rank prereport, dessignate each sequence's classifciation as that with the SINGLE greatest support (if there is multiple top taxa, sequence is left ambiguous)
        # conclusion holds the winning classification for each sequence
        conclusion = np.full(q_seqs, -1, dtype=np.int32)
        
        # get index values and starting location of each index group
        seq_indexes, seq_loc, seq_counts = np.unique(rk_prereport.index.values, return_index=True, return_counts=True)
        single = seq_counts == 1 # get sequences with a single classification
        tax_array = rk_prereport.tax.values
        supp_array = rk_prereport.norm_support.values
        
        # get sequences with a single classification, those are assigned directly
        single = seq_counts == 1
        single_idxs = seq_indexes[single]
        single_locs = seq_loc[single]
        conclusion[single_idxs] = tax_array[single_locs]
        
        # get winning classifications for each sequence with support for multiple taxa
        for idx, loc in zip(seq_indexes[~single], seq_loc[~single]):
            if supp_array[loc] > supp_array[loc+1]:
                conclusion[idx] = tax_array[loc]
        
        # get the winning taxa's support
        clear_support = np.zeros(q_seqs)
        clear_support[seq_indexes] = supp_array[seq_loc]
        clear_support[conclusion < 0] = np.nan
        abv_reports.append(np.array([conclusion, clear_support]))
    
    # merge all abreviation reports
    abv_reports = np.concatenate(abv_reports, axis=0)
    header = pd.MultiIndex.from_product((pre_report.keys(), ['Taxon', 'support']))
    report = pd.DataFrame(abv_reports.T, columns=header)
    for rk in np.unique(report.columns.get_level_values(0)):
        tax_codes = report.loc[:, (rk, 'Taxon')].values
        report.loc[:, (rk, 'Taxon')] = guide.loc[tax_codes, 'SciName'].values
    # add sequence counts
    report[('n_seqs', 'n_seqs')] = seqs_per_branch
    return report

def characterize_sample(report):
    # report taxa present in sample, specifying rank, name, number of sequences and of branches, mean support and std support
    counts = []
    for rk in report.columns.get_level_values(0).unique()[:-1]:
        rk_report = report[[rk, 'n_seqs']].droplevel(0,1)
        rk_counts = []
        for tax, tax_report in rk_report.groupby('Taxon'):
            # get number of sequences, branches, mean support and std support for the current taxon
            total_seqs = tax_report.n_seqs.sum()
            n_branches = len(tax_report)
            mean_support = (tax_report.support * tax_report.n_seqs).sum() / total_seqs # account for the number of sequences in these calculations
            std_support = ((tax_report.support - mean_support)**2 * tax_report.n_seqs).sum() / total_seqs
            rk_counts.append((tax, total_seqs, n_branches, mean_support, std_support))
        rk_counts = pd.DataFrame(rk_counts, columns=['taxon', 'n_seqs', 'n_branches', 'mean_support', 'std_support'])
        rk_counts['rank'] = rk
        counts.append(rk_counts.set_index(['rank', 'taxon']))
    
    counts = pd.concat(counts)
    return counts

def designate_branch_seqs(branches, accs):
    br_array = np.concatenate([[br_idx]*len(br) for br_idx, br in enumerate(branches)])
    acc_idxs = np.concatenate(branches)
    acc_array = np.array(accs)[acc_idxs]
    designation = pd.DataFrame({'branch':br_array, 'accession':acc_array})
    return designation

#%% classes
class Classifier:
    def __init__(self, out_dir, overwrite=False):
        self.__db = None
        self.__last_calibration = None
        self.query_file = None
        self.query_map_file = None
        self.query_acc_file = None
        self.set_outdir(out_dir, overwrite)
        self.update_meta()
    
    @property
    def db(self):
        return self.__db
    @db.setter
    def db(self, db):
        self.__db = db
        self.update_meta()
    @property
    def last_calibration(self):
        return self.__last_calibration
    @last_calibration.setter
    def last_calibration(self, last_calibration):
        self.__last_calibration = last_calibration
        self.update_meta()
        
    def update_meta(self):
        with open(self.out_dir + '/meta.json', 'w') as handle:
            # json.dump({'db':self.db, 'last_calibration':self.last_calibration}, handle)
            self.meta = {'db':self.__db,
                         'query_file':self.query_file,
                         'query_map_file':self.query_map_file,
                         'query_acc_file':self.query_acc_file,
                         'last_calibration':self.__last_calibration}
            json.dump(self.meta, handle)
    
    def read_meta(self):
        self.set_database(self.meta['db'])
        self.__last_calibration = self.meta['last_calibration']
        self.query_file = self.meta['query_file']
        self.query_map_file = self.meta['query_map_file']
        self.query_acc_file = self.meta['query_acc_file']
        self.load_query()
    
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
            os.makedirs(self.query_dir)
            os.makedirs(self.tmp_dir)
            os.makedirs(self.warn_dir)
            self.meta = {}
        
    # load and map query file
    def set_query(self, query_file, evalue=0.005, dropoff=0.05, min_height=0.1, min_width=2, threads=1):
        if not hasattr(self, 'db'):
            raise Exception('You must set a Graboid databse before loading a query file.')
        if not self.query_file is None:
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
                         mat_code,
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
                               min_n,
                               min_k,
                               criterion,
                               collapse_hm=True,
                               threads=threads)
        self.last_calibration = calibrator.out_dir
    
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
                 mat_code,
                 criterion='orbit',
                 method='wknn',
                 save=True,
                 save_dir=''):
        method = method.lower()
        methods = {'unweighted':unweighted,
                   'wknn':wknn,
                   'dwknn':dwknn,
                   'rara':wknn_rara}
        if not method in methods.keys():
            raise Exception(f'Invalid method: {method}. Must be one of: "unweighted", "wknn", "dwknn"')
        # collapse reference and query matrices for the given window coordinates w_start & w_end, selecting the n most informative sites for the given rank
        ref_window, qry_window, qry_branches, win_tax, sites = collapse(self,
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
        cost_mat = cost_matrix.get_matrix(mat_code)
        distances = get_distances(qry_window, ref_mat, cost_mat)
        
        # sort distances and get k nearest neighbours
        sorted_idxs = np.argsort(distances, axis=1)
        sorted_dists = np.sort(distances, axis=1)
        compressed = [np.unique(dist, return_index=True, return_counts = True) for dist in sorted_dists] # for each qry_sequence, get distance groups, as well as the index where each group begins and the count for each group
        # get k nearest orbital or orbital containing the kth neighbour
        if criterion == 'orbit':
            k_nearest = get_k_nearest_orbit(compressed, k)
        else:
            k_nearest = get_k_nearest_neigh(compressed, k)
        
        # assign classifications
        classifications = classify(k_nearest, sorted_idxs, window_tax, methods[method])
        
        #build reports
        pre_report = build_prereport(classifications, seqs_per_branch)
        report = build_report(pre_report, self.guide, q_seqs, seqs_per_branch)
        characterization = characterize_sample(report)
        designation = designate_branch_seqs(qry_branches, self.query_accs)
        
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