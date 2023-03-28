#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 09:54:19 2022

@author: hernan
This script retrieves windows from a given data matrix and calculates entropy & entropy gain
"""

#%% libraries
import logging
import numba as nb
import numpy as np
import pandas as pd
import time

from preprocess import sequence_collapse as sq
#%%
# filter matrix
def filter_matrix(matrix, thresh = 1, axis = 0):
    # filter columns (axis = 0) or rows (axis = 1) according to a maximum number of empty values
    # return filtered indexes
    empty_cells = matrix == 0
    n_empty = np.sum(empty_cells, axis = axis)
    
    max_empty = thresh * matrix.shape[axis]
    filtered = np.argwhere(n_empty <= max_empty)
        
    return filtered.flatten()

def get_consensus_taxonomy(taxa):
    # return the consensus taxon at the highest possible level
    last_consensus = None
    for lvl in taxa.T:
        lvl_tax = np.unique(lvl)
        lvl_tax = lvl_tax[~np.isnan(lvl_tax)]
        if len(lvl_tax) == 0:
            # none of the sequences have a known tax for this level
            continue
        if len(lvl_tax) > 1:
            # multiple taxa found at this level, conflict
            return last_consensus
        # there is consensus at the current level
        last_consensus = lvl_tax[0]
    return last_consensus
#%% classes
class WindowLoader:
    def __init__(self, ranks, logger='WindowLoader'):
        self.ranks = ranks
        # logger set at initialization (because this class may be used by multiple modules)
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)
        
    def set_files(self, mat_file, acc_file, tax_file, guide_file):
        self.mat_file = mat_file
        self.acc_file = acc_file
        self.tax_file = tax_file
        self.guide_file = guide_file # should be the extended guide
        # load matrix
        try:
            matrix_data = np.load(mat_file)
        except ValueError:
            raise Exception(f'Error: matrix file {mat_file} is not a valid numpy file')
        self.matrix = matrix_data['matrix']
        self.matrix[self.matrix > 4] = 0 # TODO: fix this in mapper
        self.bounds = matrix_data['bounds']
        self.coverage = matrix_data['coverage']
        self.mesas = matrix_data['mesas']
        self.dims = self.matrix.shape
        # load acclist & tax tab
        with open(acc_file, 'r') as acc_handle:
                self.acclist = np.array(acc_handle.read().splitlines())
        
        # load the taxonomy table and guide file
        self.tax_tab = pd.read_csv(tax_file, index_col=0)
        self.tax_guide = pd.read_csv(guide_file, index_col=0)
    
    def get_window(self, row_thresh=0.2, col_thresh=0.2, **kwargs):
        start = 0
        end = self.dims[1]
        if 'start' in kwargs.keys():
            start = kwargs['start']
        if 'end' in kwargs.keys():
            end = kwargs['end']
        window_cols = np.arange(start, end)
        if 'cols' in kwargs.keys():
            window_cols = np.array(kwargs['cols'])
        if window_cols.min() < 0 or window_cols.max() > self.dims[1]:
            raise Exception(f'Invalid window dimensions: start: {window_cols.min()}, end: {window_cols.max()}. Must be between 0 and {self.dims[1]}')
        
        # filter out too incomplete rows and columns
        window_mat = self.matrix[:, window_cols]
        rows = filter_matrix(window_mat, row_thresh, axis = 1)
        cols = filter_matrix(window_mat[rows], col_thresh, axis = 0)
        
        # collapse redundant sequences
        if len(rows) > 0:
            branches = sq.collapse_window(window_mat[rows][:, cols])
            branch_indexes = [rows[br] for br in branches] # translate each branch's indexes to its original position in the alignment matrix
        else:
            self.logger.warning('No rows passed the threshold!')
        
        # get collapsed sequence taxonomy
        window_taxonomy = []

        for br in branch_indexes:
            branch_taxa = self.tax_tab.iloc[br]['TaxID'].values
            extended_taxa = self.tax_guide.loc[branch_taxa].to_numpy()
            window_taxonomy.append(get_consensus_taxonomy(extended_taxa))
        
        window_reprs = [br[0] for br in branch_indexes]
        window = window_mat[window_reprs][:, cols]
        
        # These two variables list the accession codes included in each branch and the column indexes (from the entire alignment) of the used columns
        # currently have no use to them, but leave them at hand just in case
        # branch_accs = [self.acclist[br] for br in branch_indexes]
        # columns = window_cols[cols]
        
        return window, window_taxonomy
    
    # OLD method
    # def get_window(self, start, end, row_thresh=0.2, col_thresh=0.2):
    #     if self.dims is None:
    #         return

    #     if start < 0 or end > self.dims[1]:
    #         raise Exception(f'Invalid window dimensions: start: {start}, end: {end}. Must be between 0 and {self.dims[1]}')

    #     # Windows are handled as a different class
    #     return Window(self.matrix, start, end, row_thresh, col_thresh, self)

class Window:
    def __init__(self, matrix, start, end, row_thresh=0.2, col_thresh=0.2, loader=None):
        self.matrix = matrix
        self.start = start
        self.end = end
        self.loader = loader
        self.window = None
        self.window_tax = None
        self.process_window(row_thresh, col_thresh)
    
    @property
    def tax_tab(self):
        if self.loader is None:
            return None
        return self.loader.tax_tab.iloc[self.rows]
    
    @property
    def tax_guide(self):
        if self.loader is None:
            return None
        return self.loader.tax_guide
    
    @property
    def ranks(self):
        if self.loader is None:
            return None
        return self.loader.ranks

    def process_window(self, row_thresh, col_thresh, safe=True):
        # generates a collapsed window using the specified row_thresh and col_thresh
        # attributes generated are self.window (collapsed window, numpy array) and self.window_tax (consensus taxonomy, segment of tax table, pandas dataframe)
        # run this method every time you want to change the column threshold
        self.row_thresh = row_thresh
        self.col_thresh = col_thresh
        
        
        # crop the portion of interest of the matrix
        matrix = self.matrix[:, self.start:self.end]
        # fitler rows first
        rows = filter_matrix(matrix, row_thresh, axis = 1)
        cols = filter_matrix(matrix[rows], col_thresh, axis = 0)
        
        self.rows = rows
        self.cols = cols + self.start
        self.shape = (len(rows), len(cols))
        # self.window = window
        if len(rows) > 0:
            self.collapse_window()
        else:
            self.loader.logger.warning('No rows passed the threshold!')
    
    def collapse_window(self):
        t0 = time.time()
        branches = sq.collapse_window(self.matrix[self.rows][:, self.cols])
        branch_taxonomy = get_consensus_taxonomy()
        elapsed = time.time() - t0
        self.loader.logger.debug(f'Collapsed window of size {self.shape}) in {elapsed:.3f} seconds')
        return
    
    # OLD
    # def collapse_window(self):
    #     t0 = time.time()
    #     # collapse the sequences of the selected rows and columns
    #     self.branches, seq_guide = seq_collapse_nb(self.matrix[self.rows][:, self.cols])
    #     self.window = build_collapsed(self.branches, seq_guide)
    #     self.n_seqs = len(self.branches)
    #     # generate the consensus taxonomy
    #     tax = []
    #     for branch in self.branches:
    #         accs = self.tax_tab.iloc[branch].index.values
    #         tax.append(build_effective_taxonomy(accs, self.tax_tab, self.tax_guide, self.ranks))
    #     self.window_tax = self.tax_guide.loc[tax]
    #     elapsed = time.time() - t0
    #     self.loader.logger.debug(f'Collapsed window of size {self.shape}) in {elapsed:.3f} seconds')
        

#%%
@nb.njit
def seq_collapse_nb(matrix):
    # matrix data
    base_range = matrix.max() + 1
    seq_len = matrix.shape[1]
    n_seqs = matrix.shape[0]
    # set up branch container and guide
    branches = [[0 for i in range(0)] for seq in range(n_seqs)]
    # branches is a list containing the indexes of all members of each generated branch
    # used to recover accession codes and generar consensus taxonomies after collapsing is done
    seq_guide = [[set([0 for i in range(0)]) for char in range(base_range)] for site in range(seq_len)]
    # seq_guide is a list of lists of shape (5 (unique bases + n), window length)
    # each cell indicates what EXISTING branches contain a given value in a given position
    # when a sequence is being evaluated, its values are used to check what existing branches it is compatible with
    # the number of available branches always decreases, if it reaches 0, the sequence generates a new branch
    # value 0 always contains all possible branches (can't be used to discard)
    all_branches = set([0 for i in range(0)])
    for site_idx in range(seq_len):
        seq_guide[site_idx][0] = all_branches
    # all_branches same variable referenced by position 0 of all sites in seq_guide
    # this ensures that missing values don't spawn new branches
    
    # initialize branch counter as 0
    n_branch = 0
    # make a single pass along the entire matrix
    for seq_idx, seq in enumerate(matrix):
        # all branches are possible at first
        possible_branches = all_branches
        # discard invalid branches by checking the seq_guide for each site
        for base_idx, base in enumerate(seq):
            # possible_branches is updated to keep only those elements already in it that have base in position base_idx
            # possible_branches always decreases
            merged = set([0 for i in range(0)])
            for b in seq_guide[base_idx][1:]:
                merged = merged.union(b)
            wrong_branches = merged.difference(seq_guide[base_idx][base])
            possible_branches = possible_branches.difference(wrong_branches)
            if len(possible_branches) == 0:
                # all existing branches discarded, this sequence represents a new branch
                # HOWEVER, maybe the current_branches had an unknown value at this place
                # possible_branches is empty, add the new branch n_branch
                # also update all_branches
                possible_branches.add(n_branch)
                all_branches.add(n_branch)
                # update counter
                n_branch += 1
                break
        # update branches and seq_guide
        # always do this in case an existing branch has incorporated new non ambiguous data
        for br in possible_branches:
            # if a new branch was defined, add the current seq_idx to it
            # if the current sequence could be added to an existing branch
            #   (the remaining value un possible_branches after completing the loop),
            #   add its seq_idx to that branch
            branches[br].append(seq_idx)
        for base_idx in np.argwhere(seq != 0).flatten():
            # add the current sequence's values to the seq_guide
            # this step should be done even if no new branch was generated
            #   as the newly incorporated sequence may have clarified missing values in the branch it incorporated to
            base = seq[base_idx]
            for br in possible_branches:
                seq_guide[base_idx][base].add(br)
    
    # return a list of branches (with the index of every belonging sequence)
    # return seq_guide (used to build the collapsed matrix)
    return branches[:n_branch], seq_guide

def build_collapsed(branches, guide):
    # generate collapsed map
    # ths step ensures the collapsed map contains the least amount of missing sites (choose to fill a site whenever possible)
    collapsed = np.zeros((len(branches), len(guide)), dtype=np.int16)
    for site, br_values in enumerate(guide):
        for base, seqs in enumerate(br_values[1:]):
            collapsed[list(seqs), site] = base + 1
    return collapsed

@nb.njit
def seq_collapse_nb0(matrix):
    # matrix data
    base_range = matrix.max() + 1
    seq_len = matrix.shape[1]
    n_seqs = matrix.shape[0]
    # set up branch container and guide
    branches = [[0 for i in range(0)] for seq in range(n_seqs)]
    # branches is a list containing the indexes of all members of each generated branch
    # used to recover accession codes and generar consensus taxonomies after collapsing is done
    seq_guide = [[set([0 for i in range(0)]) for char in range(base_range)] for site in range(seq_len)]
    # seq_guide is a list of lists of shape (5 (unique bases + n), window length)
    # each cell indicates what EXISTING branches contain a given value in a given position
    # when a sequence is being evaluated, its values are used to check what existing branches it is compatible with
    # the number of available branches always decreases, if it reaches 0, the sequence generates a new branch
    # value 0 always contains all possible branches (can't be used to discard)
    all_branches = set([0 for i in range(0)])
    for site_idx in range(seq_len):
        seq_guide[site_idx][0] = all_branches
    # all_branches same variable referenced by position 0 of all sites in seq_guide
    # this ensures that missing values don't spawn new branches
    
    # initialize branch counter as 0
    n_branch = 0
    # make a single pass along the entire matrix
    for seq_idx, seq in enumerate(matrix):
        # all branches are possible at first
        possible_branches = all_branches
        # discard invalid branches by checking the seq_guide for each site
        for base_idx, base in enumerate(seq):
            # possible_branches is updated to keep only those elements already in it that have base in position base_idx
            # possible_branches always decreases
            possible_branches = possible_branches.intersection(seq_guide[base_idx][base])
            if len(possible_branches) == 0:
                # all existing branches discarded, this sequence represents a new branch
                # possible_branches is empty, add the new branch n_branch
                # also update all_branches
                possible_branches.add(n_branch)
                all_branches.add(n_branch)
                # update counter
                n_branch += 1
                break
        # update branches and seq_guide
        # always do this in case an existing branch has incorporated new non ambiguous data
        for br in possible_branches:
            # if a new branch was defined, add the current seq_idx to it
            # if the current sequence could be added to an existing branch
            #   (the remaining value un possible_branches after completing the loop),
            #   add its seq_idx to that branch
            branches[br].append(seq_idx)
        for base_idx in np.argwhere(seq != 0).flatten():
            # add the current sequence's values to the seq_guide
            # this step should be done even if no new branch was generated
            #   as the newly incorporated sequence may have clarified missing values in the branch it incorporated to
            base = seq[base_idx]
            for br in possible_branches:
                seq_guide[base_idx][base].add(br)
    
    # generate collapsed map
    # ths step ensures the collapsed map contains the least amount of missing sites (choose to fill a site whenever possible)
    collapsed = np.zeros((n_branch, seq_len), dtype=np.int16)
    for site, br_values in enumerate(seq_guide):
        for base, branches in enumerate(br_values[1:]):
            collapsed[branches, site] = base + 1
    # get effective indexes
    eff_rows = [i[0] for i in branches[:n_branch]]
    return eff_rows, branches[:n_branch]