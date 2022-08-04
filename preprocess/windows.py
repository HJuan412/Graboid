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
from classif import cost_matrix

#%%
# filter matrix
def filter_matrix(matrix, thresh = 1, axis = 0):
    # filter columns (axis = 0) or rows (axis = 1) according to a maximum number of empty values
    # return filtered indexes
    empty_cells = matrix == 0
    n_empty = np.sum(empty_cells, axis = axis)
    
    max_empty = thresh * matrix.shape[axis]
    filtered = np.argwhere(n_empty <= max_empty)
        
    return filtered

#%% Old collapsing functions
# TODO: delete this entire cell once the new one is tested
# collapse unique rows
def build_cons_tax(subtab):
    # recieves a taxonomc subtable of all the integrants of a sequence cluster
    # TODO: NOTE: curent confilciting values are left as 0
    cols = subtab.columns[1:] # drop accession column
    cons_tax = pd.Series(index=cols, dtype = int)
    for idx, rk in enumerate(cols):
        uniq_vals = subtab[rk].unique()
        if len(uniq_vals) == 1:
            # if there are no conflicting taxonomies at the given rank (rk), assign it as the consensus taxon
            # set the lower values as the current unconflicting taxon
            cons_tax.iloc[idx:] = uniq_vals[0]
    return cons_tax

# new function to get effective seqs better, stronger faster
ID_MAT = cost_matrix.id_matrix()
ID_MAT[-1]=0
ID_MAT[:,-1]=0
@nb.njit
def get_ident(seq0, seq1):
    # similar to the calc_distance function but interrupts itself at the first difference
    for site0, site1 in zip(seq0, seq1):
        if ID_MAT[site0, site1] > 0:
            return False
    return True

#
SH_MAT = np.zeros((17,17), dtype = int)
SH_MAT[-1,:-1] = -1
SH_MAT[:-1,-1] = 1
# build roadmap
def build_roadmap(matrix):
    # build a map of the positions of each value in each column of the matrix
    roadmap = []
    for col in matrix.T:
        col_vals = {val:[] for val in np.unique(col)}
        for idx, val in enumerate(col):
            col_vals[val].append(idx)
        col_vals = {k:set(v) for k,v in col_vals.items()}
        roadmap.append(col_vals)
    return roadmap

# build nodes
# should start with idxs as set(np.arange(matrix.shape[0]))
def build_nodes(seq=[], idxs=set(), roadmap=[]):
    if len(roadmap) == 0:
        return [seq], [idxs]
    total_seqs = []
    total_idxs = []
    col_vals = roadmap[0]
    for val, val_idxs in col_vals.items():
        overlap = idxs.intersection(val_idxs) # if set is empty there is no overlap
        if len(overlap) == 0:
            continue
        new_seq = seq + [val]
        branch_seqs, branch_idxs = build_nodes(new_seq, overlap, roadmap[1:])
        total_seqs += branch_seqs
        total_idxs += branch_idxs
    return total_seqs, total_idxs

def collapse_0(matrix):
    # collapse identical sequences and keep their indexes
    roadmap = build_roadmap(matrix)
    init_idxs = set(np.arange(matrix.shape[0]))
    effective_seqs, effective_idxs = build_nodes(idxs=init_idxs, roadmap=roadmap)
    effs = np.array(effective_seqs)
    effi = [list(ei) for ei in effective_idxs]
    return effs, effi

def get_ident_matrix(eff_seqs):
    # build a matrix with the pairwise identity between the effective sequences
    # 1 : sequences have the same identity
    # 0 : sequences differ in at least one effective site
    ident_mat = np.zeros((eff_seqs.shape[0], eff_seqs.shape[0]))
    for idx0, seq0 in enumerate(eff_seqs):
        for idx1, seq1 in enumerate(eff_seqs[idx0+1:]):
            ident_mat[idx0, idx1+idx0+1] = get_ident(seq0, seq1)
    return ident_mat

def get_shscore(seq0, seq1):
    # get the shared score between sequences with the same identity, keep the one with less missing data
    # if shscore > 0 : seq0 is more complete than seq1
    # if shscore < 0 : seq1 is more complete than seq0
    # if shscore = 0 : seq1 and seq0 are equally complete but missing different sites
    score = 0
    for s0, s1 in zip(seq0, seq1):
        score += SH_MAT[s0, s1]
    # returns the index of the most incomplete sequence
    # TODO: what happens if there is a draw?
    return int(score > 0)

def compare_ident(ident, matrix):
    # score the pairs of sequences with identity, keep the most complete one in each case
    pairs = np.argwhere(ident == 1)
    scores = np.zeros(pairs.shape[0], dtype = int)
    for idx, pair in enumerate(pairs):
        seq0 = matrix[pair[0]]
        seq1 = matrix[pair[1]]
        scores[idx] = get_shscore(seq0, seq1)
    return pairs, scores

def get_winners(nseqs, pairs, scores):
    # return the indexes of the winners (sequences to keep)
    # losers (sequences to drop) are repeated sequences with more missing data
    winners = np.arange(nseqs, dtype = int)
    losers = np.zeros(len(pairs), dtype = int)
    for idx, (pair, sc) in enumerate(zip(pairs, scores)):
        losers[idx] = pair[sc]
    losers = np.unique(losers)
    winners = np.delete(winners, losers)
    return winners

def crop_effectives(effective_seqs, effective_idxs):
    # remove redundant sequences from the effective sequences cluster
    ident = get_ident_matrix(effective_seqs)
    pairs, scores = compare_ident(ident, effective_seqs)
    winners = get_winners(len(effective_seqs), pairs, scores)
    cropped_seqs = effective_seqs[winners]
    cropped_idxs = [effective_idxs[wn] for wn in winners]
    return cropped_seqs, cropped_idxs

def collapse_1(matrix, tax_tab):
    # directs construction of the collapsed matrix and taxonomy table
    effective_seqs, effective_idxs = collapse_0(matrix)
    cropped_seqs, cropped_idxs = crop_effectives(effective_seqs, effective_idxs)

    collapsed_tax = pd.DataFrame(index = np.arange(len(cropped_idxs)), columns = tax_tab.columns[1:], dtype = int)
    
    for idx, clust in enumerate(cropped_idxs):
        if len(clust) == 1:
            collapsed_tax.at[idx] = tax_tab.loc[clust[0]] # add the representative's taxonomy to the consensus tab (if it is the only member of the group)
        else:
            # build consensus taxonomy for all the integrants of the cluster
            subtab = tax_tab.loc[clust]
            collapsed_tax.at[idx] = build_cons_tax(subtab)
    
    collapsed_tax.reset_index(drop=True, inplace=True)
    return cropped_seqs, collapsed_tax.astype(int)

#%% New collapsing functions
# TODO: Test these
def build_effective_matrix(eff_idxs, matrix):
    # construct the effective sequence for each given custer
    effective_matrix = np.zeros((len(eff_idxs), matrix.shape[1]))
    for idx, cluster in enumerate(eff_idxs):
        # since unknown values are 0 and each column can have AT MOST two values
        # effective sequence is the maximum value for each column
        effective_matrix[idx] = np.max(matrix[cluster], axis = 0)
    return effective_matrix

def build_effective_taxonomy(eff_idxs, tax_tab):
    # construct the effective taxonomy for each cluster
    n_ranks = tax_tab.shape[1]
    effective_taxes = np.zeros((len(eff_idxs), n_ranks))
    # turn table to array, easier to handle
    tax_mat = tax_tab.to_numpy()
    for idx0, cluster in enumerate(eff_idxs):
        # prepare consensus taxonomy
        clust_tax = np.zeros(n_ranks)
        # transpose sub_mat to iterate trough columns
        sub_mat = tax_mat[cluster].T
        # previous unconflicting taxon, starts as 0
        p_tax = 0
        for idx, rank in enumerate(sub_mat):
            uniq_vals = np.unique(rank)
            if len(uniq_vals) > 1:
                # rank contains multiple taxon. Conflict, set this and all subsequent positions as last unconflicting taxon and break
                clust_tax[idx:] = p_tax
                break
            # no conflict, set taxon and update p_tax
            clust_tax[idx] = uniq_vals
            p_tax = uniq_vals
        # add completed taxonomy
        effective_taxes[idx0] = clust_tax
    effective_taxes = pd.DataFrame(effective_taxes, columns = tax_tab.columns)
    return effective_taxes

def collapse_window(matrix, tax_tab):
    tree = Tree()
    tree.build(matrix)
    
    eff_idxs = [lv[0] for lv in tree.leaves]
    eff_mat = build_effective_matrix(eff_idxs, matrix)
    eff_tax = build_effective_taxonomy(eff_idxs, tax_tab)
    return eff_mat, eff_tax
#%% classes
class WindowLoader:
    def __init__(self, logger = 'WindowLoader'):
        # logger set at initialization (because this class may be used by multiple modules)
        self.logger = logging.getLogger(logger)
        self.mat_file = None
        self.acc_file = None
        self.tax_file = None
        self.matrix = None
        self.dims = None
        self.acclist = None
        self.tax_tab = None
        
    def set_files(self, mat_file, acc_file, tax_file):
        self.mat_file = mat_file
        self.acc_file = acc_file
        self.tax_file = tax_file
        # load matrix
        self.matrix = np.load(mat_file)
        self.dims = self.matrix.shape
        # load acclist
        with open(acc_file, 'r') as acc_handle:
            self.acclist = acc_handle.read().splitlines()
        # load tax tab
        self.tax_tab = pd.read_csv(tax_file, index_col=0)
    
    def get_window(self, start, end, row_thresh=0.2, col_thresh=0.2):
        self.rows = []
        
        if self.dims is None:
            return

        if start < 0 or end > self.dims[1]:
            self.logger.warning(f'Invalid window dimensions: start: {start}, end: {end}. Must be between 0 and {self.dims[1]}')
            return

        window = np.array(self.matrix[:, start:end])
        # Windows are handled as a different class
        out_window = Window(window, start, end, row_thresh, col_thresh, self)
        return out_window

class Window:
    def __init__(self, matrix, start, end, row_thresh = 0.2, col_thresh = 0.2, loader = None):
        self.matrix = matrix
        self.start = start
        self.end = end
        self.loader = loader
        self.shape = (0,0)
        self.process_window(row_thresh, col_thresh)
        self.window = None
        self.cons_mat = None
        self.cons_tax = None
    
    @property
    def window(self):
        return self.matrix[self.rows][:, self.cols]
    
    @property
    def tax_tab(self):
        if self.loader is None:
            return None
        return self.loader.tax_tab.iloc[self.rows]
    
    @property
    def col_idxs(self):
        return self.cols + self.start

    def process_window(self, row_thresh, col_thresh):
        # run this method every time you want to change the column threshold
        self.row_thresh = row_thresh
        self.col_thresh = col_thresh
        
        # fitler rows first
        rows = filter_matrix(self.matrix, row_thresh, axis = 1)
        cols = filter_matrix(self.matrix[rows], col_thresh, axis = 0)
        
        self.rows = rows
        self.cols = cols
        self.shape = (len(rows), len(cols))
        # self.window = window
        
        # collapse windows
        # self.cons_mat, self.cons_tax = collapse_1(self.window, self.tax_tab)
        self.collapse_window()
    
    def collapse_window(self):
        tree = Tree()
        tree.build(self.window)
        
        self.eff_idxs = [lv[0] for lv in tree.leaves]
        self.eff_mat = build_effective_matrix(self.eff_idxs, self.window)
        self.eff_tax = build_effective_taxonomy(self.eff_idxs, self.tax_tab)
        

#%%
# TODO: test these
# TODO: maybe coud be changed into a numba function
class Tree:
    def build(self, matrix):
        self.leaves = []
        t_matrix = matrix.T
        Node(0, None, t_matrix[0], np.arange(t_matrix.shape[1]), t_matrix, self)

class Node:
    def __init__(self, lvl, value, row, indexes, matrix, tree):
        self.lvl = lvl
        self.row = row
        self.indexes = indexes
        self.matrix = matrix
        self.tree
        self.children = []
        if lvl + 1 < len(matrix):
            self.get_children()
        else:
            tree.leaves.append(indexes)
    
    def get_children(self):
        vals = np.unique(self.row)
        vals = vals[vals != 0]
        zeros = np.argwhere(self.row == 0).flatten()
        for val in vals:
            indexes = np.argwhere(self.row == val).flatten()
            indexes = np.concatenate((zeros, indexes))
            indexes = self.indexes[indexes]
            row = self.matrix[self.lvl + 1, indexes]
            self.children.append(Node(self.lvl + 1, self, val, row, indexes, self.matrix, self.tree))
