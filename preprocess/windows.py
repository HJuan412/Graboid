#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 09:54:19 2022

@author: hernan
This script retrieves windows from a given data matrix and calculates entropy & entropy gain
"""

#%% libraries
import numba as nb
import numpy as np
import os
import pandas as pd
from classif import cost_matrix
#%%
# filter matrix
def filter_matrix(matrix, thresh = 1, axis = 0):
    # filter rows (axis = 0) or columns (axis = 1) according to a maximum number of empty values
    n_lines = matrix.shape[axis]
    n_places = matrix.shape[[1,0][axis]]
    filtered_idx = []
    max_empty = thresh * n_places

    for idx in np.arange(n_lines):
        if axis == 0:
            line = matrix[idx, :]
        else:
            line = matrix[:, idx]

        vals, counts = np.unique(line, return_counts = True)
        if not 16 in vals:
            filtered_idx.append(idx)
            continue
        empty_idx = np.argwhere(vals == 16)
        if counts[empty_idx] < max_empty:
            filtered_idx.append(idx)
    
    return np.array(filtered_idx).astype(int)

def filter_rows(matrix):
    # keeps the index of all non empty rows
    filtered_idx = []
    n_rows, n_cols = matrix.shape

    for idx, row in enumerate(matrix):
        vals, counts = np.unique(row, return_counts = True)
        if len(vals) == 1 and vals[0] == 16:
            continue
        filtered_idx.append(idx)
    return filtered_idx

def filter_cols(matrix, thresh = 0.2):
    # filter columns by a minimum of empty rows (given as a percentage by thresh)
    filtered_idx = []
    n_rows, n_cols = matrix.shape
    max_empty = n_rows * thresh

    for i in range(n_cols):
        vals, counts = np.unique(matrix[:,i], return_counts = True)
        if not 16 in vals:
            filtered_idx.append(i)
            continue
        empty_idx = np.argwhere(vals == 16)
        if counts[empty_idx] <= max_empty:
            filtered_idx.append(i)
    
    return filtered_idx

# collapse unique rows
#%%
@nb.njit
def get_slice(arr, idx):
    # this function slices a 2d array in numba (numba can't handle regular slicing)
    n = len(idx)
    k = arr[0].shape[0]
    sliced = np.zeros((n,k), dtype = np.int32)

    for iidx, i in enumerate(idx):
        sliced[iidx] = arr[i]

    return sliced

@nb.njit
def get_val_idx(col):
    # get the indexes for each unique value in the column
    col_values = np.unique(col)
    col_values = col_values[col_values != 16]
    missing = np.argwhere(col == 16).flatten()
    indexes = []
    for value in col_values:
        val_idxs = np.argwhere(col == value).flatten()
        indexes.append(np.concatenate((val_idxs, missing)))
    return indexes

@nb.njit
def get_effective_seqs(matrix=np.array([[]]), row_idx=None, col_idx=0):
    # returns a list of indexes from each cluster of unique sequences
    # set initial rows
    if row_idx is None:
        row_idx = np.arange(matrix.shape[0])
    uniq_seqs = [[0]] # this is used to define the datatype (lists of integers) to be used in uniq_seqs (keeps numba from bitching)
    sub_mat = get_slice(matrix, row_idx)
    uniq_val_idxs = get_val_idx(sub_mat[:, col_idx])
    
    for val_idx in uniq_val_idxs:
        row_idx1 = row_idx[val_idx].flatten()
        # stop conditions: last column reached or only one row remaining
        if len(row_idx1) == 1 or col_idx + 1 == matrix.shape[1]:
            uniq_seqs.append(list(row_idx1))
            continue
        uniq_seqs += get_effective_seqs(matrix, row_idx1, col_idx + 1)
    
    return uniq_seqs[1:] # slice result to omit the initial 0 when defining uniq_seqs
# run the function to pre-compile it
TEST = np.array([[1,2,3],[1,2,3],[1,2,4]])
get_effective_seqs(TEST)
def build_cons_tax(subtab):
    # recieves a taxonomc subtable of all the integrants of a sequence cluster
    cols = subtab.columns[1:] # drop accession column
    cons_tax = pd.Series(index=cols, dtype = int)
    for idx, rk in enumerate(cols):
        uniq_vals = subtab[rk].unique()
        if len(uniq_vals) == 1:
            # if there are no conflicting taxonomies at the given rank (rk), assign it as the consensus taxon
            # set the lower values as the current unconflicting taxon
            cons_tax.iloc[idx:] = uniq_vals[0]
    return cons_tax

def get_reps(array):
    # generates a dictionary of representatives for each cluster of effective sequences
    reps = {}
    for uq in array:
        if len(uq) == 1:
            reps[uq[0]] = uq
        else:
            reps[uq[0]] = uq
    return reps

def collapse(matrix, tax_tab):
    # directs construction of the collapsed matrix and taxonomy table
    effective_seqs = get_effective_seqs(matrix)
    reps = get_reps(effective_seqs)
    collapsed_mat = matrix[list(reps.keys())]
    collapsed_tax = pd.DataFrame(index = reps.keys(), columns = tax_tab.columns[1:], dtype = int)
    
    for rep, clust in reps.items():
        if len(clust) == 1:
            collapsed_tax.at[rep] = tax_tab.loc[rep] # add the representative's taxonomy to the consensus tab (if it is the only member of the group)
        else:
            # build consensus taxonomy for all the integrants of the cluster
            subtab = tax_tab.loc[clust]
            collapsed_tax.at[rep] = build_cons_tax(subtab)

    return collapsed_mat, collapsed_tax.astype(int)

#%% new function to get effective seqs better, stronger faster
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

def get_effective_seqs_3(matrix):
    # uses the numba get_ident function for distance calculation
    # removes clustered sequences from the search
    seqs = set(np.arange(matrix.shape[0]))
    repeated = set()
    for idx0, seq0 in enumerate(matrix):
        if idx0 in repeated:
            continue
        for idx1, seq1 in enumerate(matrix[idx0+1:]):
            if (idx1 + idx0 + 1) in repeated:
                continue
            ident = get_ident(seq0, seq1)
            if ident:
                repeated.add(idx0)
                repeated.add(idx1 + idx0 + 1)
    effective_seqs = np.array(list(seqs.difference(repeated)))
    return effective_seqs
#%% classes
class WindowLoader():
    def __init__(self, taxon, marker, in_dir, out_dir, tmp_dir, warn_dir):
        self.taxon = taxon
        self.marker = marker
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.tmp_dir = tmp_dir
        self.warn_dir = warn_dir
        self.warnings = []
        self.prefix = f'{in_dir}/{taxon}_{marker}'
        self.__get_files()
        self.__load_metafiles()
        self.__set_matfile()
        self.window = np.empty(0)
    
    def __get_files(self):
        suffixes = ['.npy', '.dims', '.acclist', '.taxid']
        
        files = [None, None, None, None]
        
        for idx, sfx in enumerate(suffixes):
            filename = f'{self.prefix}{sfx}'
            if not os.path.isfile(filename):
                self.warnings.append(f'WARNING: file {filename} not found in {self.in_dir}')
                continue
            files[idx] = filename

        self.mat_file = files[0]
        self.dim_file = files[1]
        self.acc_file = files[2]
        self.tax_file = files[3]
    
    def __load_metafiles(self):
        self.dims = None
        self.accs = None

        if not self.dim_file is None:
            self.dims = tuple(pd.read_csv(self.dim_file, sep = '\t', header = None).iloc[0])

        if not self.acc_file is None:
            with open(self.acc_file, 'r') as handle:
                self.accs = handle.read().splitlines()
    
    def __set_matfile(self):
        if not self.mat_file is None:
            self.matrix = np.memmap(self.mat_file, dtype = np.int64, mode = 'r', shape = self.dims)
    
    def get_window(self, wstart, wend, row_thresh = 0.2, col_thresh = 0.2):
        self.window = np.empty(0)
        self.rows = []
        
        if self.dims is None:
            return

        if wstart < 0 or wend > self.dims[1]:
            print(f'Invalid window dimensions: start:{wstart}, end:{wend}.\nMust be between 0 and {self.dims[1]}')
            return

        window = np.array(self.matrix[:, wstart:wend])
        # Windows are handled as a different class
        out_window = Window(window, wstart, wend, row_thresh, col_thresh, self)
        return out_window
    
    def load_acctab(self, row_list):
        filtered_accs = [self.accs[row] for row in row_list]
        tab = pd.read_csv(self.tax_file, index_col = 0)
        return tab.loc[filtered_accs].reset_index()

class Window():
    def __init__(self, matrix, wstart, wend, row_thresh = 0.2, col_thresh = 0.2, loader = None):
        self.matrix = matrix
        self.wstart = wstart
        self.wend = wend
        self.width = wend - wstart
        self.loader = loader
        self.shape = (0,0)
        self.process_window(row_thresh, col_thresh)
    
    def process_window(self, row_thresh, col_thresh):
        # run this method every time you want to change the column threshold
        self.row_thresh = row_thresh
        self.col_thresh = col_thresh
        
        # filter columns first
        # cols = filter_matrix(self.matrix, col_thresh, axis = 1)
        # rows = filter_matrix(self.matrix[:,cols], row_thresh, axis = 0)
        
        # fitler rows first
        rows = filter_matrix(self.matrix, row_thresh, axis = 0)
        cols = filter_matrix(self.matrix[rows], col_thresh, axis = 1)
        window = self.matrix[rows][:, cols]
        self.min_seqs = window.shape[0] * (1-col_thresh)
        self.min_sites = window.shape[1] * (1-row_thresh)
        self.cols = cols
        self.rows = rows
        self.shape = window.shape
        self.window = window
        self.get_taxtab()
        # collapse windows
        self.cons_mat, self.cons_tax = collapse(self.window, self.tax_tab)

    def get_taxtab(self):
        # load the accession table for the current window (depends on the window loader existing)
        self.tax_tab = None
        if self.loader is None:
            return
        self.tax_tab = self.loader.load_acctab(self.rows)
    
    def get_col_idxs(self):
        return np.array(self.cols) + self.wstart