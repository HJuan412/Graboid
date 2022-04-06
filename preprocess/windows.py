#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 09:54:19 2022

@author: hernan
This script retrieves windows from a given data matrix and calculates entropy & entropy gain
"""

#%% libraries
import numpy as np
import os
import pandas as pd
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
def get_val_idx(col):
    # get the indexes for each unique value in the column
    col_values = np.unique(col)
    val_idxs = [np.argwhere(col == value) for value in col_values]
    return val_idxs

def collapse_matrix(matrix, row_idx = None, col_idx = 0):
    # returns a list of indexes from each cluster of unique sequences
    # set initial rows
    if row_idx is None:
        row_idx = np.arange(matrix.shape[0])
    
    uniq_seqs = []
    sub_mat = matrix[row_idx]
    uniq_val_idxs = get_val_idx(sub_mat[:, col_idx])

    for val_idx in uniq_val_idxs:
        row_idx1 = row_idx[val_idx].flatten()
        # stop conditions: last column reached or only one row remaining
        if len(row_idx1) == 1 or col_idx + 1 == matrix.shape[1]:
            uniq_seqs.append(list(row_idx1))
            continue
        uniq_seqs += collapse_matrix(matrix, row_idx1, col_idx + 1)
    
    return uniq_seqs

def build_cons_tax(subtab):
    # recieves a taxonomc subtable of all the integrants of a sequence cluster
    cols = subtab.columns[1:] # drop accession column
    cons_tax = pd.Series(index=cols)
    for rk in cols:
        uniq_vals = subtab[rk].unique()
        if len(uniq_vals) == 1:
            # if there are no conflicting taxonomies at the given rank (rk), assign it as the consensus taxon, elsewhere leave it empty
            cons_tax.at[rk] = uniq_vals[0]
    return cons_tax

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
        self.get_acctab()
        # collapse windows
        self.uniques = collapse_matrix(self.window) # Collapse the window, this is a list of OF LISTS of the indexes of unique sequences
        self.get_reps()
        self.build_cons_mat()
    
    def get_acctab(self):
        # load the accession table for the current window (depends on the window loader existing)
        self.acc_tab = None
        if self.loader is None:
            return
        self.acc_tab = self.loader.load_acctab(self.rows)
    
    def get_col_idxs(self):
        return np.array(self.cols) + self.wstart
    
    def get_reps(self):
        reps = {}
        for uq in self.uniques:
            if len(uq) == 1:
                reps[uq[0]] = uq
            else:
                reps[uq[0]] = uq
        self.reps = reps

    def build_cons_mat(self):
        # this builds a matrix and taxid_table with unique sequences
        # in the case of repeated sequences, a consensus taxonomy is built, conflicting ranks are left blank
        cons_mat = []
        cols = self.acc_tab.columns[1:] # drop the accession column
        reps = list(self.reps.keys()) # representatives of each cluster
        cons_tax = pd.DataFrame(index = reps, columns = cols)
        
        for uq in reps:
            cons_mat.append(self.window[uq]) # append representative to the consensus matrix
            if len(self.reps[uq]) == 1:
                cons_tax.at[uq] = self.acc_tab.loc[uq] # add the representative's taxonomy to the consensus tab (if it is the only member of the group)
            else:
                # build consensus taxonomy for all the integrants of the cluster
                subtab = self.acc_tab.loc[self.reps[uq]]
                cons_tax.at[uq] = build_cons_tax(subtab)
        self.cons_mat = np.array(cons_mat)
        self.cons_tax = cons_tax
