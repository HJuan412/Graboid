#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 09:54:19 2022

@author: hernan
This script retrieves windows from a given data matrix and calculates entropy & entropy gain
"""

#%% libraries
from numba import njit
import numpy as np
import os
import pandas as pd
#%%
# filter matrix
def filter_rows(matrix):
    # keeps the index of all non empty rows
    filtered_idx = []
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
    cols = subtab.columns[1:]
    cons_tax = pd.Series(index=cols)
    for rk in cols:
        uniq_vals = subtab[rk].unique()
        if len(uniq_vals) == 1:
            cons_tax.at[rk] = uniq_vals[0]
    return cons_tax
        
# TODO: handle entropy calculation in a different
def entropy(matrix):
    n_cols = matrix.shape[1]

    entropy = np.zeros(n_cols)
    for i in range(n_cols):
        valid_rows = matrix[np.argwhere(matrix[:,i] != 16), i]
        n_rows = len(valid_rows)
        freqs = np.unique(valid_rows, return_counts = True)[1] / n_rows
        entropy[i] = -np.sum(np.log2(freqs) * freqs)
        
    return (2-entropy) / 2 # 1 min entropy, 0 max entropy

def per_tax_entropy(matrix, acc_tab, acc_list):
    n_cols = matrix.shape[1]
    ent_dict = {}
    for rk in acc_tab.columns:
        rank_col = acc_tab[rk]
        uniq_taxes = rank_col.unique()
        n_taxs = len(uniq_taxes)
        per_tax_ent = np.zeros((n_taxs, n_cols))

        for idx, tax in enumerate(uniq_taxes):
            tax_idx = rank_col.loc[rank_col == tax].index.tolist()
            sub_matrix = matrix[tax_idx]
            sub_entropy = entropy(sub_matrix)
            per_tax_ent[idx, :] = sub_entropy
        
        ent_dict[rk] = per_tax_ent
    return ent_dict
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

        if not self.dim_file is None:
            with open(self.acc_file, 'r') as handle:
                self.accs = handle.read().splitlines()
    
    def __set_matfile(self):
        if not self.mat_file is None:
            self.matrix = np.memmap(self.mat_file, dtype = np.int64, mode = 'r', shape = self.dims)
    
    def get_window(self, wstart, wend, thresh = 0.2):
        self.window = np.empty(0)
        self.rows = []
        
        if self.dims is None:
            return

        if wstart < 0 or wend > self.dims[1]:
            print(f'Invalid window dimensions: start:{wstart}, end:{wend}.\nMust be between 0 and {self.dims[1]}')
            return

        window = np.array(self.matrix[:, wstart:wend])
        # Windows are handled as a different class
        out_window = Window(window, wstart, wend, thresh, self)
        return out_window
    
    def load_acctab(self, row_list):
        filtered_accs = [self.accs[row] for row in row_list]
        tab = pd.read_csv(self.tax_file, index_col = 0)
        return tab.loc[filtered_accs].reset_index()

class Window():
    def __init__(self, matrix, wstart, wend, thresh = 0.2, loader = None):
        self.matrix = matrix
        self.wstart = wstart
        self.wend = wend
        self.width = wend - wstart
        self.loader = loader
        self.shape = (0,0)
        self.__preproc_window()
        self.process_window(thresh)
    
    def __preproc_window(self):
        rows = filter_rows(self.matrix)
        self.rows = rows
    
    def process_window(self, thresh):
        # run this method every time you want to change the column threshold
        if type(thresh) is not float:
            return
        if 0 >= thresh > 1:
            return
        self.thresh = thresh
        window = self.matrix[self.rows]
        self.min_seqs = window.shape[0] * (1-thresh)
        cols = filter_cols(window, self.thresh)
        self.window = window[:, cols]
        self.cols = cols
        self.shape = self.window.shape
        self.get_acctab()
        # collapse windows
        self.uniques = collapse_matrix(self.window) # Collapse the window, this is a list of OF LISTS of the indexes of unique sequences
        self.get_reps()
    
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
        cols = self.acc_tab.columns[1:]
        reps = list(self.reps.keys())
        cons_tax = pd.DataFrame(index = reps, columns = cols)
        
        for uq in reps:
            cons_mat.append(self.window[uq])
            if len(self.reps[uq]) == 1:
                cons_tax.at[uq] = self.acc_tab.loc[uq]
            else:
                subtab = self.acc_tab.loc[self.reps[uq]]
                cons_tax.at[uq] = build_cons_tax(subtab)
        self.cons_mat = np.array(cons_mat)
        self.cons_tax = cons_tax
        

#%% test
if __name__ == '__main__':
    wl = WindowLoader('nematoda', '18s', 'nematoda_18s/out_dir', 'nematoda_18s/out_dir', 'nematoda_18s/tmp_dir', 'nematoda_18s/warn_dir')
    window = wl.get_window(200, 300)