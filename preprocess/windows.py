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
        self.min_seqs = window.shape[1] * (1-thresh)
        cols = filter_cols(window, self.thresh)
        self.window = window[:, cols]
        self.cols = cols
        self.shape = (len(self.rows), len(self.cols))
    
    def get_acctab(self):
        # load the accession table for the current window (depends on the window loader existing)
        self.acc_tab = None
        if self.loader is None:
            return
        self.acc_tab = self.loader.load_acctab(self.rows)

#%% test
if __name__ == '__main__':
    wl = WindowLoader('nematoda', '18s', 'nematoda_18s/out_dir', 'nematoda_18s/out_dir', 'nematoda_18s/tmp_dir', 'nematoda_18s/warn_dir')
    window = wl.get_window(0, 100)
    window.get_acctab()