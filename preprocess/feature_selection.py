#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 12:32:25 2022

@author: hernan
Feature selection
"""

#%% modules
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd

#%% functions
# table manipulation
def get_taxid_tab(tax_file, mat_accs):
    # formats the given tax table (keeps only tax_id columns and removes the '_id' tail)
    # extract only the rows present in the alignment matrix, given by mat_accs
    tax_tab = pd.read_csv(tax_file, index_col=0)
    cols = [col for col in tax_tab if '_' in col]
    tr_dict = {col:col.split('_')[0] for col in cols}
    taxid_tab = tax_tab[cols].rename(columns = tr_dict)
    taxid_tab = taxid_tab.loc[mat_accs]
    return taxid_tab

# information quantification
@nb.njit
def get_entropy(array):
    valid_rows = array[array != 0]
    n_rows = len(valid_rows)
    values = np.unique(valid_rows)
    counts = np.array([(valid_rows == val).sum() for val in values])
    freqs = counts / n_rows
    return -np.sum(np.log2(freqs) * freqs, dtype=np.float32)

def get_matrix_entropy(matrix):
    entropy = np.zeros(matrix.shape[1], dtype=np.float32)
    for idx, col in enumerate(matrix.T):
        entropy[idx] = get_entropy(col)
    
    # maximum possible entropy is log2(num of classes)
    # fasta code has 15 possible classes (not counting gaps and missing values)
    # most frequently 4 clases (acgt), log2(4) = 2
    return (2-entropy) / 2 # 1 min entropy, 0 max entropy

def per_tax_entropy(matrix, tax_tab):
    # returns entropy_tab with multiindex (rank, tax) and ncols = matrix.shape[1]
    tabs = [] # will concatenate into entropy_tab at the end
    # iterate over every rank
    for rank, tax_col in tax_tab.T.iterrows():
        # get unique taxons in rank
        tax_list = tax_col.unique().tolist()
        # get the entropy for each taxon
        entropy_mat = np.zeros((len(tax_list), matrix.shape[1]), dtype=np.float32)
        for idx, tax in enumerate(tax_list):
            entropy_mat[idx] = get_matrix_entropy(matrix[tax_col == tax])
        tabs.append(pd.DataFrame(entropy_mat, index = pd.MultiIndex.from_product([[rank], tax_list])))
    # merge entropy tables for each taxon
    entropy_tab = pd.concat(tabs)
    return entropy_tab

def get_ent_diff(matrix, tax_tab):
    general_entropy = get_matrix_entropy(matrix)
    p_tax_ent = per_tax_entropy(matrix, tax_tab)
    diff_tab = p_tax_ent - general_entropy
    return diff_tab
    
def get_gain(matrix, tax_tab):
    gain_dict = {}
    rows, cols = matrix.shape

    for rk in tax_tab.columns:
        gain = np.zeros(cols)
        
        for col_idx in range(cols):
            col = matrix[:,col_idx]
            for val in np.unique(col):
                val_idxs = np.argwhere(col == val).flatten()
                tax_array = tax_tab.iloc[val_idxs,:].loc[:,rk].values
                
                gain[col_idx] += (len(val_idxs)/rows) * get_entropy(tax_array)
        
        gain_dict[rk] = gain
    
    gain_tab = pd.DataFrame.from_dict(gain_dict, orient = 'index')
    return gain_tab

def plot_gain(table, rank, criterium='diff', figsize=(7,10)):
    fig, ax = plt.subplots(figsize = figsize)
    
    sub_tab = table.loc[rank]
    x = sub_tab.shape[1]
    y = sub_tab.mean().to_numpy()
    y_std = sub_tab.std().to_numpy()
    
    ax.bar(x, y, yerr = y_std)
    ax.set_xlabel('Position')
    ax.set_ylabel('Entropy difference')
    ax.set_title(f'Entropy difference for rank {rank} ({len(sub_tab)} taxons)')
    ax.margins(x = 0.05, y = 0.01)

#%%
class Selector:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.order_file = f'{out_dir}/order.npz'
        self.diff_file = f'{out_dir}/diff.csv'
    
    def build_tabs(self, matrix, bounds, coverage, mat_accs, tax_file, min_seqs=0, rank='genus'):
        # filter taxa below the min_seqs sequence threshold at the given rank
        tax_tab = get_taxid_tab(tax_file, mat_accs)
        ranks = {rk:idx for idx, rk in enumerate(tax_tab.columns)}
        tax_counts = tax_tab[rank].value_counts()
        selected = tax_counts.loc[tax_counts >= min_seqs]
        taxa = selected.index.to_numpy(dtype = int)
        seqs = np.argwhere(tax_tab[rank].isin(taxa).values).flatten()
        
        sub_mat = matrix[seqs]
        sub_tax = tax_tab.iloc[seqs]
        
        # Quantify information per site per taxon per rank
        self.diff_tab = get_ent_diff(sub_mat, sub_tax)
        
        # Get ordered bases for each taxon
        taxons = []
        orders = []
        for rk, rk_idx in ranks.items():
            sub_diff = self.diff_tab.loc[rk]
            for idx, (tax, row) in enumerate(sub_diff.iterrows()):
                taxons.append([rk_idx, tax])
                orders.append(row.sort_values(ascending = False).index.values)
        
        # tax_tab is a 2-column array containing rank and taxID of each row in order_tab
        # order_tab is a matrix containing the sites ordered in function of decreasing entropy difference (firts elements are the most informative)
        # use tax_tab to locate the rows in order_tab that belong to a given rank
        self.order_tax = np.array(taxons, dtype = np.int32)
        self.order_tab = np.array(orders, dtype=np.int16)
        self.order_bounds = bounds
        
        # save data
        np.savez_compressed(self.order_file,
                            order = self.order_tab,
                            bounds = self.order_bounds,
                            taxs = self.order_tax)
        self.diff_tab.to_csv(self.diff_file)
    
    def load_order_mat(self, file):
        order_data = np.load(file)
        self.order_tab = order_data['order_tab']
        self.order_bounds = order_data['order_bounds']
        self.order_tax = order_data['order_tax']
    
    def load_diff_tab(self, file):
        self.diff_tab = pd.read_csv(file, index_col = [0, 1])
    
    def select_sites(self, nsites, rank, cols):
        # select the first nsites more informative sites from the given cols
        # returns a list with an array of column indexes per value of nsites
        # nsites number of sites to select, can be an iterable
        # rank (name) to select sites by
        # cols list of column indexes to select
        self.selected_rank = rank
        rk = self.ranks[rank]
        
        # get the portion of the matrix corresponding to the given rank rows, extract relevant cols
        rank_submat = self.order_tab[self.order_tav[:,0] == rk]
        cols_submat = np.zeros((len(rank_submat), len(cols)))
        for idx, row in enumerate(rank_submat):
            rk_cols = row[np.isin(row, cols)]
            cols_submat[idx] = rk_cols
        
        selected_sites = [np.unique(cols_submat[:n]) for n in list(nsites)]
        return selected_sites
    
    def get_sites(self, n_range, rank, cols=None, start=None, end=None):
        # for a given range of sites, generate a dictionary containing the new sites selected at each n
        # used for exploring multiple n values in calibration and classification, (avoids repeating calculations)
        sites = {}
        total_sites = np.array([], dtype = np.int8)
        for n in n_range:
            # TODO: test fix for site selection
            # n_sites = self.select_sites(start, end, n, rank)
            n_sites = self.select_sites(n, rank, cols)
            new_sites = n_sites[np.in1d(n_sites, total_sites, invert=True)]
            if len(new_sites) > 0:
                sites[n] = new_sites
                total_sites = np.concatenate([total_sites, new_sites])
        return sites
