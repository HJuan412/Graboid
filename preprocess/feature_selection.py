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
def get_taxid_tab(tax_tab):
    # formats the given tax table (keeps only tax_id columns and removes the '_id' tail)
    cols = [col for col in tax_tab if len(col.split('_')) > 1]
    tr_dict = {col:col.split('_')[0] for col in cols}
    taxid_tab = tax_tab[cols].rename(columns = tr_dict)
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
    def __init__(self):
        self.taxa = None
        self.seqs = None
        self.set_ranks()
        
    def set_matrix(self, matrix, bounds, tax_tab=None):
        self.matrix = matrix
        self.bounds = bounds
        if not tax_tab is None:
            self.tax_tab = get_taxid_tab(tax_tab)
    
    def filter_taxons(self, min_seqs=10, rank='genus'):
        tax_counts = self.tax_tab[rank].value_counts(ascending = False)
    
        selected = tax_counts.loc[tax_counts >= min_seqs]
        taxa = selected.index.to_numpy(dtype = int)
        seqs = np.argwhere(self.tax_tab[rank].isin(taxa).values).flatten()
        
        self.taxa = taxa
        self.seqs = seqs
    
    def build_tabs(self):
        # Get the entropy difference for each base/taxon/rank.
        # Uses filtered matrix if a filter has been applied
        # Define matrix and tax table to utilize
        if self.seqs:
            sub_mat = self.matrix[self.seqs]
            sub_tax = self.tax_tab.loc[self.seqs, self.taxa]
        else:
            sub_mat = self.matrix
            sub_tax = self.tax_tab
        
        # Quantify information per site per taxon per rank
        self.diff_tab = get_ent_diff(sub_mat, sub_tax)
        
        # Get ordered bases for each taxon
        taxons = []
        orders = []
        for rk, rk_idx in self.ranks.items():
            sub_diff = self.diff_tab.loc[rk]
            for idx, (tax, row) in enumerate(sub_diff.iterrows()):
                taxons.append([rk_idx, tax])
                orders.append(row.sort_values(ascending = False).index.values)
        
        # tax_tab is a 2-column array containing rank and taxID
        # order_tab is a matrix containing the sites ordered in function of decreasing entropy difference (firts elements are the most informative)
        self.order_tax = np.array(taxons, dtype = np.int32)
        self.order_tab = np.array(orders, dtype=np.int16)
        self.order_bounds = self.bounds
    
    def set_ranks(self, ranks = ['phylum', 'class', 'order', 'family', 'genus', 'species']):
        self.ranks = {rk:idx for idx, rk in enumerate(ranks)}
    
    def save_order_mat(self, out_file):
        with open(out_file, 'wb') as out_handle:
            np.savez(out_handle, order_tab = self.order_tab, order_bounds = self.order_bounds, order_tax = self.order_tax)
    
    def load_order_mat(self, file):
        order_data = np.load(file)
        self.order_tab = order_data['order_tab']
        self.order_bounds = order_data['order_bounds']
        self.order_tax = order_data['order_tax']
    
    def save_diff_tab(self, out_file):
        self.diff_tab.to_csv(out_file)
    
    def load_diff_tab(self, file):
        self.diff_tab = pd.read_csv(file, index_col = [0, 1])
    
    def select_sites(self, start, end, nsites, rank):
        # get the nsites more informative sites per taxon for the current rank
        # must run this AFTER generate diff_tab
        # should run this after select_taxons
        self.selected_rank = rank
        rk = self.ranks[rank]
        
        # get sites, first nsites columns in the order table
        sub_order = []
        for row in self.order_tab:
            sub_order.append(row[np.logical_and(row >= start, row <= end)])
        sub_order = np.array(sub_order)
        order_subtab = sub_order[self.order_tax[:,0] == rk,:nsites]
        # adjust offset, discard sites outside the matrix bounds
        selected_sites = np.unique(order_subtab)
        
        return selected_sites
    
        # self.sites = selected_sites
        # if self.seqs:
        #     selected_matrix = self.matrix[self.seqs, selected_sites]
        #     selected_tax = self.tax_tab.iloc[self.seqs]
        #     return selected_matrix, selected_tax
        
        # selected_matrix = self.matrix[:, selected_sites]
        # return selected_matrix
    
    def get_sites(self, n_range, start, end, rank):
        # for a given range of sites, generate a dictionary containing the new sites selected at each n
        # used for exploring multiple n values in calibration and classification, (avoids repeating calculations)
        sites = {}
        total_sites = np.array([], dtype = np.int8)
        for n in n_range:
            n_sites = self.select_sites(start, end, n, rank)
            new_sites = n_sites[np.in1d(n_sites, total_sites, invert=True)]
            if len(new_sites) > 0:
                sites[n] = new_sites
                total_sites = np.concatenate(total_sites, new_sites)
        return sites
    
class SelectorOLD:
    def __init__(self, matrix, tax):
        self.matrix = matrix
        self.tax = tax
        self.ranks = tax.columns.tolist()
        self.diff_tab = None
        self.order_tab = None
        # default selection is all the sites and sequences
        self.selected_tax = {rk:None for rk in self.ranks}
        self.selected_seqs = {rk:np.arange(matrix.shape[0]) for rk in self.ranks}
        self.selected_rank = None
        self.selected_sites = np.arange(matrix.shape[1])

    def build_diff_tab(self):
        # Get the entropy difference for each base/taxon/rank. Uses filtered matrix if a filter has been applied
        sub_mat = self.matrix[self.selected_seqs]
        sub_tax = self.tax.iloc[self.selected_seqs]
        self.diff_tab = get_ent_diff(sub_mat, sub_tax)
        
        taxons = []
        orders = []
        for rk_idx, rk in enumerate(self.diff_tab.index.levels[0]):
            sub_diff = self.diff_tab.loc[rk]
            for idx, (tax, row) in enumerate(sub_diff.iterrows()):
                taxons.append([rk_idx, tax])
                orders.append(row.sort_values(ascending = False).index.values)
        
        # tax_tab is a 2-column array containing rank and taxID
        # order_tab is a matrix containing the sites ordered in function of decreasing entropy difference (firts elements are the most informative)
        self.taxtab = np.array(taxons, dtype = np.int32)
        self.order_tab = np.array(orders, dtype=np.int16)
    
    def set_ranks(self, ranks = ['phylum', 'class', 'order', 'family', 'genus', 'species']):
        self.ranks = {rk:idx for idx, rk in enumerate(ranks)}
        
    def select_sites(self, nsites, rank='genus'):
        # get the nsites more informative sites per taxon for the current rank
        # must run this AFTER generate diff_tab
        # should run this after select_taxons
        self.selected_rank = rank
        rk = self.ranks[rank]
        order_subtab = self.order_tab[self.taxtab[:,0] == rk,:nsites]
        self.selected_sites = np.unique(order_subtab)
        selected_seqs = self.selected_seqs[rank]
        
        selected_matrix = self.matrix[selected_seqs, self.selected_sites]
        selected_tax = self.tax.iloc[selected_seqs]
        return selected_matrix, selected_tax
            
    def select_taxons(self, ntaxes=None, minseqs=None, thresh=None):
        # ntaxes: select the ntaxes most populated taxons
        # minseqs (int): select all taxes with more than minseqs sequences
        # thresh (float): select taxes representing more than thresh percent of the total
        # get the sequences corresponding to the m most populated taxons
        
        for rank, rank_col in self.tax.T.iteritems():
            tax_counts = rank_col.value_counts(ascending = False)
            
            # can use different criteria for selecting taxons
            if not ntaxes is None:
                selected = tax_counts.iloc[:ntaxes]
            elif not minseqs is None:
                selected = tax_counts.loc[tax_counts >= minseqs]
            elif not thresh is None:
                seq_thresh = len(rank_col) * thresh
                selected = tax_counts.loc[tax_counts >= seq_thresh]
            
            selected_tax = selected.index.to_numpy(dtype = int)
            selected_idx = np.argwhere(rank_col.isin(selected_tax).values).flatten()
            self.selected_tax[rank] = selected_tax
            self.selected_seqs[rank] = selected_idx
