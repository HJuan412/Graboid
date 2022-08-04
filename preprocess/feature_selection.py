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

#%% functions - information quantification
@nb.njit
def get_entropy(array):
    valid_rows = array[array != 0]
    n_rows = len(valid_rows)
    values = np.unique(valid_rows)
    counts = np.array([(valid_rows == val).sum() for val in values])
    freqs = counts / n_rows
    return -np.sum(np.log2(freqs) * freqs)

def get_matrix_entropy(matrix):
    entropy = np.zeros(matrix.shape[1])
    for idx, col in enumerate(matrix.T):
        entropy[idx] = get_entropy(col)
        
    return (2-entropy) / 2 # 1 min entropy, 0 max entropy

def pte(matrix, tax_tab):
    # returns entropy_tab with multiindex (rank, tax) and ncols = matrix.shape[1]
    tabs = [] # will concatenate into entropy_tab at the end
    # iterate over every rank
    for rank, tax_col in tax_tab.T.iteritems():
        # get unique taxons in rank
        tax_list = tax_col.unique().tolist()
        # get the entropy for each taxon
        entropy_mat = np.zeros((len(tax_list), matrix.shape[1]))
        for idx, tax in enumerate(tax_list):
            entropy_mat[idx] = get_matrix_entropy(matrix[tax_col == tax])
        tabs.append(pd.DataFrame(entropy_mat, index = pd.MultiIndex.from_product([[rank], tax_list])))
    # merge entropy tables for each taxon
    entropy_tab = pd.concat(tabs)
    return entropy_tab

#TODO: replace with pte after test
def per_tax_entropy(matrix, tax_tab):
    n_cols = matrix.shape[1]
    cols = tax_tab.columns
    ent_tab = pd.DataFrame(columns = list(range(n_cols)) + ['rank'])

    for rk in cols:
        rank_col = tax_tab[rk].to_numpy()
        uniq_taxes = tax_tab[rk].unique()

        ent_subtab = pd.DataFrame(index = uniq_taxes, columns = list(range(n_cols)))
        for idx, tax in enumerate(uniq_taxes):
            tax_idx = np.argwhere(rank_col == tax).flatten()
            sub_matrix = matrix[tax_idx]
            sub_entropy = get_matrix_entropy(sub_matrix)
            ent_subtab.at[tax] = sub_entropy
        ent_subtab['rank'] = rk
        ent_tab = pd.concat([ent_tab, ent_subtab], axis = 0)
    return ent_tab

def get_ent_diff(matrix, tax_tab):
    general_entropy = get_matrix_entropy(matrix)
    # p_tax_ent = per_tax_entropy(matrix, tax_tab)
    # p_tax_ent = p_tax_ent.loc[p_tax_ent.index.notnull()]
    # diff_tab = p_tax_ent.iloc[:,:-1] - general_entropy
    # diff_tab['rank'] = p_tax_ent['rank']
    
    #TODO: clear commented (previous body) after pte test
    p_tax_ent = pte(matrix, tax_tab)
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
        self.order_tab = np.argsort(self.diff_tab, )
    
    def select_sites(self, nsites, rank='genus'):
        # get the nsites more informative sites per taxon for the current rank
        # must run this AFTER generate diff_tab
        # should run this after select_taxons
        self.selected_rank = rank
        order_subtab = self.order_tab.iloc[rank, -nsites:] # np.argsort doesn't allow descending order
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
