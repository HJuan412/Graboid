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
    valid_rows = array[np.argwhere(array != 16).flatten()]
    n_rows = len(valid_rows)
    values = np.unique(valid_rows)
    counts = np.array([(valid_rows == val).sum() for val in values])
    freqs = counts / n_rows
    return -np.sum(np.log2(freqs) * freqs)

def get_matrix_entropy(matrix):
    n_cols = matrix.shape[1]

    entropy = np.zeros(n_cols)
    for i in range(n_cols):
        array = matrix[:,i]
        entropy[i] = get_entropy(array)
        
    return (2-entropy) / 2 # 1 min entropy, 0 max entropy

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
    p_tax_ent = per_tax_entropy(matrix, tax_tab)
    p_tax_ent = p_tax_ent.loc[p_tax_ent.index.notnull()]
    diff_tab = p_tax_ent.iloc[:,:-1] - general_entropy
    diff_tab['rank'] = p_tax_ent['rank']
    
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

#%% functions - feature selection
def select_features(table, rank, nsites, criterium = 'diff'):
    selected = set()
    if criterium == 'diff':
        sub_tab = table.loc[table['rank'] == f'{rank}_id'].drop('rank', axis = 1)
        for tax, row in sub_tab.iterrows():
            sorted_diff = row.sort_values(ascending = False).index.tolist()
            selected.update(sorted_diff[:nsites])
    elif criterium == 'gain':
        series = table.loc[f'{rank}_id']
        sorted_idxs = series.sort_values(ascending = False).index.tolist()
        selected.update(sorted_idxs[:nsites])
    
    return list(selected)

def build_training_data(matrix, col_list):
    return matrix[:,col_list]
# TODO: how do the selected sites map to the reference sequence

def plot_gain(table, rank, criterium = 'diff'):
    fig, ax = plt.subplots(figsize = (7,10))
    
    # TODO: handle header and labels
    if criterium == 'diff':
        sub_mat = table.loc[table['rank'] == f'{rank}_id'].drop('rank', axis = 1).to_numpy()
        x = np.arange(sub_mat.shape[1])
        y = sub_mat.mean(axis = 0)
        y_std = np.array([col.std() for col in sub_mat.T]) # have to do this shit to get deviation for some reason
        ax.bar(x, y, yerr = y_std)
    elif criterium == 'gain':
        data = table.loc[f'{rank}_id'].to_numpy()
        x = np.arange(len(data))
        ax.bar(x, data)
    
    ax.margins(x = 0.05, y = 0.01)

#%%
class Selector():
    def __init__(self, matrix, tax):
        self.matrix = matrix
        self.tax = tax
        self.ranks = tax.columns.tolist()
        self.diff_tabs = {rk:np.empty(0) for rk in self.ranks}
        # default selection is all the sites and sequences
        self.selected_tax = {rk:np.empty(0) for rk in self.ranks}
        self.selected_seqs = {rk:np.arange(matrix.shape[0]) for rk in self.ranks}
        self.selected_sites = {rk:np.arange(matrix.shape[1]) for rk in self.ranks}
        # TODO: kill the _id suffix in the taxid table (i hate it)

    def generate_diff_tab(self):
        # Get the entropy difference for each base/taxon/rank
        for rank in self.ranks:
            sub_mat = self.matrix[self.selected_seqs[rank]]
            sub_tax = self.tax.iloc[self.selected_seqs[rank]]
    
            general_entropy = get_matrix_entropy(sub_mat)
            p_tax_ent = per_tax_entropy(sub_mat, sub_tax)
            p_tax_ent = p_tax_ent.loc[p_tax_ent.index.notnull()]
            diff_tab = p_tax_ent.iloc[:,:-1] - general_entropy
            diff_tab['rank'] = p_tax_ent['rank']
            
            self.diff_tabs[rank] = diff_tab
    
    def select_sites(self, nsites):
        # get the nsites more informative sites per taxon for the current rank
        # must run this AFTER generate diff_tab
        # should run this after select_taxons
        for rank in self.ranks:
            selected = set()
            diff_tab = self.diff_tabs[rank]
            sub_tab = diff_tab.loc[diff_tab['rank'] == f'{rank}'].drop('rank', axis = 1)
            for tax, row in sub_tab.iterrows():
                sorted_diff = row.sort_values(ascending = False).index.tolist()
                selected.update(sorted_diff[:nsites])
            self.selected_sites[rank] = np.array(list(selected), dtype = int)
    
    def select_taxons(self, ntaxes=None, minseqs=None, thresh=None):
        # ntaxes: select the ntaxes most populated taxons
        # minseqs (int): select all taxes with more than minseqs sequences
        # thresh (float): select taxes representing more than thresh percent of the total
        # get the sequences corresponding to the m most populated taxons
        
        for rank in self.ranks:
            rank_col = self.tax[f'{rank}']
            tax_counts = rank_col.value_counts(ascending = False)
            
            # can use different criteria for selecting taxons
            if not ntaxes is None:
                selected = tax_counts.iloc[:ntaxes]
            elif not minseqs is None:
                selected = tax_counts.loc[tax_counts >= minseqs]
            elif not thresh is None:
                seq_thresh = len(rank_col) * thresh
                selected = tax_counts.loc[tax_counts >= seq_thresh]
            
            selected_tax = selected.index.to_numpy()
            selected_idx = np.argwhere(rank_col.isin(selected_tax).values).flatten()
            self.selected_tax[rank] = selected_tax.astype(int)
            self.selected_seqs[rank] = selected_idx.astype(int)
    
    def get_training_data(self, rank):
        rank = f'{rank}_id' # TODO: DELETE "_id"
        row_idxs = self.selected_seqs[rank]
        col_idxs = self.selected_sites[rank]

        selected_matrix = self.matrix[row_idxs][:, col_idxs]
        selected_tax = self.tax.iloc[row_idxs]
        return selected_matrix, selected_tax