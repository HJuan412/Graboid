#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 12:32:25 2022

@author: hernan
Feature selection
"""

#%% modules
import numpy as np
import pandas as pd

#%% functions - feature selection
def get_entropy(array):
    # entropy of a single array
    valid_rows = array[np.argwhere(array != 16)]
    n_rows = len(valid_rows)
    freqs = np.unique(valid_rows, return_counts = True)[1] / n_rows
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

# TODO: build training data