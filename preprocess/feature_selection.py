#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 12:32:25 2022

@author: hernan
Feature selection
"""

#%% modules
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
def entropy_nb(matrix):
    # calculate entropy for a whole matrix
    # null columns take an entropy value of + infinite to differentiate them from columns with a single valid value
    entropy = np.full(matrix.shape[1], np.inf)
    for idx, col in enumerate(matrix.T):
        valid_rows = col[col != 0] # only count known values
        values = np.unique(valid_rows)
        counts = np.array([(valid_rows == val).sum() for val in values])
        n_rows = counts.sum()
        if n_rows > 0:
            # only calculate entropy for non-null columns
            freqs = counts / n_rows
            entropy[idx] = -np.sum(np.log2(freqs) * freqs)
    return entropy

def get_sorted_sites(matrix, tax_table, return_general=False, return_entropy=False, return_difference=False):
    # calculate entropy difference for every taxon present in matrix
    # tax_table is an extended taxonomy dataframe for the records contained in matrix
    # return ordered sites (by ascending entropy difference) for each taxon by default (first sites are the best)
    # also returns a list of taxa for organization purposes
    # if return_general is set to True, return the general entropy array
    # if return_entropy is set to True, return the per taxon entropy array
    # if return_difference is set to True, return the (unordered) entropy difference matrix
    general_entropy = entropy_nb(matrix)
    
    tax_list = np.array([])
    tax_entropy = []
    
    # get entropy per taxon
    for rk, col in tax_table.T.iterrows():
        taxa = col.dropna().unique()
        tax_list = np.concatenate((tax_list, taxa))
        for tax in taxa:
            tax_submat = matrix[col == tax]
            tax_entropy.append(entropy_nb(tax_submat))
    tax_entropy = np.array(tax_entropy)
    
    ent_diff = tax_entropy - general_entropy
    ent_diff_order = np.argsort(ent_diff, 1)
    
    result = (ent_diff_order, tax_list)
    if return_general:
        result += (general_entropy,)
    if return_entropy:
        result += (tax_entropy,)
    if return_difference:
        result += (ent_diff,)
        
    return result

def get_nsites(sorted_sites, min_n=None, max_n=None, step_n=None, n=None):
    # takes the resulting array from get_sorted_sites, returns a list with the (unique) n-1:n best sites for the range min_n:max_n with step_
    # if n is given in kwargs, use a single value of n
    if not n is None:
        n_sites = np.array([n], dtype=int)
    else:
        n_sites = np.arange(min_n, max_n+1, step_n)
    
    site_lists = []
    all_sites = np.array([]) # store already included sites here, avoid repetition
    for n in n_sites:
        sites = np.unique(sorted_sites[:, :n])
        sites = sites[~np.isin(sites, all_sites)]
        site_lists.append(sites)
        all_sites = np.concatenate((all_sites, sites))
    return site_lists

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

def per_tax_entropy(matrix, tax_tab, ranks):
    # builds entropy difference tab, columns : rank_idx, TaxID, records, bases..., n (number of records)
    bases = np.arange(matrix.shape[1])
    sub_tabs = []
    for rk_idx, rk in enumerate(ranks):
        taxa = tax_tab[rk].dropna().unique()
        ent_matrix = np.zeros((len(taxa), len(bases)))
        n_counts = []
        tax_sorted = []
        for idx, (tax, tax_subtab) in enumerate(tax_tab.groupby(rk)):
            tax_sorted.append(tax)
            rows = tax_subtab.index.values
            # record taxon entropy
            rk_entropy = get_matrix_entropy(matrix[rows])
            ent_matrix[idx] = rk_entropy
            n_counts.append(len(rows))
        rank_tab = pd.DataFrame(ent_matrix)
        rank_tab['Rank'] = rk_idx
        rank_tab['TaxID'] = tax_sorted
        rank_tab['n'] = n_counts
        sub_tabs.append(rank_tab)
    entropy_tab = pd.concat(sub_tabs)
    return entropy_tab.set_index(['Rank', 'TaxID'])

def get_ent_diff(matrix, tax_tab, ranks):
    general_entropy = get_matrix_entropy(matrix)
    tax_entropy = per_tax_entropy(matrix, tax_tab, ranks)
    diff_tab = tax_entropy.drop(columns='n') - general_entropy
    # reattach the n column
    diff_tab['n'] = tax_entropy.n.values
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

def extract_cols(matrix, cols):
    # extract the position of each column specified in cols from every row in matrix
    cols_submat = np.zeros((len(matrix), len(cols)), dtype = matrix.dtype)
    for idx, row in enumerate(matrix):
        cols_submat[idx] = row[np.isin(row, cols)]
    return cols_submat
#%%
class Selector:
    def __init__(self, out_dir, ranks):
        self.out_dir = out_dir
        self.ranks = ranks
        self.rk_dict = {rk:idx for idx, rk in enumerate(ranks)}
        self.order_file = f'{out_dir}/order.npz'
        self.diff_file = f'{out_dir}/diff.csv'
    
    def build_tabs(self, matrix, accs, tax_tab, guide):
        # filter guide for the accs present in the alignment matrix
        # matrix : generated alignment matrix
        # accs : accession list obtained from mapper (contains the accessions of records that made into the matrix)
        # tax_tab : table containing the taxonomic index assigned to each record
        # guide : EXTENDED guide
        taxids = tax_tab.loc[accs, 'TaxID'].values
        tax_guide = guide.loc[taxids].reset_index(drop=True) # this table contains the full taxonomy of each SECUENCE
        
        # build difference table
        print('Calculating entropy differences...')
        diff_tab = get_ent_diff(matrix, tax_guide, self.ranks)
        self.diff_tab = diff_tab
        print('Done!')
        # order bases by decreasing information difference
        # ordered contains the placement of each column per row in the difference tab
        # taxa is the difference tab index as a numpy array
        print('Sorting columns by entropy difference...')
        self.order_tab = np.flip(np.argsort(diff_tab.drop(columns='n').to_numpy(), 1), 1).astype(np.int16)
        self.order_tax = np.array([diff_tab.index.get_level_values(0), diff_tab.index.get_level_values(1)], dtype=int).T
        print('Done!')
        # save data
        np.savez_compressed(self.order_file,
                            order = self.order_tab,
                            taxs = self.order_tax)
        self.diff_tab.to_csv(self.diff_file)
    
    def load_order_mat(self, file):
        order_data = np.load(file)
        self.order_tab = order_data['order']
        self.order_tax = order_data['taxs']
    
    def load_diff_tab(self, file):
        # remember that the last column of diff tab ('n') is the count of records for that taxon
        self.diff_tab = pd.read_csv(file, index_col = [0, 1])
    
    def get_sites(self, n_range, rank, cols=None):
        # for a given range of sites, generate a dictionary containing the new sites selected at each n
        # used for exploring multiple n values in calibration and classification, (avoids repeating calculations)
        if rank not in self.ranks:
            raise Exception(f'Invalid rank {rank} not found in: {" ".join(self.ranks)}')            
        self.selected_rank = rank
        # get the selected rank's index
        rk = self.rk_dict[rank]
        # extract the rows corresponding to the given rank
        rank_submat = self.order_tab[self.order_tax[:,0] == rk]
        
        # if no columns were specified, select among every column (YOU SHOULDN'T DO THIS)
        col_submat = rank_submat
        if not cols is None:
            if min(cols) < 0 or max(cols) > self.order_tab.max():
                raise Exception(f'Invalid column indexes, must be between 0 and {self.order_tab.max()}')
            # get the specified columns
            col_submat = extract_cols(rank_submat, cols)
        
        # sites dictionary will keep the selected sites specific for a given n (include the sites from the previous n values) 
        sites = {}
        # total_sites keeps account of sites that have already been incorporated
        total_sites = np.array([], dtype = np.int8)
        for n in n_range:
            # get the best n sites for every row in col_submat, select all that are not already selected
            n_sites = np.unique(col_submat[:, :n])
            new_sites = n_sites[np.in1d(n_sites, total_sites, invert=True)]
            # only update sites dictionary if new sites are incorporated for n
            if len(new_sites) > 0:
                sites[n] = new_sites
                total_sites = np.concatenate([total_sites, new_sites])
        return sites
