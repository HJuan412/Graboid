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

def get_entropy(matrix, omit_missing=True):
    counts = np.array([np.sum(matrix == i, axis=0) for i in range(5)]).T
    if omit_missing:
        counts[:,0] = 0
    
    freqs = np.divide(counts, counts.sum(axis=1).reshape(-1, 1))
    with np.errstate(divide='ignore'):
        entropy = -np.sum(np.where(freqs > 0, np.log2(freqs), 0) * freqs, dtype=np.float32, axis=1)
    return entropy

def build_tax_series(tax_tab):
    # reformat taxonomy_table to facilitate per taxon entropy calculation
    # tax_series contains the positions (row indexes) of each taxon occurrence for every rank in the alignment matrix
    # index values are taxIds and are not unique, the number of appearances of each taxon equals the number of rows that belong to said taxon
    # tax_series contains no rank information
    
    # add index positions
    _tab = tax_tab.copy()
    _tab['idx'] = np.arange(len(tax_tab))
    tax_series = []
    # extract index positions of each rank
    for rk in tax_tab.columns:
        tax_series.append(_tab.set_index(rk)['idx'])
    # sort series (cluster taxa occurrences together) and remove unknown values (0)
    tax_series = pd.concat(tax_series).sort_index()
    try:
        tax_series.drop(index=0)
    except:
        pass
    
    return tax_series

def sans_tax_entropy(matrix, tax_tab, omit_missing=True):
    """
    Calculate entropy of the alignemnt matrix after removing instances of each
    taxon.

    Parameters
    ----------
    matrix : numpy.array
        Alignment matrix.
    tax_tab : pandas.DataFrame
        Lineage table of alignment sequences.
    omit_missing : bool, optional
        Omit missing values. The default is True.

    Returns
    -------
    entropy_array : numpy.array
        Calculated entropy for each site (column) of the matrix for each taxon
        (rows).
    taxids : list
        List of taxonomic ids for each sequence.
    tax_counts : list
        Counts of instances of each taxon.

    """
    # builds entropy difference tab, columns : rank_idx, TaxID, records, bases..., n (number of records)
    tax_series = build_tax_series(tax_tab)
    
    entropy_array = []
    taxids = []
    tax_counts = []
    for tax, subseries in tax_series.groupby(level=0):
        tax_submat = np.delete(matrix, subseries.values, axis=0)
        tax_entropy = get_entropy(tax_submat, omit_missing)
        taxids.append(tax)
        entropy_array.append(tax_entropy)
        tax_counts.append(len(subseries))
    entropy_array = np.array(entropy_array)
    return entropy_array, taxids, tax_counts

def get_information_gain(matrix, tax_tab, omit_missing=False):
    """
    Calculate the information gain for each site (column) for each taxon.
    Information gain for a given taxon in a given site is taken as the
    difference in entropy between the complete dataset and the dataset
    resulting of removing the instances of the taxon in question.
    High gain values mean the site is a good signal for the taxon.

    Parameters
    ----------
    matrix : numpy.array
        2D array, contains the alignment data.
    tax_tab : pandas.DataFrame
        Dataframe containing the taxonomic classification for each sequence.
        Each column of the dataframe correspond to a givenn taxonomic rank.
    omit_missing : bool, optional
        Do not take missing values into account for entropy calculations.
        The default is False.

    Returns
    -------
    diff_tab : pandas.DataFrame
        Table containing the information gain value for each taxon at each site.
    tax_counts : pandas.Series
        Series accounting the number of representative sequences in each taxon.

    """
    
    general_entropy = get_entropy(matrix, omit_missing)
    tax_entropy, taxids, tax_counts = sans_tax_entropy(matrix, tax_tab, omit_missing)
    entropy_difference = general_entropy - tax_entropy
    diff_tab = pd.DataFrame(entropy_difference, index=taxids)
    tax_counts = pd.Series(tax_counts, index=diff_tab.index)
    return diff_tab, tax_counts