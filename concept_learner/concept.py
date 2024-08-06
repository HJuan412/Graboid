#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 11:49:07 2024

@author: hernan
"""
#%% modules
import numpy as np
import pandas as pd

#%% functions
def get_distinct_shared(matrix, tax_sequences):
    # detect distinct and shared values for the current tax
    # counc how many distinctive and shared values are found in each site
    
    # format indexes for tax sequence extraction
    tax_sequences = np.reshape(tax_sequences, -1)
    nums = np.arange(1, 5)
    
    # count values within the taxon and in all the matrix
    in_values = matrix[tax_sequences].sum(axis=0)
    all_values = matrix.sum(axis=0)
    
    # calculate value differences, count changed values and detect new zeros
    differences = all_values - in_values
    changed_values = differences != all_values
    
    # Identify and count distinct values in each site
    new_zeros = ((all_values != 0) & (differences == 0))
    distinct_vals = [nums[new_z] for new_z in new_zeros]
    distinct_vals_encoded = new_zeros
    distinct_counts = new_zeros.sum(axis=1)
    
    # Identify and count shared values in each site
    shared = changed_values & ~new_zeros
    shared_vals = [nums[sh] for sh in shared]
    shared_vals_encoded = shared
    shared_counts = shared.sum(axis=1)
    
    return distinct_vals, distinct_vals_encoded, distinct_counts, shared_vals, shared_vals_encoded, shared_counts

def get_site_types(distinct_vals, distinct_counts, shared_vals, shared_counts):
    
    # get types
    has_distinct = distinct_counts > 0
    has_shared = shared_counts > 0
    type1 = (has_distinct & ~has_shared)
    type2 = (has_distinct & has_shared)
    type3 = ((~has_distinct) & (shared_counts == 1))
    type4 = ((~has_distinct) & (shared_counts > 1))
    
    # prepare results
    site_types = type1 + type2*2 + type3*3 + type4*4
    type_summmary = pd.Series([type1.sum(),
                          type2.sum(),
                          type3.sum(),
                          type4.sum()], index=[1,2,3,4], name='Type')
    return site_types, type_summmary

def get_single_rules(matrix, tax_sequences, site_types, distinct_vals, shared_vals):
    # identify single site rules in the alignment, distinguish between whole and partial rules
    # rules are dictionaries with key:values -> site index:values
    single_whole = {}
    single_partial = {}
    
    site_indexes = np.arange(len(site_types))
    
    # identify single whole rules
    # site is type 1 and all values are known within the taxon
    has_all = np.any(matrix[tax_sequences], axis=2).sum(axis=0) == len(tax_sequences)
    whole_sites = site_indexes[(site_types == 1) & has_all]
    single_whole = {site:distinct_vals[site] for site in whole_sites}
    
    # identify single whole partial
    # type 1 sites with missing values in the taxon are counted in as partial rules
    partial_sites1 = site_indexes[(site_types == 1) & ~has_all]
    single_partial = {site:distinct_vals[site] for site in partial_sites1}
    # all type 2 sites are taken as partial rules
    partial_sites2 = site_indexes[site_types == 2]
    for site in partial_sites2:
        single_partial[site] = np.concatenate((distinct_vals[site], shared_vals[site]))
    
    return single_whole, single_partial

def check_single(whole, partial, matrix, tax_sequences, distinct_vals, shared_vals, distinct_vals_encoded, shared_vals_encoded):
    # verify which sequences of the taxon concept can be solved with the set of single rules
    solved = np.array([])
    unsolved = np.array([])
    confused = np.array([])
    
    # There is at least one whole rule, no need to keep searching
    if len(whole) > 0:
        solved = tax_sequences
    else:
        # If there are no single whole rules, it is possible that some sequences of the concept taxon are unsolved
        partial_sites = np.array(list(partial.keys()))
        partial_distinct = distinct_vals_encoded[partial_sites]
        partial_shared = shared_vals_encoded[partial_sites]
        
        # get tax sequences with at least one distinct value
        in_partial = matrix[tax_sequences][:, partial_sites]
        in_solved = np.any(in_partial & partial_distinct, axis=(1,2))
        solved = tax_sequences[in_solved]
        unsolved = tax_sequences[~in_solved]
        
        if len(unsolved) > 0:
            # if there are unsolved sequences, get outsider sequences that are compatible with the current concept
            # confused outsider sequences -> any ousider sequences that contains a shared value in ALL partial sites
            # get partial rule sites from outsider sequences
            out_partial = np.delete(matrix, tax_sequences, axis=0)[:, partial_sites]
            # get all out sequences that contain shared values in all partial rule sites
            out_confused = np.all(np.any(out_partial & partial_shared, axis=2), axis=1)
            confused = np.delete(np.arange(matrix.shape[0]), tax_sequences)[out_confused]
    
    # out_matrix = np.delete(matrix, tax_sequences, axis=0)
    # out_indexes = np.delete(np.arange(matrix.shape[0]), tax_sequences)
    # for site in partial:
    #     # get taxon sequences that contain one of the distinctive values found in the partial rule
    #     solved_by_site = tax_sequences[np.any(matrix[tax_sequences][:, site][:, distinct_vals[site]-1], axis=1)]
    #     solved.append(solved_by_site)
    #     confused.append(np.any(out_matrix[:, site][:, shared_vals[site]-1], axis=1))
    
    # solved = np.unique(np.concatenate(solved))
    # unsolved = np.setdiff1d(tax_sequences, solved)
    # confused = np.all(np.array(confused), axis=0)
    # confused = out_indexes[confused]
    return solved, unsolved, confused

def get_composite_rule(matrix, solved, unsolved, confused, shared_vals, site_types):
    composite_rule = []
    rule_vals = []
    rule_confused = confused
    
    # get all type 3 sites
    sites_3 = np.arange(matrix.shape[1])[site_types == 3]
    vals_3 = np.array([shared_vals[site] for site in sites_3]).flatten()
    
    # get all confused sequences
    confused_seqs = matrix[confused]
    
    # build composite rule
    for _ in np.arange(len(sites_3)):
        # count shared values
        shared_counts = confused_seqs[:, sites_3, vals_3].sum(axis=0)
        # get site with least shared values
        best_site = np.argmin(shared_counts)
        
        # break if no sequences can be removed, partial rule
        if shared_counts[best_site] == confused_seqs.shape[0]:
            break
        
        # add site & value to rule
        composite_rule.append(sites_3[best_site])
        rule_vals.append(vals_3[best_site])
        
        # break if no confused sequences remain, whole rule
        if shared_counts[best_site] == 0:
            break
        
        # remove confused sequences without shared sites
        confused_seqs = confused_seqs[confused_seqs[:, best_site]]
        rule_confused = rule_confused[best_site]
        
        # remove selected site & value
        sites_3 = np.delete(sites_3, best_site)
        vals_3 = np.delete(vals_3, best_site)
    
    composite_rule = np.array(composite_rule)
    rule_vals = np.array(rule_vals)
    
    return composite_rule, rule_vals, confused

#%%

# make dummy alignment of 8 sequences
# test taxon comprised of sequences 1, 2 & 3

dummy = np.array([
    [1,2,2,2,1],# 0, type 1 single value full
    [1,0,2,2,1],# 1, type 1 single value partial
    [1,2,3,3,1],# 2, type 1 multi value full
    [1,2,0,3,1],# 3, type 1 multi value partial
    [1,1,2,2,1],# 4, type 2 single value, one shared
    [1,2,3,1,1],# 5, type 2 multi value
    [1,1,1,1,2],# 6, type 3 a
    [2,2,2,2,1],# 7, type 3 a
    [1,2,2,2,2],# 8, type 3 b
    [1,1,2,2,2],# 9, type 4
    [1,1,1,3,2]# 10, type 2 single value, multiple shared, one distinctive
    
    ]).T

encoded = np.array([(dummy == i).T for i in [1,2,3,4]]).T
#%%
tax_seqs = np.array([1,2,3])

def test(matrix, indexes):
    distinct_vals, distinct_vals_encoded, distinct_counts, shared_vals, shared_vals_encoded, shared_counts = get_distinct_shared(matrix, indexes)
    site_types, type_summary = get_site_types(distinct_vals, distinct_counts, shared_vals, shared_counts)
    single_whole, single_partial = get_single_rules(matrix, indexes, site_types, distinct_vals, shared_vals)
    solved, unsolved, confused = check_single(single_whole, single_partial, matrix, indexes, distinct_vals, shared_vals, distinct_vals_encoded, shared_vals_encoded)
    return solved, unsolved, confused

[0,1,2,3,4,5,6,7,8,9]
# case 1, dummy alignment includes 1 single whole site, can be fully solved
test(encoded, tax_seqs)

# case 2, dummy alignment has no single whole sites, can be fully solved with single partial rules
test(encoded[:, [1,2,4,6,7,8,9]], tax_seqs)

# case 3, dummy alignment has no single whole sites, can't be fully solved with single partial rules, no outsider sequences confused
test(encoded[:, [3,6,7,8,9]], tax_seqs)

# case 4, dummy alignment has no single whole sites, can't be fully solved with single ppartial rules, outsider sequences confused
test(encoded[:, [4, 6, 9]], tax_seqs)

#%%
def get_distinct_vals(matrix, in_indexes):
    """
    Identify all sequences within the in_indexes-defined group that contain at least one distinctive site for the concept taxon
    
    Parameters
    ----------
    matrix : numpy.array
        Encoded alignment array. 3D boolean array of shape (sequences, sites, 4)
    in_indexes : array-like
        Array containing the indexes of the concept sequences. Values must be between 0 and n_sequences - 1
    
    Returns
    -------
    distinct_sites : numpy.array
        Array of sites containing distinctive values for the concept taxon
    distinct_vals : list
        List of arrays containing the distinctive values found in each site
        Contains the same number of elements as distinct_sites
        Each site may contain multiple distinctive values
    solved : numpy.array
        Array of indexes of solved (fully distinguished from the outsider sequences by at least one distinctive value) concept taxon sequences
    unsolved : numpy.array
        Array of indexes of concept taxon sequences without distinctive values
    
    """
    
    # count values in full alignment and taxon sub_alignment
    full_count = matrix.sum(axis=0)
    
    # define in_matrix as a subset of the matrix, count values in in_matrix
    in_matrix = matrix[in_indexes]
    in_count = in_matrix.sum(axis=0)
    
    # get count differences
    diff_count = full_count - in_count
    
    # identify distinctive sites & values
    new_zeros = ((full_count != 0) & (diff_count == 0)) # sites where the removal of in_matrix generates one or more value count to drop to 0
    distinct_sites = np.arange(matrix.shape[1])[new_zeros.sum(axis=1) > 0]
    distinct_vals = new_zeros * np.array([1,2,3,4])
    distinct_vals = [dv[nz] for dv, nz in zip(distinct_vals[distinct_sites], new_zeros[distinct_sites])] # get the distinctive values in each of the selected sites
    
    # determine solved sequences
    solved = np.full(len(in_indexes), False)
    for site, vals in zip(distinct_sites, distinct_vals):
        solved_by_site = np.any(in_matrix[:, site, vals-1], axis=1) # get sequences that are solved by the current value's distinctive sites
        solved = solved | solved_by_site
    
    # record indexes of solved and unsolved sequences
    unsolved = in_indexes[~solved].values
    solved = in_indexes[solved].values
    
    return distinct_sites, distinct_vals, solved, unsolved

def filter_sites(matrix, max_unknown=0.05):
    """
    Select sites that contain a single KNOWN value across all rows of the matrix
    Filter out sites with more than max_unknown % unknown sites
    
    Parameters
    ----------
    matrix : numpy.array
        Encoded alignment array. 3D boolean array of shape (sequences, sites, 4)
    max_unknown : float, optional
        Unknown values threshold. The default is 0.05.
    
    Returns
    -------
    single_val_sites : numpy.array
        Array of sites with a single known value
    filtered_sites : numpy.array
        Array of single value sites with less than max_unknown % missing values
    
    """
    # get sites with single value in the in_matrix
    # get a list of site indexes in the in_matrix that have less than max_unknown missing values
    
    # get sites with single known value
    site_indexes = np.arange(matrix.shape[1])
    single_val_sites = site_indexes[np.any(matrix, axis=0).sum(axis=1) == 1]
    
    # filter sites by known content
    sites_content = np.any(matrix[:, single_val_sites], axis=2).sum(axis=0) / matrix.shape[0]
    sites_content = sites_content >= (1 - max_unknown)
    filtered_sites = single_val_sites[sites_content]
    
    return single_val_sites, filtered_sites

def get_full_sequences(matrix):
    """
    Get sequences that contain no missing values in the single value sites

    Parameters
    ----------
    matrix : numpy.array
        Encoded alignment array. 3D boolean array of shape (sequences, sites, 4)

    Returns
    -------
    seq_full : numpy.array
        Indexes of sequences of matrix that contain no missing values in the single value sites

    """
    # get full sequences with single known values from the concept matrix
    
    # get single value sites
    single_val_sites, filtered_sites = filter_sites(matrix, 0)
    
    # get full sequences & incomplete sequences
    seq_full = np.any(matrix[:, single_val_sites], axis=2).sum(axis=1) == single_val_sites.shape
    
    # returns boolean array indicating which sequences in matrix are full
    return seq_full

def solve(matrix, in_indexes, out_indexes, site_indexes):
    """
    Identify the signal that differentiates the sequences of in_indexes from the largest amount of outsider sequences
    
    Parameters
    ----------
    matrix : numpy.array
        Encoded alignment array. 3D boolean array of shape (sequences, sites, 4)
    in_indexes : numpy.array
        Array of indexes of concept sequences to be solved
    out_indexes : numpy.array
        Array of indexes of outsider sequences
    site_indexes : numpy.array
        Array of indexes of pre-filtered sites (single value, all known)
    
    Returns
    -------
    signal : numpy.array
        Array of site indexes included in the signal
    signal_solved : bool
        Boolean that indicates if the entirety of the outsider sequences could be distinguished
    to_exclude_seqs : numpy.array
        Array of outsider sequences that the signal sites cannot differentiate from the concept sequences
    
    """
    
    # get value for each site in the inner matrix & generate list of unchecked sites
    in_matrix_vals = np.any(matrix[in_indexes], axis=0)
    unchecked_sites = site_indexes.copy()
    
    # initialize signal array
    signal = []
    signal_solved = False
    to_exclude_seqs = out_indexes.copy()
    
    # iterate while solving signal
    for _ in np.arange(len(site_indexes)):
        # count shared values for each remaining unchecked site
        shared_vals = matrix[to_exclude_seqs][:, unchecked_sites].sum(axis=0)[in_matrix_vals[unchecked_sites]]
        min_shared = shared_vals.min()
        
        if min_shared == len(to_exclude_seqs):
            # if minimum shared values equals the number of remaining outsider sequences, we can't solve the signal any further
            break
        
        # select site with least shared values, best site is taken from the unchecked sites list
        sorted_sites = np.argsort(shared_vals)
        best_site = unchecked_sites[sorted_sites[0]]
        # update signal array
        signal.append(best_site)
        
        if min_shared == 0:
            # if there are no shared values remaining, signal is resolved
            signal_solved = True
            break
        
        # remove solved sequences from out_matrix
        best_site_value = in_matrix_vals[best_site]
        rows_to_keep = matrix[to_exclude_seqs, best_site, best_site_value].flatten() # keep all sequences that match the submatrix value at best site
        to_exclude_seqs = to_exclude_seqs[rows_to_keep]
        
        # remove selected best site from unchecked sites
        unchecked_sites = np.delete(unchecked_sites, sorted_sites[0])
    
    signal = np.array(signal)
    # out_excluded = np.setdiff1d(out_indexes, to_exclude_seqs, assume_unique=True)
    # return signal, signal_solved, out_excluded
    
    # return not excluded out_sequences (to_exclude_seqs)
    return signal, signal_solved, to_exclude_seqs

def get_composite_signal(matrix, in_indexes, unsolved_indexes):
    """
    Identify taxon signals that allow to differentiate unsolved concept sequences from the greatest number of outsider sequences
    Vertical signal: determined using only full sites (columns) of concept taxon alignment
    Horizontal signal: calculated using only full sequences (rows) of the contept taxon alignment
    
    Parameters
    ----------
    matrix : numpy.array
        Encoded alignment array. 3D boolean array of shape (sequences, sites, 4)
    in_indexes : array-like
        Array containing the indexes of the concept sequences. Values must be between 0 and n_sequences - 1
    unsolved_indexes : array-like
        Array containing the indexes of the concept sequences with no distinctive values. These sequences are the ones to be solved
    
    Returns
    -------
    signal_v : numpy.array
        Array of site indexes included in the VERTICAL signal
    values_v : numpy.array
        Array of semi-distinctive values of each site in the VERTICAL signal
    signal_h : numpy.array
        Array of site indexes included in the HORIZONTAL signal
    values_h : numpy.array
        Array of semi-distinctive values of each site in the HORIZONTAL signal
    intersection_v : numpy.array
        Array of outsider sequences indexes not differentiated by the VERTICAL signal
    intersection_h : numpy.array
        Array of outsider sequences indexes not differentiated by the HORIZONTAL signal
    seqs_h : numpy.array
        Array of concept sequences solved by the HORIZONTAL signal
        This array may contain incomplete concept sequences that nonetheless have known values in every site of the HORIZONTAL signal
    
    """
    
    # get matrix of unsolved taxon sequences & matrix of outsider sequences
    in_matrix = matrix[unsolved_indexes]
    out_indexes = np.delete(np.arange(matrix.shape[0]), in_indexes)
    
    # get full sites and full sequences
    single_val_sites, full_value_sites = filter_sites(in_matrix, 0)
    seqs_full = get_full_sequences(in_matrix) # seqs full is a boolean array that specifies which rows of in_matrix are complete sequences (no unknowns)
    
    # predefine output values
    signal_v = np.array([], dtype=int)
    values_v = np.array([], dtype=int)
    signal_h = np.array([], dtype=int)
    values_h = np.array([], dtype=int)
    intersection_v = np.array([], dtype=int)
    intersection_h = np.array([], dtype=int)
    seqs_h = np.array([], dtype=int)
    
    # get vertical signal (use full single value sites)
    if len(single_val_sites) > 0:
        signal_v, signal_solved_v, intersection_v = solve(matrix, unsolved_indexes, out_indexes, full_value_sites)
    
    # get horizontal signal (use complete sequences)
    if seqs_full.sum() > 0:
        signal_h, signal_solved_h, intersection_h = solve(matrix, unsolved_indexes[seqs_full], out_indexes, single_val_sites)
    
        # get incomplete sequences that include all sites of the horizontal signal
        incomp_in_h = np.any(in_matrix[~seqs_full][:, signal_h], axis=2).sum(axis=1) == (~seqs_full).sum()
        seqs_h = np.concatenate((unsolved_indexes[seqs_full], unsolved_indexes[~seqs_full][incomp_in_h]))
    
    # get signal values
    if len(signal_v) > 0:
        values_v = np.any(in_matrix[:, signal_v], axis=0)
        values_v = (values_v * np.array([1,2,3,4]))[values_v]
    if len(signal_h) > 0:
        values_h = np.any(in_matrix[:, signal_h], axis=0)
        values_h = (values_h * np.array([1,2,3,4]))[values_h]
    
    return signal_v, values_v, signal_h, values_h, intersection_v, intersection_h, seqs_h

def process_lineage(rank, lineage_tab):
    """
    Extracts the columns corresponding to rank and the one immediately below from the lineage table
    Renames column names as Rank and Sub_rank
    Replaces missing values in rank column for the values at the leading rank (unless rank is the highest)
    If the selected rank is the lowest one, both columns correspond to rank
    
    Parameters
    ----------
    rank : str
        Rank column to be selected
    lineage_tab : pandas.DataFrame
        Lineage table to be processed
    
    Returns
    -------
    lineage_processed : pandas.DataFrame
        Two column dataframe containing the rank column with updated missing values (values missing in the leading rank are left as 0)
    rank_tail : str
        Rank immediately below the selected one
    
    """
    
    rank_lead = ({lineage_tab.columns[0]:None} | {rk0:rk1 for rk0, rk1 in zip(lineage_tab.columns[1:], lineage_tab.columns[:-1])})[rank]
    rank_tail = ({rk0:rk1 for rk0, rk1 in zip(lineage_tab.columns[:-1], lineage_tab.columns[1:])} | {lineage_tab.columns[-1]:None})[rank]
    rank_tail = rank if rank_tail is None else rank_tail
    
    # get current and following ranks, if current rank is the lowest, use current rank as the following
    lineage_processed = lineage_tab[[rank, rank_tail]].copy()
    lineage_processed.columns = ['Rank', 'Sub_rank']
    
    if rank_lead is None:
        return lineage_processed, rank_tail
    
    #fill missing values in rank (if not at the highest rank)
    missing_vals = lineage_processed.query('Rank == 0').index
    lineage_processed.loc[missing_vals, 'Rank'] = lineage_tab.loc[missing_vals, rank_lead]
    
    return lineage_processed, rank_tail

def get_unsolved_subtaxa(seqs_h, seqs_v, in_indexes, rank, lineage_tab):
    """
    Count the number of unsolved sequences for vertical and horizontal signals
    
    Parameters
    ----------
    seqs_h : numpy.array
        Array of concept sequences solved by the HORIZONTAL signal
    seqs_v : numpy.array
        Array of concept sequences solved by the VERTICAL signal
    in_indexes : numpy.array
        Array containing the indexes of the concept sequences. Values must be between 0 and n_sequences - 1
    rank : str
        Taxonomic rank of the concept
    lineage_tab : pandas.DataFrame
        Lineage table
    
    Returns
    -------
    solved_subtaxa : pandas.DataFrame
        Data frame counting the unsolved sequences for each subtaxon of the concept. Also includes the total count of sequences for each subtaxon
    
    """
    
    # all solved sequences have 0 intersections
    
    # process lineage table (columns = multiindex with level 0 values Rank, Sub_rank)
    lineage_tab, sub_rank = process_lineage(rank, lineage_tab)
    
    in_subtaxa = lineage_tab.loc[in_indexes, 'Sub_rank'].unique()
    
    solved_subtaxa = pd.DataFrame(0, index=in_subtaxa, columns='Unsolved_h Unsolved_v Total'.split())
    solved_subtaxa['Total'] = lineage_tab.loc[in_indexes, 'Sub_rank'].value_counts()
    
    # get number of intersections in partial_h
    h_subtaxa = lineage_tab.loc[seqs_h, 'Sub_rank'].value_counts()
    solved_subtaxa.loc[h_subtaxa.index, 'Unsolved_h'] = h_subtaxa
    
    # get number of intersections ONLY in partial_v
    v_subtaxa = lineage_tab.loc[seqs_v, 'Sub_rank'].value_counts()
    solved_subtaxa.loc[v_subtaxa.index, 'Unsolved_v'] = v_subtaxa
    
    return solved_subtaxa

def get_confused_out_taxa(intersect_h, intersect_v, in_indexes, rank, lineage_tab):
    """
    Count the number  number of non distinguishable outsider sequences for vertical and horizontal signals
    
    Parameters
    ----------
    intersect_h : numpy.array
        Array of outsider sequences indexes not differentiated by the HORIZONTAL signal
    intersect_v : numpy.array
        Array of outsider sequences indexes not differentiated by the VERTICAL signal
    in_indexes : numpy.array
        Array containing the indexes of the concept sequences. Values must be between 0 and n_sequences - 1
    rank : str
        Taxonomic rank of the concept
    lineage_tab : pandas.DataFrame
        Lineage table
    
    Returns
    -------
    confused_subtaxa : pandas.DataFrame
        Data frame counting the number of non distinguishable sequences for each outsider subtaxa. Also includes the total count of sequences for each subtaxon

    """
    
    # sequences in partial_h have !exclusion.Horizontal intersections
    # sequences ONLY in partial_v have !exclusion.Vertical intersections
    
    out_taxa = lineage_tab.drop(in_indexes)[rank].unique()
    
    confused_out_taxa = pd.DataFrame(0, index=out_taxa, columns='Confused_h Confused_v Total'.split())
    confused_out_taxa['Total'] = lineage_tab.drop(in_indexes)[rank].value_counts()
    
    # get number of intersections in partial_h
    h_intersections = lineage_tab.loc[intersect_h, rank].value_counts()
    confused_out_taxa.loc[h_intersections.index, 'Confused_h'] = h_intersections
    
    # get number of intersections ONLY in partial_v
    v_intersections = lineage_tab.loc[intersect_v, rank].value_counts()
    confused_out_taxa.loc[v_intersections.index, 'Confused_v'] = v_intersections
    
    return confused_out_taxa

def get_MES(matrix, branch, branch_sites):
    """
    Select the MES (most entropic site) for the given group of sequences

    Parameters
    ----------
    matrix : numpy.array
        Encoded alignment array. 3D boolean array of shape (sequences, sites, 4)
    branch : numpy.array
        Array of indexes of the sequences included in the branch to be split
    branch_sites : list
        List of sites already in use for the branch to be split

    Returns
    -------
    MES : int
        Index of the most entropic site among the branch sequences (accounting for already used sites)

    """
    # remove used sites from the matrix
    branch_matrix = np.delete(matrix[branch], branch_sites, axis=1)
    used_sites = np.delete(np.arange(matrix.shape[1]), branch_sites)
    
    # select MES (most entropic site)
    freqs = branch_matrix.sum(axis=0) / branch_matrix.shape[0]
    ent = -(np.where(freqs == 0, freqs, np.log2(freqs)) * freqs).sum(axis=1)
    MES = np.argsort(ent)[-1]
    
    # adjust MES
    MES = used_sites[MES]
    return MES

def split_branch(matrix, branch, branch_sites):
    """
    Find the best site to separate a given group of sequences

    Parameters
    ----------
    matrix : numpy.array
        Encoded alignment array. 3D boolean array of shape (sequences, sites, 4)
    branch : numpy.array
        Array of indexes of the sequences included in the branch to be split
    branch_sites : list
        List of sites already in use for the branch to be split

    Returns
    -------
    new_branches : list
        List of newly generated branches, each one is a sub array of the original branch array
    new_branch_sites : list
        List of used sites in the newly generated branches, contains one list for each new branch
        New sites lists are a copies of the branch_sites list, updated to include the site selected in the current iteration
    closed_branches : list
        List of newly generated branches that are closed (cannot be further divided)

    """
    # locate most entropic site
    branch_MES = get_MES(matrix, branch, branch_sites)
    branch_sites = branch_sites + [branch_MES]
    
    # split by MES (we don't care about the value)
    splitter = matrix[branch, branch_MES].T # get MES in the encoded matrix
    splitter = splitter[splitter.any(axis=1)] # filter empty columns (bases without representatives)
    
    sub_branches = [branch[base] for base in splitter]
    
    new_branches = []
    closed_branches = []
    # filter sub branches
    for sub_branch in sub_branches:
        # if a sub branch has no variable sites or a single sequence, it is complete
        has_variable_sites = matrix[sub_branch].any(axis=0).sum(axis=1).max() > 1 # count number of values per each site (any & sum), determine if at least one site has multiple values (max > 1)
        if len(sub_branch) > 1 and has_variable_sites:
            new_branches.append(sub_branch)
        else:
            closed_branches.append(sub_branch)
    new_branch_sites = [branch_sites for _ in new_branches]
    return new_branches, new_branch_sites, closed_branches

def get_convos(matrix, sequences, variable_sites):
    """
    Get unique value combinations of variable selected sites in the concept

    Parameters
    ----------
    matrix : numpy.array
        Encoded alignment array. 3D boolean array of shape (sequences, sites, 4)
    sequences : numpy.array
        Array of indexes of the sequences included in the concept tax
    variable_sites : numpy.array
        Array of indexes of the variable selected sites in the concept

    Returns
    -------
    convos : numpy.array
        2D-Array containing all unique combinations of values on the variable selected sites

    """
    
    # select variable sites from matrix
    matrix = matrix[:, variable_sites]
    # Define lists containing branches and branch sites
    branches = [sequences]
    branch_sites = [[]]
    # define list containing completed branches
    branches_closed = []
    
    # iterate and split branches
    for _ in variable_sites:
        # define containers for newly generated branches
        new_branches = []
        new_branch_sites = []
        
        for branch, br_sites in zip(branches, branch_sites):
            new_subbranches, new_subbranch_sites, closed_subbranches = split_branch(matrix, branch, br_sites)
            new_branches += new_subbranches
            new_branch_sites += new_subbranch_sites
            branches_closed += closed_subbranches
        branches = new_branches
        branch_sites = new_branch_sites
        if len(branches) == 0:
            # no more unsolved branches to solve
            break
    
    # get convo sequences
    bases_mat = np.tile([1,2,3,4], (matrix.shape[1], 1))
    convos = []
    for br in branches_closed:
        consensus = matrix[br].any(0)
        missing = consensus.any(1)
        br_seq = np.zeros(consensus.shape[0], dtype=int)
        br_seq[missing] = bases_mat[consensus]
        convos.append(br_seq)
    convos = np.array(convos)
    return convos

def compress(matrix, sequences, *sites):
    """
    Separate variable and invraiable sites, extract representatives of each combination of values present in the concept sequences

    Parameters
    ----------
    matrix : numpy.array
        Encoded alignment array. 3D boolean array of shape (sequences, sites, 4)
    sequences : numpy.array
        Array of indexes of the sequences included in the concept tax
    *sites : TYPE
        Arrays of selected sites in the concept tax

    Returns
    -------
    sites_single : numpy.array
        Array of indexes of selected sites with a single known value among the concept sequences
    values_single : numpy.array
        Array of values present in the single value sites of the concept taxon
    sites_multi : numpy.array
        Array of indexes of selected sites with multiple known values among the concept sequences
    values_multi : numpy.array
        2D array containing every unique combination of values present among the multiple value sites in the concept taxon

    """
    # get a list of unique and variable sites among the sequence set
    sites = np.unique(np.concatenate(sites)).astype(int)
    
    # get invariable sites
    single_value = matrix[sequences][:, sites].any(axis=0).sum(axis=1) <= 1
    sites_single = sites[single_value]
    sites_multi = sites[~single_value]
    
    # get representative values
    values_single = matrix[sequences][:, sites_single].any(axis=0)
    values_single = np.tile([1,2,3,4], (values_single.shape[0], 1))[values_single]
    values_multi = get_convos(matrix, sequences, sites_multi)
    return sites_single, values_single, sites_multi, values_multi

def encode_rule_vals(rule_values):
    encoded_vals = np.tile([1,0,0,0,0], (len(rule_values), 1)).astype(bool)
    for idx, v in enumerate(rule_values):
        encoded_vals[idx, v] = True
    return encoded_vals

#%% classes
class Concept:
    def __init__(self, name, rank=None):
        self.name = name
        self.rank = rank
        self.sequences = np.array([])
        
        self.rules = []
        self.rules_values = []
        
        self.informative_sites = np.array([])
        self.informative_values = np.array([])
        
        self.seqs_v = np.array([])
        self.seqs_h = np.array([])
        self.seqs_v_only = np.array([])
        self.signal_v = np.array([])
        self.signal_h = np.array([])
        self.values_v = np.array([])
        self.values_h = np.array([])
        self.intersection_v = np.array([])
        self.intersection_h = np.array([])
        self.intersection_hv = np.array([])
        self.intersection_vh = np.array([]) # alias to avoid confusion
        self.signal = pd.Series()
        self.non_shared = np.array([])
        
        self.unsolved_subtaxa = None
        self.confused_out_taxa = None
        
        self.solved = np.array([])
        self.not_solved = np.array([])
        self.fully_solved = False
    
    def learn(self, matrix, concept_sequences, lineage_tab):
        # register indexes of concept and outsider sequences 
        self.sequences = concept_sequences.values
        
        # select sequences with at least one distinctive value
        dist_sites, dist_vals, solved, unsolved = get_distinct_vals(matrix, concept_sequences)
        self.informative_sites = dist_sites
        self.informative_values = dist_vals
        
        # attempt to find composite signal for unsolved sequences
        if len(unsolved) > 0:
            signal_v, values_v, signal_h, values_h, intersection_v, intersection_h, seqs_h = get_composite_signal(matrix, concept_sequences, unsolved)
            
            self.seqs_v = unsolved
            self.seqs_v_only = np.setdiff1d(unsolved, seqs_h, assume_unique=True)
            self.seqs_h = seqs_h
            self.signal_v = signal_v
            self.signal_h = signal_h
            self.values_v = values_v
            self.values_h = values_h
            self.intersection_v = intersection_v
            self.intersection_h = intersection_h
            self.intersection_hv = np.intersect1d(intersection_v, intersection_h)
            self.intersection_vh = self.intersection_hv # alias to avoid confusion
            
            # signal attributes is a series that merges the sites and values of horizontal and vertical signal
            self.signal = pd.Series(0, index=np.unique(np.concatenate((self.signal_v, self.signal_h))).astype(int))
            self.signal.loc[self.signal_v] = self.values_v
            self.signal.loc[self.signal_h] = self.values_h
            
            # count fully and partially solved sequences
            if len(intersection_v) == 0:
                # vertical signal includes all unsolved sequences, if it excludes all outsider sequences (no intersection), all the concept is solved
                solved = np.concatenate([solved, unsolved])
                unsolved = np.array([])
            elif len(intersection_h) == 0:
                # sequences solved by horizontal signal registered, concept sequences not included in the horizontal signal are unsolved
                solved = np.concatenate([solved, seqs_h])
                unsolved = self.seqs_v_only
        self.solved = solved
        self.not_solved = unsolved
        if len(unsolved) == 0:
            self.fully_solved = True
        
        self.unsolved_subtaxa = get_unsolved_subtaxa(self.seqs_h, self.seqs_v, self.sequences, self.rank, lineage_tab)
        self.confused_out_taxa = get_confused_out_taxa(self.intersection_h, self.intersection_vh, self.sequences, self.rank, lineage_tab)
        
        # compress concept sequences
        self.sites_single, self.values_single, self.sites_multi, self.values_multi = compress(matrix, self.sequences, self.informative_sites, self.signal_v, self.signal_h)
        
        # get rules
        self.rules = np.concatenate((self.informative_sites, self.signal.index)).astype(int)
        self.rules_values = self.informative_values + self.signal.values.tolist()
        self.rules_encoded = encode_rule_vals(self.rules_values)
    
    def learn_rules(self, matrix, concept_sequences, lineage_tab):
        
        distinct_vals, distinct_vals_encoded, distinct_counts, shared_vals, shared_vals_encoded, shared_counts = get_distinct_shared(matrix, concept_sequences)
        site_types, type_summmary = get_site_types(distinct_vals, distinct_counts, shared_vals, shared_counts)
        
        # single full rules
        get_single_rules(matrix, concept_sequences, site_types, distinct_vals, shared_vals)
        
        # single partial rules
        
        # composite rule
        pass