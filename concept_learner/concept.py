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
def get_site_types_in(in_matrix, out_matrix):
    # locate distinct, shared, missing (in and out) values, as well as invariable sites
    # returns encoded matrices of shape:
        # distinct_values, shared_values : (4, sites)
        # in_missing, out_missing, invariable : (sites)
        
    # get values found in the inner and outer matrices
    in_values = in_matrix.any(axis=0)
    out_values = out_matrix.any(axis=0)
    
    # get sites with missing values in the inner and outer matrices
    in_missing = ~in_matrix.any(axis=2).all(axis=0)
    out_missing = ~out_matrix.any(axis=2).all(axis=0)
    
    # get encoded distinct and shared values
    distinct_values = in_values & ~out_values
    shared_values = in_values & out_values
    
    # get invariable sites
    invariable = np.concatenate((in_matrix, out_matrix), axis=0).any(axis=0).sum(axis=1) == 1
    return distinct_values, shared_values, in_missing, out_missing, invariable

def get_site_types0(distinct_values, shared_values, in_missing, out_missing, invariable):
    # process output of get_site_types_in to typify the type of each site in the alignment
    # returns encoded sites array of shape (5, sites), indicating with a boolean the type of each site
    
    has_distinct = distinct_values.any(axis=1)
    has_shared = shared_values.any(axis=1)
    single_shared = shared_values.sum(axis=1) <= 1
    connecting_unknowns = in_missing & out_missing
    
    type0 = invariable
    type1 = has_distinct & ~has_shared & ~connecting_unknowns
    type2 = has_distinct & (has_shared | connecting_unknowns)
    type3 = ~has_distinct & single_shared & ~connecting_unknowns & ~invariable
    type4 = ~type0 & ~type1 & ~type2 & ~type3
    
    types = np.array([type0, type1, type2, type3, type4])
    return types

def get_single_full(types, distinct_values):
    # get single full rules, locate sites that show only distinctive values
    # returns dictionary of key:values
        # site index : encoded values
    single_full_sites = np.arange(distinct_values.shape[0])[types[1]]
    single_full_values = distinct_values[:, single_full_sites]
    return single_full_sites, single_full_values

def get_single_partial(types, distinct_values, shared_values):
    # get single partial rules, locate sites that have distinctive and shared values
    # returns dictinoary of key:values
        # site index : encoded values (includes distinctive and shared values)
    single_partial_sites = np.arange(distinct_values.shape[0])[types[2]]
    single_partial_values = distinct_values[:, single_partial_sites] | shared_values[:, single_partial_sites]
    return single_partial_sites, single_partial_values

def check_single_partial(rule_indexes, in_matrix, out_matrix, distinct_values, shared_values):
    # check which concept sequences are confirmed by the set of single partial rules
    # check if any outsider sequences are compatible with the ruleset
    # returns arrays of confirmed and out compatible sequences
    
    # select sites used by the single partial rules
    in_matrix = in_matrix[:, rule_indexes]
    out_matrix = out_matrix[:, rule_indexes]
    
    distinct_values = distinct_values[rule_indexes]
    shared_values = shared_values[rule_indexes]
    
    # get confirmed sequences
    confirmed = (in_matrix & distinct_values).any(axis=(1,2))
    
    # get out compatible sequences
    out_compatible = (out_matrix & shared_values).any(axis=2).all(axis=1)
    
    return confirmed, out_compatible

def prune0(out_shared):
    # remove redundant sites (sites that do not reject any new sequences)
    # returns a list of kept site indexes (Note, these indexes are relative to the set of type 2&3 sites), to get the real indexes extract the returned indexes from the array of composite rule sites
    
    # define kept sites list and indexes array
    kept_sites = []
    sites = np.arange(out_shared.shape[1])
    
    # get matrix of rejected outsider sequences (opposite of shared matrix)
    out_rejected = ~out_shared
    # initialize array of rejected outsider sequences (at first no sequences are rejected)
    rejected = np.full(out_shared.shape[0], False)
    
    # iterate until prunning is done
    for _ in range(out_shared.shape[0]):
        # count the non-yet-rejected sequences that share values with the concept for each remaining site
        total_shared = np.sum(out_shared[~rejected][:, sites], axis=0)
        # sort sites by number of shared sequences (ascending), select the site with the least sequences
        shared_sorted = sites[np.argsort(total_shared)]
        first_place = shared_sorted[0]
        
        # verify that there are new rejected sequences (relative to the previous iteration)
        new_rejected = ~rejected & out_rejected[:, first_place]
        # no new rejections? end pruning (do not include the current site, as it adds no information)
        if new_rejected.sum() == 0:
            break
        
        # update rejected & kept sites
        rejected = rejected | out_rejected[:, first_place]
        kept_sites.append(first_place)
        
        # are all outsider sequences rejected?
        if rejected.all():
            break
        
        # remove newly redundant sites
        # remaining_rejections : all remaining sites that reject any of the yet-unrejected sequences
        remaining_rejections = out_rejected[:, ~rejected][:, sites].any(axis=0)
        sites = sites[remaining_rejections]
        
        # are there sites remaining? (this would mean there are still sequences to reject)
        if len(sites) == 0:
            break
        
    return kept_sites

def get_composite(types, shared_values, out_matrix):
    # get composite rule, prune redundant sites, check if all outsider sequences are rejected
    # returns dictionary of key : values
        # site index : encoded values
    # and array of out compatible sequences
    
    # get type 2 & 3 sites
    site23_indexes = np.arange(shared_values.shape[0])[types[2] & types[3]]
    # select sites (type-2) with a single shared value
    single_shared = shared_values[site23_indexes].sum(axis=1) == 1
    site23_indexes = site23_indexes[single_shared]
    
    # get shared values with the outsider sequences
    out_matrix = out_matrix[:, site23_indexes]
    shared_values = shared_values[site23_indexes]
    # build shared matrix
    out_shared = (out_matrix & shared_values).any(axis=2)
    
    # prune redundant sites
    pruned = prune0(out_shared)
    
    # build ruleset and list compatible outsider sequences
    composite_sites = site23_indexes[pruned]
    composite_values = shared_values[:, composite_sites]
    out_compatible = out_shared.all(axis=1)
    
    return composite_sites, composite_values, out_compatible
#%%
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

def get_site_types(matrix, distinct_vals, distinct_counts, shared_vals, shared_counts):

    # get types
    has_distinct = distinct_counts > 0
    has_shared = shared_counts > 0
    
    type0 = np.any(matrix, axis=0).sum(axis=1) == 1 # type 0 sites are invariable/unknown across the alignment
    type1 = (has_distinct & ~has_shared)
    type2 = (has_distinct & has_shared)
    type3 = ((~has_distinct) & (shared_counts == 1)) & (~type0)
    type4 = ((~has_distinct) & (shared_counts > 1))
    
    # prepare results
    site_types = type1 + type2*2 + type3*3 + type4*4
    type_summmary = pd.Series([type0.sum(),
                               type1.sum(),
                               type2.sum(),
                               type3.sum(),
                               type4.sum()], name='Type')
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

def check_single(whole, partial, matrix, tax_sequences, distinct_vals_encoded, shared_vals_encoded):
    # verify which concept sequences are confirmed, compatible or missed by the signle rules
    # verify outsider concept sequences compatible with the single rules
    
    confirmed = np.array([], dtype=int)
    in_compatible = tax_sequences
    in_missed = np.array([], dtype=int)
    out_compatible = np.array([], dtype=int)
    
    in_matrix = matrix[tax_sequences]
    out_matrix = np.delete(matrix, tax_sequences, axis=0)
    
    # get confirmed, compatible & missed taxon sequences
    if len(whole) > 0:
        # there is at least one whole single rule, all concept sequences are confirmed & compatible, all outsider sequences are rejected
        in_compatible = tax_sequences
        confirmed = tax_sequences
        
    elif len(partial) > 0:
        # there are no whole single rules
        # check confirmed, compatible, missed taxon sequences
        in_compatible = np.any(in_matrix & (distinct_vals_encoded | shared_vals_encoded), axis=(1,2))
        in_missed = tax_sequences[~in_compatible]
        in_compatible = tax_sequences[in_compatible]
        confirmed = np.any(in_matrix & distinct_vals_encoded, axis=(1,2))
        confirmed = tax_sequences[confirmed]
    
        # check compatible outsider sequences
        # compatible outsider sequences -> any ousider sequences that contains a shared value in ALL partial sites
        #   Even if a single rule has unknown values, we are strict and reject sequences that show a value different than the one seen in the concept sequences
        #   has_unknowns = ~in_matrix.any(axis=2).all(axis=0)
        #   out_compatible = np.all(np.any(out_matrix & shared_vals_encoded, axis=2) | has_unknowns, axis=1)
        partial_sites = np.array(list(partial.keys()))
        out_compatible = np.all(np.any((out_matrix & shared_vals_encoded)[:, partial_sites], axis=2), axis=1)
        out_compatible = np.delete(np.arange(matrix.shape[0]), tax_sequences)[out_compatible]
    
    else:
        # if no single rules are found, all outsider sequences are compatible
        out_compatible = np.delete(np.arange(matrix.shape[0]), tax_sequences)
        
    return confirmed, in_compatible, in_missed, out_compatible

def prune(shared_matrix):
    
    # sort sites by least shared sequences
    indexes = np.arange(shared_matrix.shape[1])
    
    # keep track of sequences that are yet to be rejected
    # information = np.full((shared_matrix.shape[0], 1), True)
    to_reject = np.full(shared_matrix.shape[0], True)
    retain = []
    
    for _ in indexes:
        # detect if any of the remaining sites can discriminate a previously confused sequence
        # keep_sites = np.any(~shared_matrix[:, sorted_sites] & information, axis=0)
        new_rejections = ~shared_matrix[to_reject][:, indexes]
        new_rejections_any = np.any(new_rejections, axis=0)
        new_rejections_count = new_rejections.sum(axis=0)
        
        if not new_rejections_any.any():
            # no new information can be extracted from the remaining sites
            break
        
        # discard sites that add no new information
        indexes = indexes[new_rejections_any]
        new_rejections = new_rejections[:, new_rejections_any]
        new_rejections_count = new_rejections_count[new_rejections_any]
        
        # select site that rejects the most sequences
        best_site = np.argmax(new_rejections_count)
        best_site_index = indexes[best_site]
        indexes = np.delete(indexes, best_site)
        
        to_reject = to_reject & shared_matrix[:, best_site_index]
        retain.append(best_site_index)
        if not to_reject.any():
            # all sequences succesfully rejected
            break
        
    retain = np.array(retain, dtype=int)
    return retain, to_reject
    
def get_composite_rule(matrix, tax_sequences, shared_vals_encoded, site_types):
    composite_rule = {}
    out_matrix = np.delete(matrix, tax_sequences, axis=0)
    
    # get shared values from type 2 sites, select only those with a single shared site
    sites_2 = np.arange(matrix.shape[1])[site_types == 2]
    sites_2 = sites_2[shared_vals_encoded[sites_2].sum(axis=1) == 1]
    # get all type 3 sites
    sites_3 = np.arange(matrix.shape[1])[site_types == 3]
    
    site_candidates = np.concatenate((sites_2, sites_3))
    # get shared values
    shared_23 = shared_vals_encoded[site_candidates]
    
    # count shred values among outsider sequences
    shared_compatible = np.any(out_matrix[:, site_candidates] & shared_23, axis=2)
    
    # get composite rule sites and update out compatibles
    composite_sites, to_reject = prune(shared_compatible)
    composite_sites = site_candidates[composite_sites]
    # out_compatible = out_compatible[to_reject]
    
    # build composite rule
    nums = np.arange(1,5)
    composite_rule = {site:nums[shared_vals_encoded[site]] for site in composite_sites}
    
    return composite_rule #, out_compatible

def check_composite(composite, matrix, tax_sequences, shared_vals_encoded, single_out_compatible):
    # get compatible outsider sequences -> outsider sequences that share values in all composite rule sites
    composite_sites = np.array(list(composite.keys()), dtype=int)
    composite_encoded = shared_vals_encoded[composite_sites]
    out_matrix = np.delete(matrix, tax_sequences, axis=0)[:, composite_sites]
    
    # get outsider sequences compatible with the composite rule 
    out_compatible = np.all(np.any(out_matrix & composite_encoded, axis=2), axis=1)
    out_compatible = np.delete(np.arange(matrix.shape[0]), tax_sequences)[out_compatible]
    
    return out_compatible

#%% test single rules
def learn_concept_rules0(matrix, tax_sequences):
    # Build the necessary rulesets to fully solve the concept (if possible)
        # First: check if the concept can be solved by Single-Full ruleset
        # Second: check if the concept can be solved by Single-Partial ruleset
        # Third: check if the concpet can be solved by the Composite ruleset
        # Fourth: check if the concept can be solved by a combination of the Single-Partial and Composite rulesets
    # Returns ruleset dictionaries of key : value -> site index : encoded values
        # rules_full : Single-Full ruleset
        # rules_partial : Single-Partial ruleset
        # rules_composite : Composite ruleset
    # Also retunrs confirmed, out_compatible arrays
        # confirmed : shape (3, concept sequences), indicates which concept sequences are confirmed by each ruleset (Single-Full, Single-Partial, Composite)
        # out_compatible : shape (3, outsider sequences), indicates which outsider sequences cannot be rejected by each ruleset (Single-Full, Single-Partial, Composite)
    
    # get submatrices & out sequence indexes
    in_matrix = matrix[tax_sequences]
    out_matrix = np.delete(matrix, tax_sequences, axis=0)
    out_sequences = np.delete(np.arange(matrix.shape[0]), tax_sequences)
    
    # find distinct and shared sites, sites with null values & invariable sites
    distinct_values, shared_values, null_in, null_out, invariable = get_site_types_in(in_matrix, out_matrix)
    # get site types (including type 0 sites)
    types = get_site_types0(distinct_values, shared_values, null_in, null_out, invariable)
    has_type = types.any(axis=1)
    
    full_sites = []
    full_values = []
    partial_sites = []
    partial_values = []
    composite_sites = []
    composite_values = []
    
    confirmed = np.full((3, len(tax_sequences)), False)
    out_compatible = np.full((3, len(out_sequences)), True)
    
    # Concept can be solved by Single-Full ruleset?
    # concept has any type-1 sites?
    if has_type[1]:
        # get single full rules
        full_sites, full_values = get_single_full(types, distinct_values)
        confirmed[0] = True
        out_compatible[0] = False
    
    else:
        # Concept can be solved by Single-Partial ruleset?
        # concept has a set of type-2 sites that confirm all concept sequences/reject all outsider sequences
        if has_type[2]:
            # get single partial rules, check if they can solve the entire concept
            partial_sites, partial_values = get_single_partial(types, distinct_values, shared_values)
            confirmed_partial, out_compatible_partial = check_single_partial(partial_sites, in_matrix, out_matrix, distinct_values, shared_values)
            confirmed[1] = confirmed_partial
            out_compatible[1] = out_compatible_partial
        
        # concept has no Single-Partial ruleset or it can't confirm all concept sequences
        if not confirmed[1].all():
            
            # Concept can be solved by Composite ruleset?
            # has a set of type-3 sites that can reject all outsider sequences
            if has_type[3] and not confirmed_partial.all():
                # get composite rule, check if it can reject all outsider sequences
                composite_sites, composite_values, out_compatible_composite = get_composite(types, shared_values, out_matrix)
                out_compatible[2] = out_compatible_composite
                if not out_compatible_composite.any():
                    confirmed[2] = True
    # pack rulesets
    rules_full = {'sites':full_sites, 'values':full_values}
    rules_partial = {'sites':partial_sites, 'values':partial_values}
    rules_composite = {'sites':composite_sites, 'values':composite_values}
    return rules_full, rules_partial, rules_composite, confirmed, out_compatible
    
def learn_concept_rules(matrix, tax_sequences):
    # detect distinct & shared values
    distinct_vals, distinct_vals_encoded, distinct_counts, shared_vals, shared_vals_encoded, shared_counts = get_distinct_shared(matrix, tax_sequences)
    
    # typify sites
    site_types, type_summmary = get_site_types(matrix,
                                               distinct_vals,
                                               distinct_counts,
                                               shared_vals,
                                               shared_counts)
    
    # identify single rules
    single_whole, single_partial = get_single_rules(matrix,
                                                    tax_sequences,
                                                    site_types,
                                                    distinct_vals,
                                                    shared_vals)
    
    # check classification efficiency
    confirmed, in_compatible, in_missed, single_out_compatible = check_single(single_whole,
                                                                              single_partial,
                                                                              matrix,
                                                                              tax_sequences,
                                                                              distinct_vals_encoded,
                                                                              shared_vals_encoded)
    
    # generate composite rule if single rules cant reject all outsiders or confirm all sequences
    composite = {}
    out_compatible = single_out_compatible
    if len(out_compatible) > 0 or len(confirmed) < len(in_compatible):
        # get composite rule & update compatible outsider sequences
        composite = get_composite_rule(matrix,
                                       tax_sequences,
                                       shared_vals_encoded,
                                       site_types)
    
        composite_out_compatible = check_composite(composite,
                                                   matrix,
                                                   tax_sequences,
                                                   shared_vals_encoded,
                                                   out_compatible)
        out_compatible = np.intersect1d(single_out_compatible, composite_out_compatible)
        if len(out_compatible) == 0:
            confirmed = tax_sequences
    
    return single_whole, single_partial, composite, confirmed, in_compatible, in_missed, out_compatible
# confirmed, in_compatible, in_missed, out_compatible = learn_concept_rules(encoded, tax_seqs)
# single_whole, single_partial, composite, confirmed, in_compatible, in_missed, out_compatible = learn_concept_rules(encoded[:, sites], tax_seqs)
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

def parse_ruleset(**rules):
    origin_set = []
    rule_sites = []
    rule_values = []
    for origin, ruleset in rules.items():
        origin_set.append(np.full(len(ruleset['sites']), origin))
        rule_sites.append(ruleset['sites'])
        rule_values.append(ruleset['values'])
    origin_set = np.concatenate(origin_set)
    rule_sites = np.concatenate(rule_sites)
    rule_values = np.concatenate(rule_values, axis=1)
    return origin_set, rule_sites, rule_values

class Concept0:
    def __init__(self, name, rank=None):
        self.name = name
        self.rank = rank
        self.confirmed = [None, None, None]
        self.out_compatible = [None, None, None]
        self.solved = 'No'
        self.solved_by = 'None'
    
    @property
    def confirmed_full(self):
        return self.sequences[self.confirmed[0]]
    @property
    def confirmed_partial(self):
        return self.sequences[self.confirmed[1]]
    @property
    def confirmed_composite(self):
        return self.sequences[self.confirmed[2]]
    @property
    def confirmed_all(self):
        return self.sequences[self.confirmed.any(axis=0)]
    
    @property
    def compatible_full(self):
        return self.out_sequences[self.out_compatible[0]]
    @property
    def compatible_partial(self):
        return self.out_sequences[self.out_compatible[1]]
    @property
    def compatible_composite(self):
        return self.out_sequences[self.out_compatible[2]]
    @property
    def compatible_all(self):
        return self.out_sequences[self.out_compatible.all(axis=0)]
    
    def learn(self, matrix, concept_sequences, lineage_tab):
        self.sequences = concept_sequences
        self.out_sequences = np.delete(np.arange(matrix.shape[0]), concept_sequences)
        
        self.full_rules, self.partial_rules, self.composite_rules, confirmed, out_compatible = learn_concept_rules0(matrix, concept_sequences)
        self.confirmed = confirmed
        self.out_compatible = out_compatible
        
        confirmed_any = confirmed.any(axis=0)
        
        # check if the concept taxon is fully solved
        if confirmed_any.any():
            self.solved = 'Partial'
            if confirmed_any.all():
                self.solved = 'Full'
        
        # generate ruleset that best solves the concept
        confirmed_all = confirmed.all(axis=1)
        # record which ruleset solves the concept taxon (if any)
        if confirmed_all[0]:
            self.solved_by = 'Single-Full'
            origin_set, rule_sites, rule_values = parse_ruleset(Single_Full=self.full_rules)
        elif confirmed_all[1]:
            self.solved_by = 'Single-Partial'
            origin_set, rule_sites, rule_values = parse_ruleset(Single_Partial=self.partial_rules)
        elif confirmed_all[2]:
            self.solved_by = 'Composite'
            origin_set, rule_sites, rule_values = parse_ruleset(Composite=self.composite_rules)
        elif np.any(~out_compatible[1:], axis=0).all():
            self.solved_by = 'Single-Partial/Composite'
            origin_set, rule_sites, rule_values = parse_ruleset(Single_Partial=self.partial_rules, Composite=self.composite_rules)
        
        self.ruleset_origin = origin_set
        self.ruleset_sites = rule_sites
        self.ruleset_values = rule_values