#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 10:52:33 2023

@author: hernan
This module contains the functions used in collapsing of redundant subsequences
"""

import concurrent.futures
import logging
import numpy as np
from time import time

#%% set logger
logger = logging.getLogger('Graboid.preprocessing.seq_collapsing')
logger.setLevel(logging.DEBUG)
#%%
def entropy(matrix):
    entropy = np.zeros(matrix.shape[1])
    for idx, col in enumerate(matrix.T):
        valid_rows = col[col != 0] # only count known values
        values, counts = np.unique(valid_rows, return_counts = True)
        n_rows = counts.sum()
        freqs = counts / n_rows
        entropy[idx] = -np.sum(np.log2(freqs) * freqs)
    return entropy

# get branches meant to work only with non empty sequences
def get_branches(sequences, matrix):
    # collapse the matrix into its disticnt rows, return a list of the grouped indexes
    # sequences keeps track of the indexes of the sequences as they are assigned to branches
    
    branches = []
    
    # if there is a single row in the matrix, return a single branch
    if matrix.shape[0] == 1:
        branches.append(sequences[[0]])
        return branches
    
    # if there is a single column remaining, split it into the corresponding branches and stop recursion
    if matrix.shape[1] == 1:
        for val in np.unique(matrix):
            # generate a sub branch for each value
            val_indexes = matrix[:, 0] == val
            branches.append(sequences[val_indexes])
        return branches
    
    # select the column with the most entropy
    site_entropy = entropy(matrix)
    selected_col = np.argsort(site_entropy)[::-1][0]
    # split sequences by their value in the selected column
    values_in_col = np.unique(matrix[:, selected_col])
    if len(values_in_col) == 1:
        # if there is a single known value in the selected column it is safe to assume the branch is exhausted
        # this is because the selected column is the most informative one in the alignment, all the rest have equal or less diversity
        return [sequences]
    
    for val in values_in_col:
        # generate a sub branch for each value
        val_indexes = matrix[:, selected_col] == val
        branch_seqs = sequences[val_indexes]
        if len(branch_seqs) == 1:
            # there is a single sequence in this potential branch, no need to explore it
            branches.append(branch_seqs)
            continue
        # remove the column used to split the sequence from the following matrices
        # This is technically not necesary because the column has a single value, and therefore entropy of 0 for every subsequent branch
        # still, it's one less column for the future entropy calculations
        sub_matrix = np.delete(matrix[val_indexes], selected_col, 1)
        branches += get_branches(branch_seqs, sub_matrix)
    return branches

def seq_in_group(seq, group, group_entropy):
    # check if seq is contained among the rows in group
    sorted_entropy = np.argsort(group_entropy)
    variable_cols = sorted_entropy[group_entropy[sorted_entropy] > 0] # any column that has an entropy of 0 can't be used to split the group
    static_cols = sorted_entropy[group_entropy[sorted_entropy] == 0]
    sub_group = group
    # iterate over the variable columns first, in descending order of entropy
    for col in variable_cols[::-1]:
        seq_val = seq[col]
        sub_group = sub_group[sub_group[:, col] == seq_val]
        if sub_group.shape[0] == 0:
            return False
    # seq wasn't separated by the variable sites, check static sites
    return (seq[static_cols] != sub_group[0, static_cols]).sum() == 0

def get_overlapping(locations, tier_indexes):
    # group the sequences that share the same known sites (not necesarily the same sequences)
    # return a list of arrays containing the indexes of grouped sequences
    
    # locations is a boolean matrix of known sites
    # tier_indexes contains the actual indexes of the sequences with n missing sites, spares having to correct afterwards
    
    ungrouped = np.arange(locations.shape[0]) # use this to keep track of grouped sequences
    n = locations[0].sum() # nuber of known sites in this group of seqs
    overlaps = []
    
    while len(ungrouped) > 0:
        sequence = locations[ungrouped[0]] # get the known columns for the first remaining ungrouped sequences
        # get the columns known in sequence
        sequence_cols = locations[ungrouped][:, sequence]
        # get the indexes of all rows with n (all) known values in the sequence's columns
        overlapping = sequence_cols.sum(1) == n
        grouped = ungrouped[overlapping]
        ungrouped = ungrouped[~overlapping]
        overlaps.append(tier_indexes[grouped]) # add the REAL indexes to the overlapping list
    return overlaps

def group_tiers(tier_indexes, missing):
    # for each completeness tier, group sequences with overlapped known sites
    grouped = [] # contains sequence groups in each tier as arrays of indexes
    for tier in tier_indexes:
        if len(tier) == 1:
            # single row with n missing
            grouped.append([tier])
            continue
        overlaps = get_overlapping(missing[tier], tier)
        grouped.append(overlaps)
    return grouped

def collapse_tiers_parallel(tiers, matrix, missing, threads=1):
    # for each tier, collapse each group of fully overlapped sequences into branches
    tier_branches = []
    for tier in tiers:
        tier_collapses = [] # store groups of collapsed (groups of equal sequences)
        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
            # select the rows corresponding to the group from the matrix, remove the unknown columns
            futures = [executor.submit(get_branches, group,matrix[group][:, ~missing[group[0]]]) for group in tier]
            for future in concurrent.futures.as_completed(futures):
                tier_collapses.append(future.result()) # store the grouped indexes for this group in the level
        tier_branches.append(tier_collapses)
    return tier_branches

def collapse_tiers(tiers, matrix, missing):
    # for each tier, collapse each group of fully overlapped sequences into branches
    tier_branches = []
    for tier in tiers:
        tier_collapses = [] # store groups of collapsed (groups of equal sequences)
        for group in tier:
            if len(group) == 1:
                # single sequence in group, no need to collapse
                tier_collapses.append(group)
                continue
            group_matrix = matrix[group][:, ~missing[group[0]]] # select the rows corresponding to the group from the matrix, remove the unknown columns
            tier_collapses += get_branches(group, group_matrix) # store the grouped indexes for this group in the level
        tier_branches.append(tier_collapses)
    return tier_branches

def filter_branches(tier_branches, window, missing):
    # filter out branches that are contained at a lower tier
    if len(tier_branches) == 1:
        # only one tier, no need to filter
        return tier_branches[0]
    
    # find a representative for each branch at each tier
    branch_reprs = []
    for br_tier in tier_branches:
        # keep branch representatives grouped by tier
        tier_branch_reps = [branch[0] for branch in br_tier]
        branch_reprs.append(tier_branch_reps)
        
    kept_branches = tier_branches[0].copy() # the lowest branch is included by default
    kept_repr = branch_reprs[0].copy()
    for rep_tier, branch_tier in zip(branch_reprs[1:], tier_branches[1:]):
        lower_mat = window[kept_repr]
        lower_entropy = entropy(lower_mat)
        for rep, branch in zip(rep_tier, branch_tier):
            rep_cols = ~missing[rep]
            rep_seq = window[rep, rep_cols]
            group_mat = lower_mat[:, rep_cols]
            group_entropy = lower_entropy[rep_cols]
            if seq_in_group(rep_seq, group_mat, group_entropy):
                continue
            kept_branches.append(branch)
            kept_repr.append(rep)
    return kept_branches

def sequence_collapse(matrix, max_unk_thresh=0.2):
    #TODO: this will be the new main collapsing function, to replace collapse_window
    # We will use set theory to cluster the collapsed sequences:
    #     Two sequences A and B are equal (A = B) if they have the same known sites with no differing values
    #     Two sequences A and B dont intersect (A ∩ B = Ø) when they have differing values in at least one site known to both, or they have no common known sites
    #     Two sequences A and B intersect (A ∩ B ≠ Ø) when they have no differing values in any of their common known sites, but both have known sites that are unknown in the other
    #     Sequence A contains B (A ⊃ B) when all of Bs known sites are known in A, A has more known sites, and no differing values are present
    
    # filter out rows that have more than <max_unk_thresh> unknown sites
    max_unks = matrix.shape[1] * max_unk_thresh
    
    # we begin by grouping the sequence by completeness level
    missing = matrix == 0
    unk_count = missing.sum(1)
    row_indexes = np.arange(matrix.shape[0])
    unk_tiers = np.unique(unk_count)
    unk_tiers = unk_tiers[unk_tiers <= max_unks]
    tier_indexes = [row_indexes[unk_count == uk_tier] for uk_tier in unk_tiers]
    
    # if no rows passed the filter (too many unknowns), raise an exception
    if len(tier_indexes) == 0:
        raise Exception(f'No rows passed the max unkown filter: {max_unk_thresh}, max unknown sites: {max_unks}')
    # group sequences with matching known sites for each completeness tier
    # sequences in a same group are the ones that can be equal (same known places)
    grouped_tiers = group_tiers(tier_indexes, missing)
    
    # generate branches for each tier
    tier_branches = collapse_tiers(grouped_tiers, matrix, missing)
    
    # filter subset branches
    kept_branches = filter_branches(tier_branches, matrix, missing)
    
    # build collapseed matrix
    reprs = [r[0] for r in kept_branches]
    collapsed_mat = matrix[reprs]
    return collapsed_mat, kept_branches

def collapse_window(window):
    # We will use set theory to cluster the collapsed sequences:
    #     Two sequences A and B are equal (A = B) if they have the same known sites with no differing values
    #     Two sequences A and B dont intersect (A ∩ B = Ø) when they have differing values in at least one site known to both, or they have no common known sites
    #     Two sequences A and B intersect (A ∩ B ≠ Ø) when they have no differing values in any of their common known sites, but both have known sites that are unknown in the other
    #     Sequence A contains B (A ⊃ B) when all of Bs known sites are known in A, A has more known sites, and no differing values are present
    
    # this function can't handle full empty rows, make sure to filter the window before collapsing
    
    # we begin by grouping the sequence by completeness level
    t0 = time()
    missing = window == 0
    unk_count = missing.sum(1)
    row_indexes = np.arange(window.shape[0])
    unk_tiers = np.unique(unk_count)
    tier_indexes = [row_indexes[unk_count == uk_tier] for uk_tier in unk_tiers]
    
    # group sequences with matching known sites for each completeness tier
    grouped_tiers = group_tiers(tier_indexes, missing)
    
    # generate branches for each tier
    tier_branches = collapse_tiers(grouped_tiers, window, missing)
    
    # filter subset branches
    kept_branches = filter_branches(tier_branches, window, missing)
    elapsed = time() - t0
    logger.debug(f'Collapsed {len(window)} into {len(kept_branches)} in {elapsed:.3f} seconds')
    return kept_branches
