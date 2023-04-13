#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 10:52:33 2023

@author: hernan
This module contains the functions used in collapsing of redundant subsequences
"""

import logging
import numpy as np
from time import time

#%% set logger
logger = logging.getLogger('Graboid.preprocessing.seq_collapsing')
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
        branch_cols = np.arange(matrix.shape[1]) != selected_col
        sub_matrix = matrix[val_indexes][:, branch_cols]
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

def get_overlapping(locations, indexes):
    # generate a list of arrays indicating all fully overlapping sequences
    # indexes contains the actual indexes of the sequences with n missing sites, spares having to correct afterwards
    
    loc_indexes = np.arange(locations.shape[0]) # use this to keep track of placed sequences
    n = locations[0].sum() # nuber of known sites in this group of seqs
    placed = set([]) # this will keep track of sequences that are already placed
    overlaps = []
    for idx, sequence in enumerate(locations):
        if idx in placed:
            # sequence is already placed, continue
            continue
        # get the columns corresponding to the current sequence
        sequence_cols = locations[:, sequence]
        # get the indexes of all rows with n (all) known values in the sequence's columns
        overlapping = loc_indexes[sequence_cols.sum(1) == n]
        overlaps.append(indexes[overlapping]) # add the REAL indexes to the overlapping list
        placed = placed.union(set(overlapping))
    return overlaps

#%%
def group_incomplete(incomplete_indexes, missing):
    # for each completeness tier, group sequences that with overlapped known sites
    if len(incomplete_indexes) == 0:
        # there are no incomplete tiers
        return []
    incomplete_grouped = [] # contains sequence groups in each tier as arrays of indexes
    for inc_tier in incomplete_indexes:
        if len(inc_tier) == 1:
            # single row with n missing
            incomplete_grouped.append([inc_tier])
            continue
        overlaps = get_overlapping(missing[inc_tier], inc_tier)
        incomplete_grouped.append(overlaps)
    return incomplete_grouped

def collapse_tiers(tiers, window, missing):
    # for each tier, collapse each group of fully overlapped sequences into branches
    tier_branches = []
    for tier in tiers:
        level_collapses = []
        for group in tier:
            if len(group) == 1:
                # single sequence in group, no need to collapse
                level_collapses.append(group)
                continue
            group_matrix = window[group][:, ~missing[group[0]]] # select the rows corresponding to the group from the window matrix, remove the unknown columns
            level_collapses += get_branches(group, group_matrix) # store the grouped indexes for this group in the level
        tier_branches.append(level_collapses)
    return tier_branches

def filter_branches(tier_branches, window, missing):
    # filter out branches that are contained at a lower tier
    if len(tier_branches) == 1:
        # only one tier, no need to filter
        return tier_branches
    
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
    
    # for each tier, group sequences that are completely overlapped
    grouped_tiers = []
    # locate complete tier
    incomplete_indexes = tier_indexes
    # remove first completeness tier if it is complete (place it to grouped_tiers)
    if unk_tiers[0] == 0:
        grouped_tiers = [[incomplete_indexes[0]]]
        incomplete_indexes = tier_indexes[1:]
    
    # group incomplete tiers
    grouped_tiers += group_incomplete(incomplete_indexes, missing)
    # generate branches for each tier
    tier_branches = collapse_tiers(grouped_tiers, window, missing)
    
    # filter subset branches
    kept_branches = filter_branches(tier_branches, window, missing)
    elapsed = time() - t0
    logger.debug(f'Collapsed {len(window)} into {len(kept_branches)} in {elapsed:.3f} seconds')
    return kept_branches
