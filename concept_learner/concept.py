#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 11:49:07 2024

@author: hernan
"""
#%% modules
import numpy as np

#%% functions
def get_site_types_in(in_matrix, out_matrix):
    # locate distinct, shared, missing (in and out) values, as well as invariable sites
    # returns encoded matrices of shape:
        # distinct_values, shared_values : (5, sites)
        # in_missing, out_missing, invariable : (sites)
        
    # get values found in the inner and outer matrices
    in_values = in_matrix.any(axis=0)
    out_values = out_matrix.any(axis=0)
    
    # get sites with missing values in the inner and outer matrices
    in_missing = in_values[:, 0]
    out_missing = out_values[:, 0]
    
    # get encoded distinct and shared values
    distinct_values = in_values & ~out_values
    shared_values = in_values & out_values
    
    # get invariable sites
    invariable = np.concatenate((in_matrix, out_matrix), axis=0).any(axis=0).sum(axis=1) == 1
    return distinct_values, shared_values, in_missing, out_missing, invariable

def get_site_types0(distinct_values, shared_values, invariable):
    # process output of get_site_types_in to typify the type of each site in the alignment
    # returns encoded sites array of shape (5, sites), indicating with a boolean the type of each site
    
    has_distinct = distinct_values.any(axis=1)
    has_shared = shared_values.any(axis=1)
    single_shared = shared_values.sum(axis=1) <= 1
    
    type0 = invariable
    type1 = has_distinct & ~has_shared
    type2 = has_distinct & has_shared
    type3 = ~has_distinct & single_shared & ~invariable
    type4 = ~type0 & ~type1 & ~type2 & ~type3
    
    types = np.array([type0, type1, type2, type3, type4])
    return types

def get_single_full(types, distinct_values):
    # get single full rules, locate sites that show only distinctive values
    # returns dictionary of key:values
        # site index : encoded values
    single_full_sites = np.arange(distinct_values.shape[0])[types[1]]
    single_full_values = distinct_values[single_full_sites]
    return single_full_sites, single_full_values

def get_single_partial(types, distinct_values, shared_values):
    # get single partial rules, locate sites that have distinctive and shared values
    # returns dictinoary of key:values
        # site index : encoded values (includes distinctive and shared values)
    single_partial_sites = np.arange(distinct_values.shape[0])[types[2]]
    single_partial_values = distinct_values[single_partial_sites] | shared_values[single_partial_sites]
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
        remaining_rejections = out_rejected[~rejected][:, sites].any(axis=0)
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
    
    # get type 3 sites
    site3_indexes = np.arange(shared_values.shape[0])[types[3]]
    
    # get shared values with the outsider sequences
    # build shared matrix
    out_shared = (out_matrix[:, site3_indexes] & shared_values[site3_indexes]).any(axis=2)
    
    # prune redundant sites
    pruned = prune0(out_shared)
    
    # build ruleset and list compatible outsider sequences
    composite_sites = site3_indexes[pruned]
    composite_values = shared_values[composite_sites]
    out_compatible = out_shared.all(axis=1)
    
    return composite_sites, composite_values, out_compatible

def tipify(in_matrix, out_matrix):
    """
    Detect sites with distinct, shared, missing values inside/outside the concept taxon and determine the type of each site.

    Parameters
    ----------
    in_matrix : numpy.array
        Subsection of the reference alignment including the concept taxon's representative sequences.
    out_matrix : numpy.array
        Subsection of the reference alignment including the sequences outsider to the concept taxon.

    Returns
    -------
    distinct_values : numpy.array
        Boolean array indicating the values that are distinctive to the concept taxon in each site of the alignment.
    shared_values : numpy.array
        Boolean array indicating the values that are shared  with sequences outside the concept taxon in each site of the alignment.
    in_missing : numpy.array
        Boolean array indicating which sites of the alignment present missing values inside the concept taxon.
    out_missing : numpy.array
        Boolean array indicating which sites of the alignment present missing values outside the concept taxon.
    invariable : numpy.array
        Boolean array indicating which sites of the alignment present a single value.
    types : numpy.array
        Boolean array indicating the type of each site with respect to the concept taxon.

    """
    distinct_values, shared_values, in_missing, out_missing, invariable = get_site_types_in(in_matrix, out_matrix)
    types = get_site_types0(distinct_values, shared_values, invariable)
    return distinct_values, shared_values, in_missing, out_missing, invariable, types

def merge_rulesets(full_sites, full_values, partial_sites, partial_values, composite_sites, composite_values):
    total_rules = len(full_sites) + len(partial_sites) + len(composite_sites)
    origin_ruleset = np.full((3, total_rules), False)
    
    origin_ruleset[0, :len(full_sites)] = True
    origin_ruleset[1, len(full_sites):len(full_sites) + len(partial_sites)] = True
    origin_ruleset[2, len(full_sites)+len(partial_sites):] = True
    
    rule_sites = np.concatenate((full_sites, partial_sites, composite_sites))
    rule_values = np.concatenate((full_values, partial_values, composite_values), axis=0)
    return rule_sites, rule_values, origin_ruleset

def get_ruleset(in_matrix, out_matrix, types, distinct_values, shared_values):
    full_sites = np.array([], dtype=int)
    full_values = np.empty((0,5), dtype=bool)
    partial_sites = np.array([], dtype=int)
    partial_values = np.empty((0,5), dtype=bool)
    composite_sites = np.array([], dtype=int)
    composite_values = np.empty((0,5), dtype=bool)
    
    has_type = types.any(axis=1)
    confirmed = np.full((3, in_matrix.shape[0]), False)
    out_compatible = np.full((3, out_matrix.shape[0]), True)
    
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
            if has_type[3]:
                # get composite rule, check if it can reject all outsider sequences
                composite_sites, composite_values, out_compatible_composite = get_composite(types, shared_values, out_matrix)
                out_compatible[2] = out_compatible_composite
                if not out_compatible_composite.any():
                    confirmed[2] = True
    # pack rulesets
    rule_sites, rule_values, rule_origin = merge_rulesets(full_sites, full_values, partial_sites, partial_values, composite_sites, composite_values)
    return rule_sites, rule_values, rule_origin, confirmed, out_compatible

#%% classes
class Concept:
    def __init__(self, name, rank=None):
        # set concept metadata
        self.name = name
        self.rank = rank
        
        self.sequences = np.array([], dtype=int)
        self.out_sequences = np.array([], dtype=int)
        self.solved = 'No'
        self.n_rules = 0
        self.rule_indexes = np.array([], dtype=int)
        self.rule_sites = np.array([], dtype=int)
        self.rule_values = np.empty((0,5), dtype=bool)
        self.ruleset_origin = np.empty((0,3), dtype=bool)
        self.confirmed = np.empty((0,3), dtype=bool)
        self.out_compatible = np.empty((0,3), dtype=bool)
        self.types = np.full(5, 0, dtype=int)
        self.confirmed_seqs = np.array([], dtype=int)
        self.out_compatible_seqs = np.array([], dtype=int)
    
    def set_rule_indexes(self, offset):
        self.rule_indexes = np.arange(self.n_rules, dtype=int) + offset
    
    def learn(self, matrix, concept_sequences):
        # store concept sequences & out_sequences
        self.sequences = concept_sequences
        self.out_sequences = np.delete(np.arange(matrix.shape[0]), concept_sequences)
        
        # detect site types
        distinct_values, shared_values, in_missing, out_missing, invariable, types = tipify(matrix[self.sequences], matrix[self.out_sequences])
        
        # define rulesets
        self.rule_sites, self.rule_values, self.ruleset_origin, self.confirmed, self.out_compatible = get_ruleset(matrix[self.sequences], matrix[self.out_sequences], types, distinct_values, shared_values)
        self.n_rules = len(self.rule_sites)
        
        # log type counts
        self.types = types.sum(axis=1)
        # register confirmed & compatible sequences
        self.confirmed_seqs = self.sequences[self.confirmed.any(axis=0)]
        self.out_compatible_seqs = self.out_sequences[self.out_compatible.all(axis=0)]
        
        # log solved status
        if len(self.confirmed_seqs) == len(self.sequences) or len(self.out_compatible_seqs) == 0:
            self.solved = 'Full'
        elif len(self.confirmed_seqs) > 0 or len(self.out_compatible_seqs) < len(self.out_sequences):
            self.solved = 'Partial'
    
    def get_signal(self, query):
        signal_values = (query[:, self.rule_sites] & self.rule_values).any(axis=2).sum(axis=1).astype(np.int16)
        return signal_values