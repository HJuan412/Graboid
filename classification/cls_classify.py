#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 14:40:44 2023

@author: hernan
Classification functions
"""

#%% libraries
import numpy as np

#%% functions
def unweighted(dists):
    weights = np.ones(dists.shape, dtype = np.int16)
    weights[dists < 0] = 0
    return weights

def wknn(dists):
    d1 = dists[:, 0]
    dk = np.max(dists, 1)
    
    weights = np.ones(dists.shape, dtype = np.float32)
    multi = d1 != dk # rows in which all neighbours are not equidistant to the query
    weights[multi] = (dk.reshape((-1,1)) - dists)[multi] / (dk - d1).reshape((-1,1))[multi]
    weights[dists < 0] = 0
    return weights

def wknn_rara(dists):
    # min distance is always the origin, max distance is the furthest neighbour + 1, questionable utility 
    d1 = 0
    dk = dists[-1] + 1
    if d1 == dk:
        return np.array([1])
    return (dk - dists) / (dk - d1)

def dwknn(dists):
    d1 = dists[:, 0]
    dk = np.max(dists, 1)
    penal = np.ones(dists.shape)
    multi = d1 != dk
    penal[multi] = (dk + d1).reshape((-1,1))[multi] / (dk.reshape((-1,1)) + np.clip(dists,0,np.inf))[multi]
    penal[dists < 0] = 0
    return wknn(dists) * penal

def get_tax_supports(taxa, supports):
    """Retrieves an array of taxa with their corresponding supports.
    Calculate total support for each taxon, filter out taxa with no support.
    Sort taxa by decreasing order of support"""
    uniq_taxa = np.unique(taxa)
    supports = np.array([supports[taxa == u_tax].sum() for u_tax in uniq_taxa])
    uniq_taxa = uniq_taxa[supports > 0]
    supports = supports[supports > 0]
    order = np.argsort(supports)[::-1]
    # normalize supports
    norm_supports = np.exp(supports)[order]
    norm_supports = norm_supports / norm_supports.sum()
    return uniq_taxa[order], supports[order], norm_supports
    
def classify(dists, positions, counts, neigh_idxs, tax_tab, weight_func):
    """Take the output list of one of the get_k_nearest... functions, the array
    of sorted neighbour distances indexes, the extended taxonomy table for the
    reference window, and one of the weighting functions."""
    # each rank's classifications are stored in a different list
    # each list contains a tuple containing an array of taxIDs and another with their corresponding total supports
    # classifications are sorted by decresing order of support
    rank_classifs = {rk:[] for rk in tax_tab.columns}
    for dist, pos, cnt, ngh_idxs in zip(dists, positions, counts, neigh_idxs):
        supports = weight_func(dist) # calculate support for each orbital
        support_array = np.concatenate([[supp]*count for supp, count in zip(supports, cnt)]) # extend supports to fit the number of neighbours per orbital
        # retrieve the neighbouring taxa
        all_neighs = ngh_idxs[pos[0,0]: pos[-1,-1]]
        neigh_taxa = tax_tab.iloc[all_neighs]
        # get total tax supports for each rank
        for rk, col in neigh_taxa.T.iterrows():
            valid_taxa = ~np.isnan(col.values)
            rank_classifs[rk].append(get_tax_supports(col.values[valid_taxa], support_array[valid_taxa]))
    return rank_classifs


def softmax(supports):
    exp_supports = np.exp(supports.astype(np.float128))
    return exp_supports / exp_supports.sum()

# vectorized functions, may replace the normal ones
def get_tax_supports_V(idx, taxa, supports, distances):
    """For each sequence, calcualte support for each taxon"""
    # idx is the index of the sequence being classified
    # taxa is a numpy array with the extended taxonomy of the sequence neighbours
    # supports is an array with the support of each neighbour
    # distances is an array with the distance of each neighbour to the sequence
    
    # returns two arrays
    # rk_tax_array has three columns: sequence_index, rank_index and taxon_id
    # rk_tax_data has five columns: total_neighbours, mean_distance, std_distance, total_support, norm_support
    
    rk_tax_array = []
    rk_tax_data = []
    taxa[np.isnan(taxa)] = -1 # use this to account for undetermined taxa
    
    for rk_idx, rk in enumerate(taxa.T):
        uniq_tax = np.unique(rk)
        n_tax = len(uniq_tax)
        rk_data = np.zeros((n_tax, 5), dtype=np.float16) # columns : total_neighs, mean_distance, std_distance, total_support, norm_support, 
        for tax_idx, u_tax in enumerate(uniq_tax):
            tax_loc = rk == u_tax
            rk_data[tax_idx, 0] = tax_loc.sum() # total neighbours
            rk_data[tax_idx, 1] = distances[tax_loc].mean() # mean distance
            rk_data[tax_idx, 2] = distances[tax_loc].std() # std distance
            rk_data[tax_idx, 3] = supports[tax_loc].sum() # total support
        
        # softmax normalize support
        rk_data[:, 4] = softmax(rk_data[:, 3])
        
        # sort by support
        order = np.argsort(rk_data[:,3])[::-1]
        rk_data = rk_data[order]
        rk_tax = uniq_tax[order]
        
        # remove undetermined
        rk_data = rk_data[rk_tax > 0]
        rk_tax = rk_tax[rk_tax > 0]
        
        # remove no support taxa
        supported = rk_data[:, 3] > 0
        rk_data = rk_data[supported]
        rk_tax = rk_tax[supported]
        
        # add squence and rank idxs
        rk_tax = np.array([np.full(len(rk_tax), idx), np.full(len(rk_tax), rk_idx), rk_tax]).T
        rk_tax_array.append(rk_tax)
        rk_tax_data.append(rk_data)
        
    rk_tax_array = np.concatenate(rk_tax_array).astype(int)
    rk_tax_data = np.concatenate(rk_tax_data)
    return rk_tax_array, rk_tax_data

def classify_V(distances, positions, counts, neigh_idxs, tax_tab, weight_func):
    """Calculates mean distances, support and normalized support for each candidate taxon for each sequence"""
    # dists contains the orbital distances of each query seq
    # positions contains the end position the last selected orbital for each query seq
    # counts contains the number of sequences in each orbital for each query seq
    # neigh_idxs contains the sorted indexes of the neighbouring sequences for each query seq
    # tax_tab is an extended taxonomy numpy array for the reference sequences
    # weight_func is one of unweighted, wknn, or dwknn
    
    # returns two arrays: the first one has columns: (query?)sequence_index, rank_index and taxon_id
    # the second array has columns: total_neighbours, mean_distances, std_distances, total_support and norm_support
    # second array uses dtype np.float16 to save memory space
    
    # calcuate supports
    supports = weight_func(distances)
    # extend supports and distances for orbit in each sequence
    support_arrays = []
    dist_arrays = []
    for supp, dists, cnts in zip(supports, distances, counts):
        support_arrays.append(np.concatenate([[spp]*cnt for spp, cnt in zip(supp, cnts)])) # extend supports to fit the number of neighbours per orbital
        dist_arrays.append(np.concatenate([[dst]*cnt for dst, cnt in zip(dists, cnts)]))
    # get the neighbour indexes for each sequence
    # neigh_arrays = [neigh_idxs[sq_idx, :max_loc] for sq_idx, max_loc in enumerate(np.max(positions, 1))]
    neigh_arrays = [neighs[:end_idx] for neighs, end_idx in zip(neigh_idxs, positions)]
    
    # get supports for each sequence
    id_arrays = []
    data_arrays = []
    for seq_idx, (neighs, supps, dists) in enumerate(zip(neigh_arrays, support_arrays, dist_arrays)):
        seq_supports = get_tax_supports_V(seq_idx, tax_tab[neighs], supps, dists)
        id_arrays.append(seq_supports[0])
        data_arrays.append(seq_supports[1])
    return np.concatenate(id_arrays), np.concatenate(data_arrays)