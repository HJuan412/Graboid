#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:36:58 2023

@author: nano

Classify test instances during calibration
"""

#%% libraries
import concurrent.futures
import numpy as np
# Graboid libraries
from classification import cls_classify
#%% functions
def get_tax_supports_V(taxa, u_supports, w_supports, d_supports, distances, counts, idx=0):
    # taxa is a numpy array with the extended taxonomy of the sequence neighbours
    # *_supports arrays contain the supports for each ORBITAL
    # distances array contains distances of each ORBITAL
    # counts array contains # neighbours in each ORBITAL
    # idx identifies the current sequence
    # returns 2d array of columns: idx, rank_id, tax_id, total_neighs, mean_distance, std_distance, total_unweighted_support, softmax_u_support, total_wknn_support, softmax_wknn_support, total_dwknn_support, softmax_d_support
    
    supports_tab =[]
    
    # expand distance and support vectors
    dist_ext = np.concatenate([[dist] * count for dist, count in zip(distances, counts)])
    u_supp_ext = np.concatenate([[u_supp] * count for u_supp, count in zip(u_supports, counts)])
    w_supp_ext = np.concatenate([[w_supp] * count for w_supp, count in zip(w_supports, counts)])
    d_supp_ext = np.concatenate([[d_supp] * count for d_supp, count in zip(d_supports, counts)])
    
    # get supports for each taxon for each rank
    for rk_idx, rk in enumerate(taxa.T):
        uniq_tax = np.unique(rk)
        n_tax = len(uniq_tax)
        rk_data = np.zeros((n_tax, 9), dtype=np.float32) # columns : total_neighs, mean_distance, std_distance, total_unweighted_support, softmax_u_support, total_wknn_support, softmax_wknn_support, total_dwknn_support, softmax_d_support
        for tax_idx, u_tax in enumerate(uniq_tax):
            tax_loc = rk == u_tax
            if sum(tax_loc) == 0:
                # no predictions for current taxon
                continue
            rk_data[tax_idx, 0] = tax_loc.sum() # total neighbours
            rk_data[tax_idx, 1] = dist_ext[tax_loc].mean() # mean distance
            rk_data[tax_idx, 2] = dist_ext[tax_loc].std() # std distance
            rk_data[tax_idx, 3] = u_supp_ext[tax_loc].sum() # total unweighted support
            rk_data[tax_idx, 5] = w_supp_ext[tax_loc].sum() # total wknn support
            rk_data[tax_idx, 7] = d_supp_ext[tax_loc].sum() # total dwknn support
        
        # softmax normalize support
        rk_data[:, 4] = cls_classify.softmax(rk_data[:, 3])
        rk_data[:, 6] = cls_classify.softmax(rk_data[:, 5])
        rk_data[:, 8] = cls_classify.softmax(rk_data[:, 7])
        
        # remove no support taxa
        supported = rk_data[:, 3] > 0
        rk_data = rk_data[supported]
        rk_tax = uniq_tax[supported]
        
        # add squence and rank idxs
        rk_tax = np.array([np.full(len(rk_tax), idx), np.full(len(rk_tax), rk_idx), rk_tax]).T
        
        # build support tab for current rank
        supports_tab.append(np.concatenate((rk_tax, rk_data), axis = 1))
    
    return np.concatenate(supports_tab).astype(np.float32)

def classify_V(package, tax_tab, threads=1):
    """Variant of the cls_classify classify_V function, calculates support using all methods(unweighted, wknn, dwknn).
    Calculates mean distances, support and normalized support for each candidate taxon for each sequence"""
    # tax_tab is an extended taxonomy numpy array for the reference sequences
    # package[0] is a 3d array of shape (# seqs, 3, n)
    # package[1] is a 2d array containing the sorted indexes of the neighbouring sequences
    # returns 2d array of columns: idx, rank_id, tax_id, total_neighs, mean_distance, std_distance, total_unweighted_support, softmax_u_support, total_wknn_support, softmax_wknn_support, total_dwknn_support, softmax_d_support
    
    # unpack classification package
    distances = package[0][:,0] # distances contains the orbital distances of each query seq
    positions = package[0][:,1] # positions contains the START positions of each orbital for each query seq (to get END positions add corresponding coutns)
    counts = package[0][:,2] # counts contains the number of sequences in each orbital for each query seq
    neigh_idxs = package[1] # neigh_idxs contains the sorted indexes of the neighbouring sequences for each query seq shape (#seqs, #seqs - 1)
    max_locs = counts.sum(1).astype(int) # get the maximum position for the neighbour indexes
    
    # calcuate supports
    u_supports = cls_classify.unweighted(distances)
    w_supports = cls_classify.wknn(distances)
    d_supports = cls_classify.dwknn(distances)
    
    # get the neighbour indexes for each sequence list of arrays (# neighbours varies per sequence)
    neigh_arrays = [neigh_idxs[sq_idx, :max_loc] for sq_idx, max_loc in enumerate(max_locs)]
    
    # get supports for each sequence
    supports_tab = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(get_tax_supports_V, tax_tab[neighs], u_supps, w_supps, d_supps, dists, cnts.astype(int), seq_idx) for seq_idx, (neighs, u_supps, w_supps, d_supps, dists, cnts) in enumerate(zip(neigh_arrays, u_supports, w_supports, d_supports, distances, counts))]
        for future in concurrent.futures.as_completed(futures):
            supports_tab.append(future.result())
    return np.concatenate(supports_tab)

def get_supports(id_array, data_array, tax_tab):
    """Get best support and support for the real taxon for every classified sequence for every rank"""
    # id_array contains columns seq_idx, rk_id, tax_id
    # data_array contains columns total_neighbours, mean_distance, std_distance, unweighted_support, softmax_u_support, wknn_support, softmax_w_support, dwknn_support, softmax_d_support
    # tax_tab is an array containing the complete taxon ids for the query sequences
    # for each support function, return two arrays of shape (seq_idx, #ranks)
        # first array contains the predicted tax_id for each sequence per rank
        # second array contains the supports for the real taxon for each rank in each sequence
        
    uniq_seqs = np.unique(id_array[:,0])
    uniq_rks = np.unique(id_array[:,1])
    
    predicted_u = np.zeros((len(uniq_seqs), len(uniq_rks)), dtype=int)
    real_u_support = np.zeros((len(uniq_seqs), len(uniq_rks)), dtype=np.float32)
    
    predicted_w = np.zeros((len(uniq_seqs), len(uniq_rks)), dtype=int)
    real_w_support = np.zeros((len(uniq_seqs), len(uniq_rks)), dtype=np.float32)
    
    predicted_d = np.zeros((len(uniq_seqs), len(uniq_rks)), dtype=int)
    real_d_support = np.zeros((len(uniq_seqs), len(uniq_rks)), dtype=np.float32)
    
    for seq_idx, seq in enumerate(uniq_seqs):
        seq_loc = id_array[:,0] == seq
        seq_classif = id_array[seq_loc]
        seq_data = data_array[seq_loc]
        seq_real_taxa = tax_tab[seq_idx]
        
        # get predicted taxon
        for rk_idx, rk in enumerate(uniq_rks):
            rk_loc = seq_classif[:, 1] == rk
            rk_data = seq_data[rk_loc]
            rk_taxa = seq_classif[rk_loc, 2]
            
            predicted_u[seq_idx, rk_idx] = rk_taxa[np.argmax(rk_data[:,4])]
            predicted_w[seq_idx, rk_idx] = rk_taxa[np.argmax(rk_data[:,6])]
            predicted_d[seq_idx, rk_idx] = rk_taxa[np.argmax(rk_data[:,8])]
            
            # get supports for real taxa
        for rk_idx, rk_tax in enumerate(seq_real_taxa):
            real_loc = seq_classif[:,-1] == rk_tax
            if real_loc.sum() == 0:
                # real taxon has no support
                real_u_support[seq_idx, rk_idx] = 0
                real_w_support[seq_idx, rk_idx] = 0
                real_d_support[seq_idx, rk_idx] = 0
                continue
            real_u_support[seq_idx, rk_idx] = seq_data[real_loc, 4]
            real_w_support[seq_idx, rk_idx] = seq_data[real_loc, 6]
            real_d_support[seq_idx, rk_idx] = seq_data[real_loc, 8]
    return predicted_u, real_u_support, predicted_w, real_w_support, predicted_d, real_d_support

### NEW DET SUPPORTS
# rk_idx = 3

# # get supports for a given RANK
# rk_supports = supports_tab[supports_tab[:,1] == rk_idx]

# # get best support for each training instance

# # sorted_*: sorted indexes (descending) for all instances
# sorted_u = np.argsort(rk_supports[:, 6])[::-1]
# sorted_w = np.argsort(rk_supports[:, 8])[::-1]
# sorted_d = np.argsort(rk_supports[:, 10])[::-1]

# # get the location of the best score for all instances
# # np.unique(rk_supports[sorted_*, 0], return_index=True)[1] gets the first position for every SORTED sequence index in the support table (sorted by a given method)
# # retrieve first indexes for every sequence (sorted_*[...])
# best_u_locs = sorted_u[np.unique(rk_supports[sorted_u, 0], return_index=True)[1]]
# best_w_locs = sorted_w[np.unique(rk_supports[sorted_w, 0], return_index=True)[1]]
# best_d_locs = sorted_d[np.unique(rk_supports[sorted_d, 0], return_index=True)[1]]

# # extract sequence index + predicted taxon
# pred_u = rk_supports[best_u_locs][:, [0,2]]
# pred_w = rk_supports[best_w_locs][:, [0,2]]
# pred_d = rk_supports[best_w_locs][:, [0,2]]

# # get support of real taxa
# # get unique taxa
# rk_taxa = np.unique(rk_supports.T[2][~np.isnan(rk_supports.T[2])])
# # get indexes of represented sequences
# supp_indexes = np.unique(rk_supports[:,0]).astype(int)
# # prepare support vectors
# true_support_u = np.full(supp_indexes.max(), -1, dtype=np.float32)
# true_support_w = np.full(supp_indexes.max(), -1, dtype=np.float32)
# true_support_d = np.full(supp_indexes.max(), -1, dtype=np.float32)

# # get true support per taxon
# for tax in rk_taxa:
#     # locate instances of taxon
#     tax_instances = supp_indexes[win_tax.T[rk_idx] == tax]
#     # extract calculated supports for taxon, then filter by instances belonging to taxon
#     tax_supports = rk_supports[rk_supports[:,2] == tax]
#     tax_supports = tax_supports[np.isin(tax_supports[:,0], tax_instances)]
    
#     # set default support as 0 (differenctiate from unclear taxa in db, with score of -1)
#     true_support_u[tax_instances] = 0
#     true_support_w[tax_instances] = 0
#     true_support_d[tax_instances] = 0
    
#     # update supports with normalized scores
#     true_support_u[tax_supports[:,0].astype(int)] = tax_supports[:, 7]
#     true_support_w[tax_supports[:,0].astype(int)] = tax_supports[:, 9]
#     true_support_d[tax_supports[:,0].astype(int)] = tax_supports[:, 11]