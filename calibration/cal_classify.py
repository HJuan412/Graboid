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
def get_tax_supports_V(idx, taxa, u_supports, w_supports, d_supports, distances):
    """Variant of the cls_classify classification function, calculates support using all methods(unweighted, wknn, dwknn)
    For each sequence, calcualte support for each taxon"""
    # idx is the index of the sequence being classified
    # taxa is a numpy array with the extended taxonomy of the sequence neighbours
    # supports is an array with the support of each neighbour
    # distances is an array with the distance of each neighbour to the sequence
    
    # returns two arrays
    # rk_tax_array has three columns: sequence_index, rank_index and taxon_id
    # rk_tax_data has 9 columns: total_neighbours, mean_distance, std_distance, unweighted_support, softmax_u_support, wknn_support, softmax_w_support, dwknn_support, softmax_d_support
    
    rk_tax_array = []
    rk_tax_data = []
    taxa[np.isnan(taxa)] = -1 # use this to account for undetermined taxa
    
    for rk_idx, rk in enumerate(taxa.T):
        uniq_tax = np.unique(rk)
        n_tax = len(uniq_tax)
        rk_data = np.zeros((n_tax, 9), dtype=np.float32) # columns : total_neighs, mean_distance, std_distance, total_unweighted_support, softmax_u_support, total_wknn_support, softmax_wknn_support, total_dwknn_support, softmax_d_support
        for tax_idx, u_tax in enumerate(uniq_tax):
            tax_loc = rk == u_tax
            rk_data[tax_idx, 0] = tax_loc.sum() # total neighbours
            rk_data[tax_idx, 1] = distances[tax_loc].mean() # mean distance
            rk_data[tax_idx, 2] = distances[tax_loc].std() # std distance
            rk_data[tax_idx, 3] = u_supports[tax_loc].sum() # total unweighted support
            rk_data[tax_idx, 5] = w_supports[tax_loc].sum() # total wknn support
            rk_data[tax_idx, 7] = d_supports[tax_loc].sum() # total dwknn support
        
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
        rk_tax_array.append(rk_tax)
        rk_tax_data.append(rk_data)
        
    rk_tax_array = np.concatenate(rk_tax_array).astype(int)
    rk_tax_data = np.concatenate(rk_tax_data)
    return rk_tax_array, rk_tax_data

def classify_V(package, tax_tab, threads=1):
    """Variant of the cls_classify classify_V function, calculates support using all methods(unweighted, wknn, dwknn).
    Calculates mean distances, support and normalized support for each candidate taxon for each sequence"""
    # tax_tab is an extended taxonomy numpy array for the reference sequences
    # weight_func is one of unweighted, wknn, or dwknn
    
    # returns two arrays: the first one has columns: (query?)sequence_index, rank_index and taxon_id
    # the second array has columns: total_neighbours, mean_distance, std_distance, unweighted_support, softmax_u_support, wknn_support, softmax_w_support, dwknn_support, softmax_d_support
    # second array uses dtype np.float16 to save memory space
    
    # unpack classification package
    distances = package[0][0]
    positions = package[0][1]
    counts = package[0][2]
    neigh_idxs = package[1]
    # dists contains the orbital distances of each query seq
    # positions contains the start and end positions of each orbital for each query seq
    # counts contains the number of sequences in each orbital for each query seq
    # neigh_idxs contains the sorted indexes of the neighbouring sequences for each query seq
    
    # calcuate supports
    u_supports = cls_classify.unweighted(distances)
    w_supports = cls_classify.wknn(distances)
    d_supports = cls_classify.dwknn(distances)
    
    # extend supports and distances for orbit in each sequence
    u_support_arrays = []
    w_support_arrays = []
    d_support_arrays = []
    dist_arrays = []
    
    def expand(supp_array, counts):
        return np.concatenate([[supp]*cnt for supp, cnt in zip(supp_array, counts)])
        
    for u_supps, w_supps, d_supps, dists, cnts in zip(u_supports, w_supports, d_supports, distances, counts):
        # extend supports to fit the number of neighbours per orbital
        u_support_arrays.append(expand(u_supps, cnts))
        w_support_arrays.append(expand(w_supps, cnts))
        d_support_arrays.append(expand(d_supps, cnts))
        dist_arrays.append(expand(dists, cnts))
        # dist_arrays.append(np.concatenate([[dst]*cnt for dst, cnt in zip(dists, counts)]))
    # get the neighbour indexes for each sequence
    neigh_arrays = [neigh_idxs[sq_idx, :max_loc] for sq_idx, max_loc in enumerate(np.max(positions, 1))]
    
    # get supports for each sequence
    id_arrays = []
    data_arrays = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(get_tax_supports_V, seq_idx, tax_tab[neighs], u_supps, w_supps, d_supps, dists) for seq_idx, (neighs, u_supps, w_supps, d_supps, dists) in enumerate(zip(neigh_arrays, u_support_arrays, w_support_arrays, d_support_arrays, dist_arrays))]
        for future in concurrent.futures.as_completed(futures):
            id_arrays.append(future.result()[0])
            data_arrays.append(future.result()[1])
    # for seq_idx, (neighs, u_supps, w_supps, d_supps, dists) in enumerate(zip(neigh_arrays, u_support_arrays, w_support_arrays, d_support_arrays, dist_arrays)):
    #     seq_supports = get_tax_supports_V(seq_idx, tax_tab[neighs], u_supps, w_supps, d_supps, dists)
    #     id_arrays.append(seq_supports[0])
    #     data_arrays.append(seq_supports[1])
    return np.concatenate(id_arrays), np.concatenate(data_arrays)

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