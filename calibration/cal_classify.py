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
def compress(sorted_distances):
    """Compress each distance array into distance orbitals"""
    # sorted_distances is a 2d-array of shape (#seqs, #seqs-1)
    # returns a list of (#seqs) arrays of shape (3, #orbitals):
        # first row contains orbital distances
        # second row contains orbital start indexes
        # third row contains orbital counts (second array may be redundant)
    
    # compress neighbours into orbits
    compressed = []
    # compressed is a list containing, for each window, a list of the compressed distance orbitals for each n level
    for dist in sorted_distances:
        compressed.append(np.array(np.unique(dist, return_index = True, return_counts = True))) # for each qry_sequence, get distance groups, as well as the index where each group begins and the count for each group
    return compressed

def get_k_neighs(orbitals, k, criterion='orbit'):
    """Extract the k nearest orbitals for each one in a group of orbitals"""
    # orbitals is a list of #seqs 2d-arrays of shape (3, #orbitals)
    # k is the max number of orbitals/neighbours 
    # criterion is either orbit or neigh
    # returns 3 2d-arrays of shape (#seqs, k (if criterion is 'orbit')/#largest break orb (if criterion is 'neigh'))
        # distances: contains unique sorted neighbour distances for each sequence
        # positions: contains the starting position for each orbital for each sequence
        # counts: contains the number of neighbours in each orbital for each sequence
    
    n_orbitals = len(orbitals)
    if criterion == 'orbit':
        # prepare output arrays
        distances = np.zeros((n_orbitals, k))
        positions = np.zeros((n_orbitals, k), dtype=int)
        counts = np.zeros((n_orbitals, k), dtype=int)
        # populate arrays
        for idx, orbs in enumerate(orbitals):
            nneighs = min(orbs.shape[1], k) # on some cases, the number of available orbitals may be lower than k
            distances[idx, :nneighs] = orbs[0, :k]
            positions[idx, :nneighs] = orbs[1, :k]
            counts[idx, :nneighs] = orbs[2, :k]
    
    elif criterion == 'neigh':
        # establish highest orbital
        break_orbs = []
        for orbs in orbitals:
            cumm_neighs = np.cumsum(orbs[2])
            break_orbs.append(np.argmax(cumm_neighs >= k) + 1)        
        n_cols = np.max(break_orbs)
        # prepare output arrays
        distances = np.zeros((n_orbitals, n_cols))
        positions = np.zeros((n_orbitals, n_cols), dtype=int)
        counts = np.zeros((n_orbitals, n_cols), dtype=int)
        # populate arrays
        for idx, orbs, break_orb in enumerate(zip(orbitals, break_orbs)):
            distances[idx, :break_orb] = orbs[0, :break_orb]
            positions[idx, :break_orb] = orbs[1, :break_orb]
            counts[idx, :break_orb] = orbs[2, :break_orb]
    return distances, positions, counts

def extend(values, counts):
    """Extend values array, repeating each element the number of times specidied in the corresponding element in the counts array"""
    extended = []
    for vals, cnts in zip(values, counts.astype(int)):
        extended.append(np.concatenate([[val] * cnt for val, cnt in zip(vals, cnts)]))
    return extended

def get_tax_supports(distances, counts, u_supports, w_supports, d_supports, sorted_neigh_taxa, seq_indexes, rk_idx=0):
    """Calculate tax support for every sequence in the current RANK"""
    # k nearest neighbour (orbital or neigh) have already been selected, this step only calulates candidates support
    # distances array contains distances of each ORBITALS
    # counts array contains # neighbours in each ORBITAL
    # *_supports arrays contain the supports for each ORBITAL
    # sorted_neigh_taxa contains the taxa of the sorted neighbours of every valid sequence in the current RANK
    # seq_indexes contains the indexes of the VALID sequences in the current RANK
    # rk_idx: current RANK index
    # returns 2d array of columns: idx, rank_id, tax_id, total_neighs, mean_distance, std_distance, total_unweighted_support, softmax_u_support, total_wknn_support, softmax_wknn_support, total_dwknn_support, softmax_d_support
        
    tax_supports = []
    
    # expand distance and support vectors
    dists_ext = extend(distances, counts)
    u_supp_ext = extend(u_supports, counts)
    w_supp_ext = extend(w_supports, counts)
    d_supp_ext = extend(d_supports, counts)
    max_poss = counts.sum(1)
    
    # get supports for each taxon for each sequence
    # sort taxa by index, transfer order to distance and support arrays
    # taxa is a list, not an array, since the number of neighbouring taxa varies between sequences
    taxa = [srt_neigh_taxa[:mx_pos] for srt_neigh_taxa, mx_pos in zip(sorted_neigh_taxa, max_poss)]
    sorted_taxa_idx = [np.argsort(tx) for tx in taxa]
    sorted_taxa = [np.sort(tx) for tx in taxa]
    
    for idx, (srt_tax, srt_tax_idx) in enumerate(zip(sorted_taxa, sorted_taxa_idx)):
        # group support and distances by taxon
        sorted_dist = dists_ext[idx][srt_tax_idx]
        sorted_u_supp = u_supp_ext[idx][srt_tax_idx]
        sorted_w_supp = w_supp_ext[idx][srt_tax_idx]
        sorted_d_supp = d_supp_ext[idx][srt_tax_idx]
        
        uniq_tax, tax_pos, tax_count = np.unique(srt_tax, return_index=True, return_counts=True)
    
        n_tax = len(uniq_tax)
        # prepare output table for the current sequence. Will contain support values of every neighbouring taxa
        seq_data = np.zeros((n_tax, 12), dtype=np.float32) # columns : seq_idx rank_id tax_id total_neighs mean_distance std_distance total_unweighted_support softmax_unweighted_support total_wknn_support softmax_wknn_support total_dwknn_support softmax_dwknn_support
        seq_data[:, 0] = seq_indexes[idx]
        seq_data[:, 1] = rk_idx
        seq_data[:, 2] = uniq_tax
        
        # calculate total support for each neighbouring taxa
        for tax_idx, (tax, tax_start, tx_count) in enumerate(zip(uniq_tax, tax_pos, tax_count)):
            tax_end = tax_start + tx_count
            seq_data[tax_idx, 3] = tx_count # total neighbours
            seq_data[tax_idx, 4] = sorted_dist[tax_start: tax_end].mean() # mean distance
            seq_data[tax_idx, 5] = sorted_dist[tax_start: tax_end].std() # std distance
            seq_data[tax_idx, 6] = sorted_u_supp[tax_start: tax_end].sum() # total unweighted support
            seq_data[tax_idx, 8] = sorted_w_supp[tax_start: tax_end].sum() # total wknn support
            seq_data[tax_idx, 10] = sorted_d_supp[tax_start: tax_end].sum() # total dwknn support
    
        # remove no support taxa
        seq_data = seq_data[seq_data[:, 3] > 0]
    
        # softmax normalize support
        seq_data[:, 7] = cls_classify.softmax(seq_data[:, 6])
        seq_data[:, 9] = cls_classify.softmax(seq_data[:, 8])
        seq_data[:, 11] = cls_classify.softmax(seq_data[:, 10])
    
        # build support tab for current rank
        tax_supports.append(seq_data)
        
    return np.concatenate(tax_supports)

def groupby_np(array, col):
    # array must be sorted by column
    values, starts, counts = np.unique(array[:,col], return_index=True, return_counts=True)
    ends = starts + counts
    for val, start, end in zip(values, starts, ends):
        yield val.astype(int), array[start:end]
        
def get_pred_taxa(k_tab, win_tax):
    """Determine the predicted taxa for a given window/n/k combination, for all scoring methods"""
    # k_tab is a table with columns seq_idx rank_id tax_id total_neighs mean_distance std_distance total_unweighted_support softmax_unweighted_support total_wknn_support softmax_wknn_support total_dwknn_support softmax_dwknn_support
    # win_tax is only used to determine the shape of the output tables
    
    # prepare prediction tables (default value -1)
    pred_u = np.full(win_tax.shape, -1)
    pred_w = np.full(win_tax.shape, -1)
    pred_d = np.full(win_tax.shape, -1)
    
    # get predicted taxon for each sequence for each rank (rank column: 1, seq column: 0), k_tab is ordered by rank first, then by sequence
    for rk_idx, rk_tab in groupby_np(k_tab, 1):
        for seq_idx, seq_tab in groupby_np(rk_tab, 0):
            # single classification candidate
            if seq_tab.shape[0] == 1:
                pred_u[seq_idx, rk_idx] = seq_tab[0,2]
                pred_w[seq_idx, rk_idx] = seq_tab[0,2]
                pred_d[seq_idx, rk_idx] = seq_tab[0,2]
                continue
            
            # multiple candidates, get the most supported one for each scoring method
            u_supp_sort = seq_tab[np.argsort(seq_tab[:, 6])][:,[2, 6]]
            w_supp_sort = seq_tab[np.argsort(seq_tab[:, 8])][:,[2, 8]]
            d_supp_sort = seq_tab[np.argsort(seq_tab[:, 10])][:,[2,10]]
    
            # only classify if there are no ties (greatest support > second greatest support)
            if u_supp_sort[-1, 1] != u_supp_sort[-2, 1]:
                pred_u[seq_idx, rk_idx] = u_supp_sort[-1, 0]
            if w_supp_sort[-1, 1] != w_supp_sort[-2, 1]:
                pred_w[seq_idx, rk_idx] = w_supp_sort[-1, 0]
            if d_supp_sort[-1, 1] != d_supp_sort[-2, 1]:
                pred_d[seq_idx, rk_idx] = d_supp_sort[-1, 0]
    return pred_u, pred_w, pred_d

def get_real_support(k_tab, win_tax):
    """Determine the support for each sequence's REAL taxon in a given window/n/k combination, for all scoring methods"""
    # k_tab is a table with columns seq_idx rank_id tax_id total_neighs mean_distance std_distance total_unweighted_support softmax_unweighted_support total_wknn_support softmax_wknn_support total_dwknn_support softmax_dwknn_support
    # win_tax is only used to determine the shape of the output tables
    
    # prepare support tables (default value -1, different from support 0)
    real_u = np.full(win_tax.shape, -1.)
    real_w = np.full(win_tax.shape, -1.)
    real_d = np.full(win_tax.shape, -1.)
    
    # get support for real taxa for each sequence for each rank (rank column: 1, seq column: 0), k_tab is ordered by rank first, then by sequence
    for rk_idx, rk_tab in groupby_np(k_tab, 1):
        for seq_idx, seq_tab in groupby_np(rk_tab, 0):
            # get real taxon of sequence
            real_tax = win_tax[seq_idx, rk_idx]
            if not real_tax in seq_tab[:,2]:
                # sequence has no support for the real taxon
                # unknown taxon index (-1) will not appear among predicted taxa candidates
                continue
            
            # get row for the real taxon, update support values
            real_row = seq_tab[seq_tab[:,2] == real_tax][0]
            real_u[seq_idx, rk_idx] = real_row[7]
            real_w[seq_idx, rk_idx] = real_row[9]
            real_d[seq_idx, rk_idx] = real_row[11]
    return real_u, real_w, real_d

def classify(distances, window_tax, n, k_range, out_dir, win_idx=0, criterion='orbit'):
    """Generates the classification array of a window for a given value of n and a range of values of k"""
    # distances: 2d-array of shape (#seqs, #seqs), paired distances (including self distances) calculated using n sites
    # window_tax: 2d-array of shape (#seqs, #ranks), unknown taxa are marked as nan
    # n: n sites used in distance calculations
    # k_range: range of orbitals to use in classificaiton
    # out_dir: destination folder for result files
    # win_idx: identifier of the current window
    # criterion: 'orbit' or 'neigh'
    
    known_taxa = ~np.isnan(window_tax) # classify only using sequences for which the taxon is known
    n_seqs = window_tax.shape[0]
    seq_idxs = np.arange(n_seqs)
    
    # generate a classification for each value of k
    k_tables = [[] for k in k_range]
    # known taxa vary between taxonomic levels. Data processed sequentially by rank
    for rk_idx, rk_taxa in enumerate(window_tax.T):
        # get valid (known taxa) sequences
        valid = known_taxa.T[rk_idx]
        valid_idxs = seq_idxs[valid]
        valid_taxa = rk_taxa[valid]
        valid_dists = distances[valid][:, valid]
        # sort distances (keep orders), remove distance to self (always first one because takes value -1)
        sorted_dists = np.sort(valid_dists)[:,1:] # shape (#valid_seqs, #valid_seqs-1)
        sorted_dists_idxs = np.argsort(valid_dists)[:,1:] # shape (#valid_seqs, #valid_seqs-1)
        sorted_neigh_taxa = valid_taxa[sorted_dists_idxs] # shape (#valid_seqs, #valid_seqs-1), contains the taxa of the sorted neighbours of every valid sequence in the current RANK
        
        # collapse neighbour distances for each VALID sequence
        orbitals = compress(sorted_dists)
        
        # select the k nearest neighbours (according to criterion) and calculate orbital supports
        for k_idx, k in enumerate(k_range):
            k_distances, k_positions, k_counts = get_k_neighs(orbitals, k, criterion)
            
            u_supports = cls_classify.unweighted(k_distances)
            w_supports = cls_classify.wknn(k_distances)
            d_supports = cls_classify.dwknn(k_distances)
            
            k_tab = get_tax_supports(k_distances, k_counts, u_supports, w_supports, d_supports, sorted_neigh_taxa, valid_idxs, rk_idx)
            k_tab = k_tab[np.argsort(k_tab[:,0])] # sort by sequence index
            k_tables[k_idx].append(k_tab)
    
    # every table is ordered by rank and sequence index
    k_tables = [np.concatenate(k_tab) for k_tab in k_tables]
    
    # determine predictions and real tax support, save temporal files
    for k, k_tab in zip(k_range, k_tables):
        pred_u, pred_w, pred_d = get_pred_taxa(k_tab, window_tax)
        real_u, real_w, real_d = get_real_support(k_tab, window_tax)
        
        np.savez(out_dir + f'/{win_idx}_{n}_{k}.npz',
                  predicted_u = pred_u,
                  predicted_w = pred_w,
                  predicted_d = pred_d,
                  real_u_support = real_u,
                  real_w_support = real_w,
                  real_d_support = real_d,
                  params = np.array([win_idx, n, k]))
    return k_tables
#%% OLD functions
def get_tax_supports_V(taxa, u_supports, w_supports, d_supports, distances, counts, idx=0):
    # taxa is a numpy array with the extended taxonomy of the sequence neighbours
    # *_supports arrays contain the supports for each ORBITAL
    # distances array contains distances of each ORBITAL
    # counts array contains # neighbours in each ORBITAL
    # idx identifies the current sequence
    # returns 2d array of columns: idx, rank_id, tax_id, total_neighs, mean_distance, std_distance, total_unweighted_support, softmax_u_support, total_wknn_support, softmax_wknn_support, total_dwknn_support, softmax_d_support
    
    taxa = np.nan_to_num(taxa, nan=-1)
    supports_tab =[]
    
    # expand distance and support vectors
    dist_ext = np.concatenate([[dist] * count for dist, count in zip(distances, counts.astype(int))])
    u_supp_ext = np.concatenate([[u_supp] * count for u_supp, count in zip(u_supports, counts.astype(int))])
    w_supp_ext = np.concatenate([[w_supp] * count for w_supp, count in zip(w_supports, counts.astype(int))])
    d_supp_ext = np.concatenate([[d_supp] * count for d_supp, count in zip(d_supports, counts.astype(int))])
    
    # get supports for each taxon for each rank
    for rk_idx, rk in enumerate(taxa.T):
        # sort taxa by index, transfer order to distance and support arrays
        sorted_rk_idx = np.argsort(rk)
        sorted_rk = rk[sorted_rk_idx]
        sorted_dist = dist_ext[sorted_rk_idx]
        sorted_u_supp = u_supp_ext[sorted_rk_idx]
        sorted_w_supp = w_supp_ext[sorted_rk_idx]
        sorted_d_supp = d_supp_ext[sorted_rk_idx]
        uniq_tax, tax_pos, tax_count = np.unique(sorted_rk, return_index=True, return_counts=True)
        
        n_tax = len(uniq_tax)
        rk_data = np.zeros((n_tax, 9), dtype=np.float32) # columns : total_neighs, mean_distance, std_distance, total_unweighted_support, softmax_u_support, total_wknn_support, softmax_wknn_support, total_dwknn_support, softmax_d_support
        
        for tax_idx, (tax, pos, count) in enumerate(zip(uniq_tax, tax_pos, tax_count)):
            end_pos = pos + count
            rk_data[tax_idx, 0] = count # total neighbours
            rk_data[tax_idx, 1] = sorted_dist[pos: end_pos].mean() # mean distance
            rk_data[tax_idx, 2] = sorted_dist[pos: end_pos].std() # std distance
            rk_data[tax_idx, 3] = sorted_u_supp[pos: end_pos].sum() # total unweighted support
            rk_data[tax_idx, 5] = sorted_w_supp[pos: end_pos].sum() # total wknn support
            rk_data[tax_idx, 7] = sorted_d_supp[pos: end_pos].sum() # total dwknn support
        
        # remove no support taxa
        supported = rk_data[:, 3] > 0
        rk_data = rk_data[supported]
        rk_tax = uniq_tax[supported]
        
        # softmax normalize support
        rk_data[:, 4] = cls_classify.softmax(rk_data[:, 3])
        rk_data[:, 6] = cls_classify.softmax(rk_data[:, 5])
        rk_data[:, 8] = cls_classify.softmax(rk_data[:, 7])
        
        # add squence and rank idxs
        rk_tax = np.array([np.full(len(rk_tax), idx), np.full(len(rk_tax), rk_idx), rk_tax]).T
        
        # build support tab for current rank
        supports_tab.append(np.concatenate((rk_tax, rk_data), axis = 1))
    
    return np.concatenate(supports_tab)

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

def get_supports(supports_tab, tax_tab):
    """Get best support and support for the real taxon for every classified sequence for every rank"""
    # id_array contains columns seq_idx, rk_id, tax_id
    # data_array contains columns total_neighbours, mean_distance, std_distance, unweighted_support, softmax_u_support, wknn_support, softmax_w_support, dwknn_support, softmax_d_support
    # tax_tab is an array containing the complete taxon ids for the query sequences
    # for each support function, return two arrays of shape (seq_idx, #ranks)
        # first array contains the predicted tax_id for each sequence per rank
        # second array contains the supports for the real taxon for each rank in each sequence
        
    predicted_u = np.full(tax_tab.shape, -1, dtype=int)
    real_u_support = np.full(tax_tab.shape, -1, dtype=np.float32)
    
    predicted_w = np.full(tax_tab.shape, -1, dtype=int)
    real_w_support = np.full(tax_tab.shape, -1, dtype=np.float32)
    
    predicted_d = np.full(tax_tab.shape, -1, dtype=int)
    real_d_support = np.full(tax_tab.shape, -1, dtype=np.float32)
    
    for rk_idx, rk_taxa in enumerate(tax_tab.T):
        # get supports for the current RANK
        rk_supports = supports_tab[supports_tab[:,1] == rk_idx]
        
        # get predicted taxa
        
        # get best support for each training instance EXCLUDING support for unknown taxa (-1)
        rk_supports_known = rk_supports[rk_supports[:,2] > 0]
        
        for pred_table, method_col in zip((predicted_u, predicted_w, predicted_d), (6,8,10)):
            # sort support table (descending) for all predictions for the current rank, sort again by sequence id
            support_sorted = rk_supports_known[np.argsort(rk_supports_known[:, method_col])[::-1]]
            support_sorted = support_sorted[np.argsort(support_sorted[:, 0])]
            
            # count candidate classifications for each sequence
            seq_idxs, seq_poss, seq_cnts = np.unique(support_sorted[:,0], return_index=True, return_counts=True)
            
            # get seqs with single candidates
            single_pred = seq_cnts == 1
            # check for ties (keep only sequences whit a clear winner prediction)
            seq_firsts = seq_poss[~single_pred]
            seq_seconds = seq_firsts + 1
            winners = support_sorted[seq_firsts, method_col] > support_sorted[seq_seconds, method_col]
            best_locs = np.concatenate([seq_firsts[winners], seq_poss[single_pred]]) # merge winner predictions and single predictions
            
            # extract sequence index + predicted taxon
            predictions = rk_supports_known[best_locs][:, [0,2]]
            
            # update predicted_tables
            pred_table[predictions[:,0].astype(int), rk_idx] = predictions[:,1]
        
        # get support of real taxa
        # get unique taxa
        unique_taxa = np.unique(rk_taxa[~np.isnan(rk_taxa)])
        # get indexes of represented sequences
        supp_indexes = np.arange(tax_tab.shape[0])
        
        # get true support per taxon
        for tax in unique_taxa:
            # locate instances of taxon
            tax_instances = supp_indexes[rk_taxa == tax]
            # extract calculated supports for taxon, then filter by instances belonging to taxon
            tax_supports = rk_supports[rk_supports[:,2] == tax]
            tax_supports = tax_supports[np.isin(tax_supports[:,0], tax_instances)]
            
            for supp_table, method_col in zip((real_u_support, real_w_support, real_d_support), (7,9,11)):
                # set default support as 0 (differenctiate from unclear taxa in db, with score of -1)
                supp_table[tax_instances] = 0
                
                # update supports with normalized scores
                supp_table[tax_supports[:,0].astype(int), rk_idx] = tax_supports[:, method_col]
    return predicted_u, real_u_support, predicted_w, real_w_support, predicted_d, real_d_support
