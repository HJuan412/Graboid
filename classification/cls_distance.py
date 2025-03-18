#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 14:32:39 2023

@author: hernan
Distance calculation functions
"""

#%% libraries
import numba as nb
import numpy as np
#%% fucntions
def combine(window):
    """Creates a dictionary for each site (column) in the window, grouping all
    the sequences (rows) sharing the same base. Reduces the amount of operations
    needed for distance calculation"""
    combined = []
    for col in window.T:
        col_vals = np.unique(col)
        col_combined = {val:np.argwhere(col==val).flatten() for val in col_vals}
        combined.append(col_combined)
    return combined

def get_distances(qry_window, ref_window, cost_mat):
    """Generates a distance matrix of shape (# qry seqs, # ref seqs)"""
    # combine query and reference sequences to (greatly) speed up calculation
    qry_combined = combine(qry_window)
    ref_combined = combine(ref_window)
    
    dist_array = np.zeros((qry_window.shape[0], ref_window.shape[0]))
    
    # calculate the distances for each site
    for site_q, site_r in zip(qry_combined, ref_combined):
        # sequences sharing values at each site are grouped, at most 5*5 operations are needed per site
        for val_q, idxs_q in site_q.items():
            for val_r, idxs_r in site_r.items():
                dist = cost_mat[val_q, val_r]
                # update distances
                for q in idxs_q: dist_array[q, idxs_r] += dist
    return dist_array

def get_distances2(query, reference, cost):
    distances = np.zeros((query.shape[0], reference.shape[0]))
    
    q_indexes = np.arange(query.shape[0])
    r_indexes = np.arange(reference.shape[0])
    for i in np.arange(query.shape[1]):
        query_val_indexes = [q_indexes[query_val] for query_val in query[:,i].T]
        ref_val_indexes = [r_indexes[ref_val] for ref_val in reference[:,i].T]
        
        for q_val, q_idxs in enumerate(query_val_indexes):
            new_distances = np.zeros(reference.shape[0])
            for r_val, r_idxs in enumerate(ref_val_indexes):
                new_distances[r_idxs] = cost[q_val, r_val]
            distances[q_idxs] += new_distances
    return distances

@nb.njit
def get_distances3(query, reference, cost):
    distances = np.zeros((query.shape[0], reference.shape[0]))
    
    q_indexes = np.arange(query.shape[0])
    r_indexes = np.arange(reference.shape[0])
    for i in np.arange(query.shape[1]):
        query_val_indexes = [q_indexes[query_val] for query_val in query[:,i].T]
        ref_val_indexes = [r_indexes[ref_val] for ref_val in reference[:,i].T]
        
        for q_val, q_idxs in enumerate(query_val_indexes):
            new_distances = np.zeros(reference.shape[0])
            for r_val, r_idxs in enumerate(ref_val_indexes):
                new_distances[r_idxs] = cost[q_val, r_val]
            distances[q_idxs] += new_distances
    return distances

import timeit
import concurrent.futures

def get_distances_multi_2(query, reference, cost, threads=1, workers=10):
    n_queries = np.arange(query.shape[0])
    chunks = np.array_split(n_queries, threads * workers)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(get_distances3, query[chk], reference, cost) for chk in chunks]
        distances = np.concatenate([future.result() for future in concurrent.futures.as_completed(futures)], axis=0)
    return distances

#timeit.timeit('get_distances_multi_2(query, reference, cost, threads=1)', number=10, globals=globals())