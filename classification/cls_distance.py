#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 14:32:39 2023

@author: hernan
Distance calculation functions
"""

#%% libraries
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
