#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:35:28 2023

@author: nano

Sort neighbours by distances
"""

#%% libraries
import concurrent.futures
import numpy as np

#%% functions
def get_sorted_neighs(sorted_idxs, sorted_dists, idx=''):
    # separate the neighbours for each individual element in the paired distance matrix
    # sorted_idxs contains the indexes array generated by get_all_distances, ordered by the distances calculated for a given level
    # returns arrays neighs (containing neighbour indexes) and dists (containing neighbour distances to the element)
    neighs = []
    dists = []
    # isolate the neighbours of each element and their respective distances
    for seq in np.arange(sorted_idxs.max() + 1):
        # get all pairs that include the current element
        seq_idxs = sorted_idxs == seq
        combined = seq_idxs[0] | seq_idxs[1] 
        seq_neighs = sorted_idxs[:, combined]
        # extract the ordered distances for the current element's neighbours
        dists.append(sorted_dists[combined])
        # extract the current element's neighbours' indexes
        ordered_neighs = np.full(seq_neighs.shape[1], -1, dtype=np.int16)
        ordered_neighs[seq_neighs[0] != seq] = seq_neighs[0][seq_neighs[0] != seq]
        ordered_neighs[seq_neighs[1] != seq] = seq_neighs[1][seq_neighs[1] != seq]
        neighs.append(ordered_neighs)
    print(f'Sorted neighbours for window {idx}^')
    return np.array(neighs), np.array(dists)

# sort the generated distance arrays
def sort_neighbours(win_list, win_dists, threads=1, win_idxs=None):
    # win_idxs is used to signal when calculations are finished for a given window
    if win_idxs is None:
        win_idxs = np.arange(len(win_list))
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        # for each window, we are getting the ascending ORDER for the distance calculated at each level
        sorted_dists = [dists for dists in executor.map(np.argsort, [win_d[0] for win_d in win_dists], [1]*len(win_list))] # win_d[0] contains the distances array, win_d[1] contains the paired indexes (used later)
    
    sorted_win_neighbours = []
    # sorted_win_neighbours is structured as:
        # window 0:
            # level 0:
                # sorted neighbour idxs
                # sorted neighbour dists
                # both these arrays have shape n_rows * n_rows - 1, as each row has n_rows - 1 neighbours                    
    for idx, (win_dists, sorted_win_dists) in enumerate(zip(win_dists, sorted_dists)):
        sorted_idxs = [win_dists[1][:, lvl] for lvl in sorted_win_dists]
        sorted_distances = [dsts[lvl] for dsts, lvl in zip(win_dists[0], sorted_win_dists)]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
            sorted_win_neighbours.append([ordered for ordered in executor.map(get_sorted_neighs, sorted_idxs, sorted_distances, win_idxs)])
    
    return sorted_win_neighbours