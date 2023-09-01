#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:03:08 2023

@author: nano

Paired distance calculation for database calibration
"""

#%% libraries
import numpy as np
# Graboid libraries
from classification import cls_distance

#%% functions
def get_distances(window, window_sites, cost_mat):
    """Calcuate paired distances for every collapsed window, for every level of n"""
    # window is is a given calibration window
    # window_sites is the list of selected sites for each level of n ([[lvl_0 sites], [lvl_1 sites], ...])
    # returns win_distances, a 3d array of shape (# levels of n, # seqs in window, # seqs in window), diagonal elements are -1
    
    win_distances = []
    # get distances for each value of n, use cumsum to include the distance of all previous levels of n
    for n_sites in window_sites:
        win_cols = window.window[:, n_sites]
        win_distances.append(cls_distance.get_distances(win_cols, win_cols, cost_mat))
    win_distances = np.cumsum(win_distances, 0) # some elements in the diagonal have distance over 0 because of unknown sites
    win_distances[:, np.arange(win_distances.shape[1]), np.arange(win_distances.shape[2])] = -1 # diagonal elements to -1 ensures distance vs self is always first place when sorting
    return win_distances