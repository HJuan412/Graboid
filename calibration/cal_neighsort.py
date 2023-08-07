#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:35:28 2023

@author: nano

Process calibration neighbours
"""

#%% libraries
import numpy as np

# Graboid libraries
from classification import cls_neighbours
#%% functions
# neighbour sorting and clustering
def get_sorted(distances):
    """Sort all paired distance arrays"""
    # distances: 3d array, shape: (n, #seqs, # seqs)
    # sort distances array , remove first column (distance to self, set as -1 so it is always the first)
    # get sorted indexes of distances array, for later use (also remove first column of each)
    # sorted_distances & sorted_indexes have shape (n, #seqs, #seqs-1)
    sorted_distances = np.sort(distances, 2)[:,:, 1:]
    sorted_indexes = np.argsort(distances, 2)[:,:, 1:]
    return sorted_distances, sorted_indexes

def compress(sorted_distances):
    """Compress each distance array into distance orbitals"""
    # sorted_distances is a 3d-array of shape (n, #seqs, #seqs-1)
    # returns a list of (n) lists of (#seqs) tuples of 3 arrays:
        # first array contains orbital distances
        # second array contains orbital start indexes
        # third array contains orbital counts (second array may be redundant)
    
    # compress neighbours into orbits
    compressed = []
    # compressed is a list containing, for each window, a list of the compressed distance orbitals for each n level
    for n_level in sorted_distances:
        compressed.append([np.unique(dist, return_index=True, return_counts = True) for dist in n_level]) # for each qry_sequence, get distance groups, as well as the index where each group begins and the count for each group
    return compressed

def sort_compress(distances):
    sorted_distances, sorted_indexes = get_sorted(distances)
    compressed = compress(sorted_distances)
    return sorted_distances, sorted_indexes, compressed

def build_packages(compressed, sorted_indexes, n_range, k_range, criterion):
    """For each parameter combination (window, n, k) get the k nearest elements for the corresponding compressed orbitals"""
    # returns a dictionary of keys (n, k) with values [(distances of the first k orbitals, start index of the first k orbitals, element counts of the first k orbitals), sorted_indexes for level n]
    classif_packages = {} # classif_packages contains all parameter combinations to be sent into the classifier
    # get the data for each parameter combination
    for n_idx, window_n in enumerate(compressed):
        n = n_range[n_idx]
        n_indexes = sorted_indexes[n_idx] # get the LAYER containing the sorted indexes for the current n value
        for k in k_range:
            # TODO: maybe get only the highest K and modify the classification function to avoid unnecesary calculations
            if criterion == 'orbit':
                classif_packages[(n, k)] = [cls_neighbours.get_knn_orbit_V(window_n, k), n_indexes]
            else:
                classif_packages[(n, k)] = [cls_neighbours.get_knn_neigh_V(window_n, k), n_indexes]
    return classif_packages