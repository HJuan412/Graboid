#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 14:37:09 2023

@author: hernan
Get nearest neighbours
"""

#%% libraries
import numpy as np

#%% functions
# k_nearest is a list containing information about a (collapsed) query sequence's neighbours
# each element is a 3 element tuple containing:
    # an array with the distances to the k nearest orbitals (or up to the orbital containing the kth neighbour)
    # a 2d array containing the start and end positions of each included orbital, shape is (# orbitals, 2)
    # an array with the number of neighbours contained in each orbital

# _V functions produce results as 3 2d-numpy arrays: k_dists (shape: # seqs, k), k_positions (shape: # seqs, k+1) and k_counts (shape: # seqs, k)
# k_dists produced by knn_neigh has default values of -1 to differentiate empty orbitals from tose with distance 0 (occupied orbitals for this function is <= k)
def get_knn_orbit_V(compressed, k):
    """Get the neighbours from the k nearest orbitals"""
    n_seqs = len(compressed)
    k_dists = np.zeros((n_seqs, k), dtype=np.float16)
    k_positions = np.zeros((n_seqs, k+1), dtype=np.int16)
    k_counts = np.zeros((n_seqs, k), dtype=np.int16)
    for seq_idx, (dists, idxs, counts) in enumerate(compressed):
        k_dists[seq_idx] = dists[:k]
        k_positions[seq_idx] = idxs[:k+1]
        k_counts[seq_idx] = counts[:k]
    return k_dists, k_positions, k_counts

def get_knn_neigh_V(compressed, k):
    """Get the neighbours from the orbitals up to the one containing the k-th neighbour"""
    n_seqs = len(compressed)
    k_dists = np.full((n_seqs, k), -1, dtype=np.float16)
    k_positions = np.full((n_seqs, k+1), -1, dtype=np.int16)
    k_counts = np.zeros((n_seqs, k), dtype=np.int16)
    for seq_idx, (dists, idxs, counts) in enumerate(compressed):
        summed = np.cumsum(counts)
        break_orb = np.argmax(summed >= k) + 1 # get orbital containing the k-th nearest neighbour
        k_dists[seq_idx, :break_orb] = dists[:break_orb]
        k_positions[seq_idx, :break_orb+1] = idxs[:break_orb+1]
        k_counts[seq_idx, :break_orb] = counts[:break_orb]
    return k_dists, k_positions, k_counts
    
def get_knn_orbit(compressed, k):
    """Get the neighbours from the k nearest orbitals"""
    k_nearest = []
    for dists, idxs, counts in compressed:
        k_dists = dists[:k]
        k_positions = np.array([idxs[:k], idxs[1:k+1]]).T
        k_counts = counts[:k]
        k_nearest.append((k_dists, k_positions, k_counts))
    return k_nearest

def get_knn_neigh(compressed, k):
    """Get the neighbours from the orbitals up to the one containing the k-th neighbour"""
    k_nearest = []
    for dists, idxs, counts in compressed:
        summed = np.cumsum(counts)
        break_orb = np.argmax(summed >= k) + 1 # get orbital containing the k-th nearest neighbour
        k_dists = dists[:break_orb]
        k_positions = np.array([idxs[:break_orb], idxs[1:break_orb+1]]).T
        k_counts = counts[:break_orb]
        k_nearest.append((k_dists, k_positions, k_counts))
    return k_nearest