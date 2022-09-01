#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 16:00:03 2022

@author: hernan
Study the internal variation of a taxonomic cluster of sequences
"""

#%% libraries
from classif import classification
import numpy as np

#%% functions
def build_tax_clusters(matrix, tax_array, dist_mat):
    # generates a dictionary of the form taxid:TaxCluster, also returns a list of the unique taxids
    clusters = {}
    # build clusters
    for tax in np.unique(tax_array):
        sub_matrix = matrix[tax_array == tax]
        clusters[tax] = TaxCluster(sub_matrix, dist_mat)
    
    return clusters

def get_paired_dists(matrix, dist_mat):
    # constructs a paired distance matrix of the given matrix using a distance matrix
    # distance data is located on the top right of the matrix
    nseqs = matrix.shape[0]
    paired = np.zeros((nseqs, nseqs))
    for idx0, seq0 in enumerate(matrix):
        for idx1 in range(idx0 + 1, matrix.shape[0]):
            seq1 = matrix[idx1]
            paired[idx0, idx1] = classification.calc_distance(seq0, seq1, dist_mat)
    return paired

def get_flat_dists(matrix):
    # flattens the top right half of a distance matrix
    distances = []
    nseqs = matrix.shape[0]
    for idx in range(nseqs):
        distances += list(matrix[idx, idx+1:])
    return np.array(distances)

def get_rowcol(mat, idx):
    # get the column and row values for the given index of a distance matrix
    col = mat[:idx,idx]
    row = mat[idx, idx+1:]
    return np.concatenate((col, row))

#%% classes
class clust_iterator:
    def __init__(self, cluster):
        self._cluster = cluster
        self._tax_list = cluster.tax_list
        self._index = 0
    
    def __next__(self):
        if self._index < len(self._tax_list):
            tax = self._tax_list[self._index]
            result = self._cluster[tax]
            self._index += 1
            return result
        raise StopIteration

class SuperCluster:
    # This class contains all the taxon clusters of selected data
    def __init__(self, matrix, tax_tab, dist_mat):
        tax_mat = tax_tab.to_numpy().T
        self.clusters = {}
        for rk_idx, tax_array in enumerate(tax_mat):
            self.clusters[rk_idx] = build_tax_clusters(matrix, tax_array, dist_mat)
        # self.centroids = {rank:{tax:cluster.centroid for tax, cluster in clusters.items()} for rank, clusters in self.clusters.items()}
        # self.centroid_dists = get_paired_dists(self.centroids, dist_mat)
        # self.get_collisions()
    
    def __getitem__(self, item):
        rk, tax = item
        return self.clusters[item]

class TaxCluster:
    # this class holds the sequences of a unique taxon
    # used to generate relevant data (paired distance, dispersion, centroids) and collapse unique sequences at the taxon level
    def __init__(self, matrix, dist_mat):
        paired = get_paired_dists(matrix, dist_mat)
        self.nseqs = matrix.shape[0]
        if self.nseqs == 1:
            self.mean = 0
            self.std = 0
            self.max_range = 0
            self.means = [0]
            self.centroid = matrix[0]
        else:
            flat_distances = get_flat_dists(paired)
            self.mean = flat_distances.mean()
            self.std = flat_distances.std()
            self.max_range = flat_distances.max()
            # get mean distance from every member of the cluster to everyone else
            # set centroid as the member with the LOWEST mean distance
            means = []
            for idx in np.arange(self.nseqs):
                dists = get_rowcol(paired, idx)
                means.append(dists.mean())
            self.means = means
            self.centroid = matrix[np.argmin(means)]
            self.dists_to_centroid = get_rowcol(paired, np.argmin(means))