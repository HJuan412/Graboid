#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 16:00:03 2022

@author: hernan
Study the internal variation of a taxonomic cluster of sequences
"""

#%%
import sys
sys.path.append('classif')
sys.path.append('preprocess')
#%% libraries
import cost_matrix
import distance
import numpy as np
import windows
#%% vars
cost = cost_matrix.cost_matrix()
#%% functions
def build_tax_clusters(matrix, tax_tab, rank, dist_matrix=cost):
    # generates a dictionary of the form taxid:TaxCluster, also returns a list of the unique taxids
    clusters = {}
    taxes = tax_tab[rank].to_numpy()
    # build clusters
    for tax in np.unique(taxes):
        sub_matrix = matrix[taxes == tax]
        clusters[tax] = TaxCluster(sub_matrix, tax, rank, dist_matrix)
    
    return clusters

def get_paired_dists(matrix, dist_matrix=cost):
    # constructs a paired distance matrix of the given matrix using a distance matrix
    # distance data is located on the top right of the matrix
    nseqs = matrix.shape[0]
    paired = np.zeros((nseqs, nseqs))
    for idx0, seq0 in enumerate(matrix):
        for idx1 in range(idx0 + 1, matrix.shape[0]):
            seq1 = matrix[idx1]
            paired[idx0, idx1] = distance.calc_distance(seq0, seq1, dist_matrix)
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
    col = mat[:idx + 1, idx + 1].flatten()
    row = mat[idx, idx+2:].flatten()
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
    def __init__(self, matrix, tax_tab, rank, dist_matrix=cost):
        self.clusters = build_tax_clusters(matrix, tax_tab, rank, cost)
        self.tax_list = list(self.clusters)
        self.centroids = {tax:cluster.centroid for tax, cluster in self.clusters.items()}
        self.centroid_dists = get_paired_dists(self.centroids, dist_matrix)
        self.get_collisions()
    
    def __getitem__(self, item):
        return self.clusters[item]
    
    def __iter__(self):
        return clust_iterator(self)

class TaxCluster:
    # this class holds the sequences of a unique taxon
    # used to generate relevant data (paired distance, dispersion, centroids) and collapse unique sequences at the taxon level
    def __init__(self, matrix, taxid, rank='genus', dist_matrix=cost):
        self.matrix = matrix
        self.paired = get_paired_dists(matrix, dist_matrix)
        self.taxid = taxid
        self.rank = rank
        self.nseqs = matrix.shape[0]
        self.get_params()
    
    def get_mean_dists(self):
        means = []
        for idx in range(self.nseqs -1):
            dists = get_rowcol(self.paired, idx)
            means.append(dists.mean())
        self.means = means
        self.centroid = self.matrix[np.argmin(means)]

    def get_params(self):
        if self.nseqs == 1:
            self.mean = 0
            self.std = 0
            self.max_range = 0
            self.means = [0]
            self.centroid = self.matrix[0]
        else:
            flat_distances = get_flat_dists(self.paired)
            self.mean = flat_distances.mean()
            self.std = flat_distances.std()
            self.max_range = flat_distances.max()
            self.get_mean_dists()
    