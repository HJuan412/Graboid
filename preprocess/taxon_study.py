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
def build_tax_clusters(matrix, tax_tab, rank, dist_matrix = cost):
    # generates a dictionary of the form taxid:TaxCluster, also returns a list of the unique taxids
    clusters = {}
    taxes = tax_tab[f'{rank}_id'].dropna().reset_index(drop = True) # select appropiate column from the tax_tab
    uniques = taxes.unique()
    for uniq_tax in uniques: # build clusters
        tax_idxs = taxes.loc[taxes == uniq_tax].index
        sub_matrix = matrix[tax_idxs]
        clusters[uniq_tax] = TaxCluster(sub_matrix, uniq_tax, rank, dist_matrix)
    
    return clusters, uniques

def get_paired_dists(matrix, dist_matrix = cost):
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
class SuperCluster():
    # This class contains all the taxon clusters of selected data
    def __init__(self, matrix, tax_tab, rank, dist_matrix = cost):
        self.clusters, self.tax_list = build_tax_clusters(matrix, tax_tab, rank, cost)
        self.centroids = np.array([self.clusters[tax].centroid for tax in self.tax_list])
        self.centroid_dists = get_paired_dists(self.centroids, dist_matrix)
        self.get_collisions()
    
    def get_collisions(self):
        collisions = np.argwhere(self.centroid_dists == 0)
        collisions = collisions[np.argwhere(collisions[:,0]<collisions[:,1])].reshape((-1,2))
        self.collisions = [(self.tax_list[col[0]], self.tax_list[col[1]]) for col in collisions]
    
    def get_collapsed_data(self):
        # this method returns the matrix of unique sequences in each taxon
        tax_collapsed = []
        seq_collapsed = []
        
        for tax in self.tax_list:
            for seq in self.clusters[tax].collapsed:
                tax_collapsed.append(tax)
                seq_collapsed.append(seq)
        
        collapsed_data = np.array(seq_collapsed)
        # collapse after building the matrix (in case of repeated sequences between taxons), remove repeats
        post_collapsed_idx = windows.collapse_matrix(collapsed_data)
        seq_postcollapsed = []
        tax_postcollapsed = []
        
        for pci in post_collapsed_idx:
            if len(pci) == 1:
                idx = pci[0]
                seq_postcollapsed.append(seq_collapsed[idx])
                tax_postcollapsed.append(tax_collapsed[idx])
        return tax_postcollapsed, np.array(seq_postcollapsed)
        

class TaxCluster():
    # this class holds the sequences of a unique taxon
    # used to generate relevant data (paired distance, dispersion, centroids) and collapse unique sequences at the taxon level
    def __init__(self, matrix, taxid, rank, dist_matrix = cost):
        self.matrix = matrix
        self.paired = get_paired_dists(matrix, dist_matrix)
        self.get_collapsed()
        self.taxid = taxid
        self.rank = rank
        self.nseqs = matrix.shape[0]
        self.bases = matrix.shape[1]
        self.get_params()
    
    def get_collapsed(self):
        uniq_idxs = windows.collapse_matrix(self.matrix)
        collapsed_idxs = [u[0] for u in uniq_idxs]
        self.collapsed = self.matrix[collapsed_idxs]
        self.nseqs_collapsed = self.collapsed.shape[0]
    
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
    