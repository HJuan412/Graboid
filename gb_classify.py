#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 13:44:10 2021

@author: hernan
"""

#%% libraries
from abc import ABC, abstractmethod
from gb_cost_matrix import cost_matrix
import gb_preprocess as pp
import numpy as np
import pandas as pd

#%% dist calculators
class DistCalculator(ABC):
    def __init__(self, data, query, cost_matrix = None):
        self.data = data
        self.query = query
        self.cost_matrix = cost_matrix
        self.dists = np.zeros((self.query.shape[0], self.data.shape[0]))

    @abstractmethod
    def get_dist(self):
        pass

class DistByID(DistCalculator):
    def get_dist(self):
        
        for idx, q in enumerate(self.query):
            q_dist = (1-(q == self.data)).sum(axis = 1)
            self.dists[idx,:] = q_dist

class DistByCost(DistCalculator):
    def get_dist(self):
        for idx0, query in enumerate(self.query):
            for idx1, ref in enumerate(self.data):
                q_cost = 0
                for q, s in zip(query, ref):
                    q_cost += self.cost_matrix[q, s]
                self.dists[idx0, idx1] = q_cost

#%% voters
class Voter(ABC):
    def __init__(self, K, distances, data_classes):
        self.K = K
        self.distances = distances
        self.data_classes = data_classes
        self.classif = False

    @abstractmethod
    def classify(self):
        pass

class MajorityVote(Voter):
    def classify(self):
        self.classif = pd.DataFrame(columns = ['Code', 'NN'])
        for idx, q_dist in enumerate(self.distances):
            sorted_dists = np.argsort(q_dist)
            k_idx = sorted_dists[:self.K]
            k_taxes = self.data_classes[k_idx]
    
            taxes, counts = np.unique(k_taxes, return_counts = True)
            most_abundant = np.argsort(counts)[-1]
            self.classif.at[idx,:] = [taxes[most_abundant], counts[most_abundant]]

class WeightedVote(Voter):
    def classify(self):
        self.classif = pd.DataFrame(columns = ['Query', 'Code', 'Weight'])
        for q_idx, q_dist in enumerate(self.distances):
            sorted_dists = np.argsort(q_dist)
            k_idx = sorted_dists[:self.K]
            k_taxes = self.data_classes[k_idx]
            w_dict = {tax:0 for tax in k_taxes}
            
            for idx in k_idx:
                tax = self.data_classes[idx]
                weight = q_dist[idx]
                w_dict[tax] += weight
            
            for k,v in w_dict.items():
                self.classif = self.classif.append(pd.Series({'Query':q_idx, 'Code':k, 'Weight':v}), ignore_index = True)

#%% testing
import window_collapser

test_file = 'Dataset/12_11_2021-23_15_53/Matrices/Nematoda/18S/17_aln_256-355_n12260.mat'
tax_file = 'Databases/12_11_2021-23_15_53/Taxonomy_files/Nematoda_18S_tax.tsv'

wc = window_collapser.WindowCollapser(tax_file)
wc.set_matrix(test_file)
wc.collapse()
matrix = np.array(wc.collapsed['seq'].tolist())

# distance calculators
dbId = DistByID(matrix[2:], matrix[:2].reshape((-1, 100)))
dbId.get_dist()

cost_mat = cost_matrix(1, 2)
dbcost = DistByCost(matrix[2:], matrix[:2].reshape((-1, 100)), cost_mat)
dbcost.get_dist()
# voters
majVote = MajorityVote(10, dbId.dists, wc.collapsed['tax'].to_numpy())
majVote = MajorityVote(10, dbcost.dists, wc.collapsed['tax'].to_numpy())
majVote.classify()

weiVo = WeightedVote(10, dbId.dists, wc.collapsed['tax'].to_numpy())
weiVo = WeightedVote(10, dbcost.dists, wc.collapsed['tax'].to_numpy())
weiVo.classify()

#%% classes
class Classifier():
    def __init__(self, matrix, tax_codes, transition = 1, transversion = 2):
        self.matrix = matrix
        self.tax_codes = tax_codes
        self.cost_matrix = cost_matrix(transition, transversion)
        self.query = None
        self.dists = None

    def set_query(self, query):
        # make sure the query is in adequate format to be processed
        self.query = None
        self.dists = None
        if query is None:
            return
        if len(query.shape) == 1:
            if query.shape[0] == self.matrix.shape[1]:
                self.query = query.reshape((1, query.shape[0]))
        else:
            if query.shape[1] == self.matrix.shape[1]:
                self.query = query
            elif query.shape[0] == self.matrix.shape[1]:
                self.query = query.T
        self.dists = np.zeros((len(self.query), len(self.matrix)))

    def dist_by_id(self):
        for idx, q in enumerate(self.query):
            q_dist = (1-(q == self.matrix)).sum(axis = 1)
            self.dists[idx,:] = q_dist

    def dist_by_cost(self):
        for idx0, query in enumerate(self.query):
            for idx1, ref in enumerate(self.matrix):
                q_cost = 0
                for q, s in zip(query, ref):
                    q_cost += self.cost_matrix[q, s]
                self.dists[idx0, idx1] = q_cost

    def get_dists(self, mode = 'id'):
        if self.query is None:
            return
        else:
            if mode == 'id':
                self.dist_by_id()
            elif mode == 'cost':
                self.dist_by_cost()
    
    def vote_classify(self, k):
        self.K = k
        self.classif = pd.DataFrame(columns = ['Code', 'NN'])
        for idx, q_dist in enumerate(self.dists):
            sorted_dists = np.argsort(q_dist)
            k_idx = sorted_dists[:k]
            k_dists = q_dist[k_idx]
            k_taxes = self.tax_codes[k_idx]

            taxes, counts = np.unique(k_taxes, return_counts = True)
            most_abundant = np.argsort(counts)[-1]
            self.classif.at[idx,:] = [taxes[most_abundant], counts[most_abundant]]
    
    def weight_classify(self, k):
        self.wK = k
        self.weighted_classif = pd.DataFrame(columns = ['Query', 'Code', 'Weight'])
        for q_dist in self.dists:
            sorted_dists = np.argsort(q_dist)
            k_idx = sorted_dists[:k]
            k_taxes = self.tax_codes[k_idx]
            w_dict = {tax:0 for tax in k_taxes}
            
            for idx in k_idx:
                weight = q_dist[idx]

#%% paired distance
# @njit
def get_cluster(clusters, idx):
    for k, v in clusters.items():
        if idx in v:
            return k
# @njit
def get_pd(matrix):
    clusters = {} # numba doesn't like dictionaries!!!!
    clustered = set()
    dist_mat = np.zeros((len(matrix), len(matrix)))

    for idx0, i in enumerate(matrix):
        cluster = set([idx0])
        if idx0 in clustered:
            continue
        for idx1, j in enumerate(matrix[idx0+1:]):
            idx = idx1 + idx0 + 1
            if idx in clustered:
                idxC = get_cluster(clusters, idx)
                dist = dist_mat[idxC, idx]
                
            
            dist = (1-(i == j)).sum()

            if dist == 0:
                cluster.add(idx)
            
            dist_mat[idx0,idx] = dist

        if len(cluster) > 1:
            clusters[idx0] = cluster
            clustered = clustered.union(cluster)
    return dist_mat, clusters

#%% main loop -DELETE
if __name__ == '__main__':
    mat_dir = 'Dataset/12_11_2021-23_15_53/Matrices/Nematoda/18S'
    tax_tab = 'Databases/12_11_2021-23_15_53/Taxonomy_files/Nematoda_18S_tax.tsv'
    acc2tax_tab = 'Databases/13_10_2021-20_15_58/Taxonomy_files/Nematoda_18S_acc2taxid.tsv'
    
    mat_browser = pp.MatrixLoader(mat_dir)
    mat_path = mat_browser.get_matrix_path(17)
    preproc = pp.PreProcessor(mat_path, tax_tab)
    preproc.select_columns(20)
    
    classifier = Classifier(preproc)
    q = classifier.matrix[:10]
    classifier.set_query(q)
    classifier.get_dists('cost')
    classifier.classify(10)
