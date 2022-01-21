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
    # template class for distance calculators
    def __init__(self, data, query, cost_matrix = None):
        self.data = data # numpy array with reference data. Values rendered as numbers
        self.query = query # numpy array with query sequences. Shape should be (n queries, sequence length)
        self.cost_matrix = cost_matrix # numpy array with susbstitution costs
        self.dists = np.zeros((self.query.shape[0], self.data.shape[0])) # distance between query sequences to every data instance

    @abstractmethod
    def get_dist(self):
        pass

class DistByID(DistCalculator):
    # distances between queries and reference sequences counted as the sum of positions with differing values
    def get_dist(self):
        for idx, q in enumerate(self.query):
            q_dist = (1-(q == self.data)).sum(axis = 1)
            self.dists[idx,:] = q_dist

class DistByCost(DistCalculator):
    # distances value for each position is taken from the cost matrix
    # TODO: cost matrix with diagonal values 0 and all remaining 1 gives distance value equal to distance by identity
    def get_dist(self):
        for idx0, query in enumerate(self.query):
            for idx1, ref in enumerate(self.data):
                q_cost = 0
                for q, s in zip(query, ref):
                    q_cost += self.cost_matrix[q, s]
                self.dists[idx0, idx1] = q_cost

#%% voters
class Voter(ABC):
    # template class for voters
    def __init__(self, K, distances, data_classes):
        self.K = K # number of neighbours to take into account
        self.distances = distances # numpy array. DistCalculator.dists attribute
        self.data_classes = data_classes # pandas series. 'tax' column of a collapsed dataset
        self.classif = False # classification table

    @abstractmethod
    def classify(self):
        pass

class MajorityVote(Voter):
    # each of the nearest neighbours casts a single vote. Query is assigned to the taxon with the most votes
    def classify(self):
        self.classif = pd.DataFrame(columns = ['Code', 'NN'])
        for idx, q_dist in enumerate(self.distances):
            sorted_dists = np.argsort(q_dist)
            k_idx = sorted_dists[:self.K]
            k_taxes = self.data_classes[k_idx]
    
            tax_counts = k_taxes.value_counts()
            max_count = tax_counts.max()
            most_abundant = tax_counts.loc[tax_counts == max_count].index.tolist()
            self.classif.at[idx,:] = [np.array(most_abundant, dtype="object"), max_count]

class WeightedVote(Voter):
    # each of the neighbours casts a vote weighted by a function of its distance to the query
    # TODO: define distance weight function
    def classify(self):
        self.classif = pd.DataFrame(columns = ['Query', 'Code', 'Weight'])
        for q_idx, q_dist in enumerate(self.distances):
            sorted_dists = np.argsort(q_dist)
            k_idx = sorted_dists[:self.K]
            k_taxes = self.data_classes[k_idx].values
            w_dict = {tax:0 for tax in k_taxes}
            
            for idx in k_idx:
                tax = self.data_classes[idx]
                weight = q_dist[idx]
                w_dict[tax] += weight # TODO: this is wrong
            
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
majVote = MajorityVote(10, dbId.dists, wc.collapsed['tax'])
majVote = MajorityVote(10, dbcost.dists, wc.collapsed['tax'])
majVote.classify()

weiVo = WeightedVote(10, dbId.dists, wc.collapsed['tax'])
weiVo = WeightedVote(10, dbcost.dists, wc.collapsed['tax'])
weiVo.classify()


# TODO remake Classifier class, handle collapser (or not), dist calculator and voter selection
class Classifier2():
    def __init__(self, data, tax_codes, transition=1, transversion=2):
        self.data = data
        self.nseqs = data.shape[0]
        self.seqlen = data.shape[1]
        self.tax_codes = tax_codes
        self.cost_matrix = cost_matrix(transition, transversion)
        
        # These dictionaries are used to select the distance calculation and voting mechanisms
        # To add a new one define a DistCalculator/Voter class and add it to the dictionary with a corresponding mode key (or keys)
        self.dist_calculators = {0:DistByID, 'id':DistByID,
                                 1:DistByCost, 'cost':DistByCost}
        self.voters = {0:MajorityVote, 'maj':MajorityVote,
                       1:WeightedVote, 'wei':WeightedVote}
    
    def set_query(self, query):
        self.query = query.reshape((-1, self.seqlen))
    
    def set_k(self, k):
        try:
            self.k = int(k)
        except:
            self.k = 3

    def get_dists(self, mode = 0):
        # creates a DistCalculator object and calculates query distances
        # modes:
            # 0 : distance by identity
            # 1 : distance by cost
        dist_calculator = self.dist_calculators[mode](self.query, self.data, self.cost_matrix)
        dist_calculator.get_dist()
        self.distances = dist_calculator.dists
    
    def classify(self, mode = 0):
        # creates a Voter object and classifies query instances
        # modes:
            # 0 : majority vote
            # 1 : weighted vote
        voter = self.voters[mode](self.K, self.distances, self.tax_codes)
        voter.classify()
        self.classif = voter.classif

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
