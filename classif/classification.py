#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 10:21:15 2022

@author: hernan
"""
#%% libraries
from . import distance
import numba as nb
import numpy as np
import pandas as pd

#%% functions
@nb.njit
def get_dists(query, data, dist_mat):
    dist = np.zeros(data.shape[0])
    for idx, ref in enumerate(data):
        dist[idx] = distance.calc_distance(query, ref, dist_mat)
    return dist

def get_neighs(q, data, dist_mat):
    # get the neighbours to q sorted by distance
    dists = get_dists(q, data, dist_mat)
    neighs = np.argsort(dists)
    neigh_dists = dists[neighs]
    
    return neighs, neigh_dists

# support functions
# these take a vector of distances (dists) as input and return a vector of calculated supports (supps)
def wknn(dists):
    # weighted KNN, weight of an instance linearly decreases with distance along a range of [0, 1]
    d1 = dists[0]
    dk = dists[-1]
    rang = dk - d1
    supps = np.ones(len(dists))
    if d1 != dk:
        supps = (dk - dists) / rang
        # equals an array of (dk - di) / (dk - d1), the distance function for wknn
    return supps

def dwknn(dists):
    # double weighted KNN, weight of an instance decreases "exponentially" with distance
    d1 = dists[0]
    dk = dists[-1]
    rang = dk + d1
    supps = np.ones(len(dists))
    if d1 != dk:
        term1 = wknn(dists)
        term2 = rang / (dk + dists)
        supps = term1 * term2
        # equals an array of ((dk - di) / (dk - d1)) * ((dk + d1) / (dk + di))
    return supps

# classification functions
def classify(q, k, data, tax_tab, dist_mat, q_name=0, mode='majority', support_func=wknn):
    # q : query sequence
    # k : number of neighbours
    # data : reference sequences
    # tax_tab : reference taxonomy table
    # dist_mat : distance matrix to be used
    # q_name : name of the query (defaults to 0)
    # mode : "majority" for majority vote or "weighted" for distance weighted
    # support_func : wknn or dwknn, used to calculate neighbour support in "weighted" mode
    # classification director, handles distance & support calculation, & neighbour selection
    # k defaults to all neighbours (this kills the majority vote)
    
    # TODO: use multiple queries, q is a 2d array, q_name is an array/list

    # TODO: make it so that it can use multiple classification criteria (majority/weighted + wknn/weighted + dwknn) in a single run
    # TODO: (cont) should return a list of result tables or a single table?
    if k == 0:
        k = data.shape[0]
    
    # get the data
    neighs, dists = get_neighs(q, data, dist_mat)
    neighs = neighs[:k]
    dists = dists[:k]
    
    # generate the result tables
    if mode == 'majority':
        results = classify_majority(neighs, tax_tab, dist_mat)
        results['total_K'] = k
    elif mode == 'weighted':
        supports = support_func(dists)
        results = classify_majority(neighs, supports, tax_tab)
    results['query'] = q_name
    return results

def calibration_classify(q, k_range, data, tax_tab, dist_mat, q_name=0):
    # classification function used in calibration, generates a prediction for multiple values of K
    # q : query sequence
    # k_range : possible numbers of neighbours
    # data : reference sequences
    # tax_tab : reference taxonomy table
    # dist_mat : distance matrix to be used
    # q_name : name of the query (defaults to 0)
    
    # classification director, handles distance & support calculation, & neighbour selection
    # k defaults to all neighbours (this kills the majority vote)
    
    # TODO: use multiple queries, q is a 2d array, q_name is an array/list
        
    maj_results = []
    wknn_results = []
    dwknn_results = []

    # get distances and sorted neighbours
    neighs, dists = get_neighs(q, data, dist_mat)
    
    # generate the result tables
    # iterate trough the values of k in k_range, generate a result for each
    for k in k_range:
        k_neighs = neighs[:k]
        k_dists = dists[:k]
        # calib majority
        maj_results.append(classify_majority(k_neighs, tax_tab, q_name, k))
        # calib wknn
        supports_wknn = wknn(k_dists)
        wknn_results.append(classify_weighted(k_neighs, supports_wknn, tax_tab, q_name, k))
        # calib dwknn
        supports_dwknn = dwknn(k_dists)
        dwknn_results.append(classify_weighted(k_neighs, supports_dwknn, tax_tab, q_name, k))
    return maj_results, wknn_results, dwknn_results

def classify_majority(neighs, tax_tab, q_name=0, total_k=1):
    # neighs: list of neighbours to consider in the classification
    # tax_tab: taxonomic dataframe of the training set
    # q_name: query name
    # total_k: k neighbours considered
    
    result = []
    # select neighbour taxonomies
    sub_tax = tax_tab.iloc[neighs].to_numpy().T
    for rank, row in enumerate(sub_tax):
        # count reperesentatives of each taxon amongst the k neighbours
        taxs, counts = np.unique(row, return_counts = True)
        # select winners as those with the maximum detected number of neighbours
        # this handles the eventuality of a draw
        max_val = np.max(counts)
        winn_idx = np.argwhere(counts == max_val)
        for winn in winn_idx:
            result.append([q_name, rank, taxs[winn][0], max_val, total_k])
    return np.array(result)

def classify_weighted(neighs, supports, tax_tab, q_name=0, total_k=1):
    # neighs: list of neighbours to consider in the classification
    # tax_tab: taxonomic dataframe of the training set
    # dist_mat: unused
    # q_name: query name
    # total_k: k neighbours considered
    
    result = []
    # select neighbour taxonomies
    sub_tax = tax_tab.iloc[neighs].to_numpy().T
    
    for rank, row in enumerate(sub_tax):
        taxes, counts = np.unique(row, return_counts=True)
        for tax, count in zip(taxes, counts):
            # calculate the total, mean and std support of each represented taxon 
            tax_supports = supports[row == tax]
            report = [q_name,
                      rank,
                      tax,
                      count,
                      total_k,
                      tax_supports.sum(),
                      tax_supports.mean(),
                      tax_supports.std()]
            result.append(report)
    return np.array(result)

def get_classif(results, mode='majority'):
    # reads the classification table, used to generate the confusion table during
    # calibration
    if mode == 'majority':
        return get_classif_majoriy(results)
    elif mode == 'weighted':
        return get_classif_weighted(results)

def get_classif_majoriy(results, n_ranks=6):
    # get single classification for each rank from the results table
    # n_ranks 6 by default (phylum, class, order, family, genus, species), could be modifyed to include less/more (should be done automatically)
    # for each rank, assigned taxon is the most populated (if there is a draw), classification is left None
    
    # initalize classification array
    classif = np.empty(n_ranks)
    classif[:] = np.NaN
    # get classification for each rank
    for rank in np.unique(results[:,1]):
        # only classify if it is unambiguous, if there are multiple winners for the current taxon, classification is left empty
        rank_mat = results[results[:,1] == rank]
        if rank_mat.shape[0] == 1:
            classif[rank] = rank_mat[0,2]
    
    return classif

def get_classif_weighted(results, n_ranks=6):
    # get single classification for each rank from the weighted results table
    # classification is based on highest total support
    # n_ranks 6 by default (phylum, class, order, family, genus, species), could be modifyed to include less/more (should be done automatically)
    
    # initialize classification array
    classif = np.empty(n_ranks)
    classif[:] = np.NaN
    # get classification for each rank
    for rank in np.unique(results[:,1].astype(int)):
        rank_mat = results[results[:,1] == rank]
        # classification is assigned to the taxon with the highest support
        # if classification is ambiguous leave it empty
        max_supp = rank_mat[:, 5].max()
        winner = rank_mat[rank_mat[:,5] == max_supp]
        if winner.shape[0] == 1:
            classif[rank] = winner[0,2]
    return classif
