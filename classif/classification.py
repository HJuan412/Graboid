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

def calibration_classify(q, k_range, data, tax_tab, dist_mat, q_name=0, mode='majority', support_func=wknn):
    # classification function used in calibration, generates a prediction for multiple values of K
    # q : query sequence
    # k_range : possible numbers of neighbours
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
    
    results = []
    # get the data
    neighs, dists = get_neighs(q, data, dist_mat)
    
    # generate the result tables
    # iterate trough the values of k in k_range, generate a result for each
    for k in k_range:
        k_neighs = neighs[:k]
        k_dists = dists[:k]
        if mode == 'majority':
            k_results = classify_majority(k_neighs, tax_tab, dist_mat)
            k_results['total_K'] = k
        elif mode == 'weighted':
            supports = support_func(k_dists)
            k_results = classify_weighted(k_neighs, supports, tax_tab)
        k_results['query'] = q_name
        results.append(k_results)

    return results

def classify_majority(neighs, tax_tab, dist_mat):
    result_tab = pd.DataFrame(columns = ['query', 'rank', 'taxon', 'K', 'total_K'])
    
    sub_tax = tax_tab.iloc[neighs]
    for rank in tax_tab.columns:
        # count reperesentatives of each taxon amongst the k neighbours
        tax_counts = sub_tax[rank].value_counts(ascending = False)
        # select winners as those with the maximum detected number of neighbours
        # this handles the eventuality of a draw
        max_val = tax_counts.max()
        winners = tax_counts.loc[tax_counts == max_val].index
        for wn in winners:
            row = pd.Series({'rank':rank,
                             'taxon':wn,
                             'K':max_val})
            result_tab = result_tab.append(row, ignore_index=True)
    return result_tab

def classify_weighted(neighs, supports, tax_tab):
    result_tab = pd.DataFrame(columns = ['query', 'rank', 'taxon', 'K', 'support', 'mean_support', 'std_support'])

    sub_tax = tax_tab.iloc[neighs]
    for rank in tax_tab.columns:
        rank_tab = pd.DataFrame(columns = ['query', 'rank', 'taxon', 'K', 'support', 'mean_support', 'std_support'])
        taxes = sub_tax[rank].unique()
        for tax in taxes:
            # calculate the total, mean and std support of each represented taxon 
            tax_neighs = np.argwhere(sub_tax[rank].to_numpy() == tax)
            tax_supports = supports[tax_neighs]
            row = pd.DataFrame({'rank':rank,
                                'taxon':tax,
                                'K':len(tax_neighs),
                                'support':tax_supports.sum(),
                                'mean_support':tax_supports.mean(),
                                'std_support':tax_supports.std()},
                               index = [0])
            rank_tab = pd.concat([rank_tab, row], ignore_index=True)
        result_tab = pd.concat([result_tab, rank_tab], ignore_index=True)
    
    return result_tab

def get_classif(results, mode='majority'):
    # reads the classification table, used to generate the confusion table during
    # calibration
    if mode == 'majority':
        return get_classif_majoriy(results)
    elif mode == 'weighted':
        return get_classif_weighted(results)

def get_classif_majoriy(results):
    # for each rank, assigned taxon is the most populated (if there is a draw), classification is left None
    classif = pd.Series()
    for rank, rk_tab in results.groupby('rank'):
        classif[rank] = None
        if rk_tab.shape[0] == 1:
            classif[rank] = rk_tab['taxon'].iloc[0]
    return classif

def get_classif_weighted(results):
    # classification is based on highest total support
    classif = pd.Series()
    for rank, rk_tab in results.groupby('rank'):
        classif[rank] = rk_tab.sort_values('support', ascending=False)['taxon'].iloc[0]
    return classif