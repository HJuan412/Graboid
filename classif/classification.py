#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 10:21:15 2022

@author: hernan
"""
#%% libraries
import numba as nb
import numpy as np
import pandas as pd

#%% functions
@nb.njit
def calc_distance(seq1, seq2, dist_mat):
    dist = 0
    for x1, x2 in zip(seq1, seq2):
        dist += dist_mat[x1, x2]
    
    return dist

@nb.njit
def get_dists(query, data, dist_mat):
    # new version, performs calculations for multiple queries
    # single query vector should have dimensions (1, len(seq))
    if len(query.shape) == 1:
        query = query.reshape((1,-1))
    dist = np.zeros((query.shape[0], data.shape[0]), dtype = np.float32)
    for idx0, q in enumerate(query):
        for idx1, d in enumerate(data):
            dist[idx0, idx1] = calc_distance(q, d, dist_mat)
    return dist

# support functions
# these take a matrix of distances (dists) as input and return a vector of calculated supports (supps)
def wknn(dists):
    # weighted KNN, weight of an instance linearly decreases with distance along a range of [0, 1]
    d1 = dists.min(axis=1).reshape((len(dists), -1)) # axis=1 get the min/max value per row. Reshape to turn into a column vector
    dk = dists.max(axis=1).reshape((len(dists), -1))
    rang = dk - d1 # range of values
    supps = np.ones(dists.shape) # default support values
    valid = (rang != 0).flatten() # instances in which all neighbours overlap with the query are invalid, the range is 0
    supps[valid] = (dk[valid] - dists[valid]) / rang[valid]
    # equals an array of (dk - di) / (dk - d1), the distance function for wknn
    return supps

def dwknn(dists):
    # double weighted KNN, weight of an instance decreases "exponentially" with distance
    d1 = dists.min(axis=1).reshape((len(dists), -1))
    dk = dists.max(axis=1).reshape((len(dists), -1))
    rang = dk + d1
    term1 = wknn(dists)
    term2 = np.ones(dists.shape) # default support values
    valid = (rang != 0).flatten() # instances in which all neighbours overlap with the query are invalid, the range is 0
    term2[valid] = rang[valid] / (dk[valid] + dists[valid])
    supps = term1 * term2
    # equals an array of ((dk - di) / (dk - d1)) * ((dk + d1) / (dk + di))
    return supps

def softmax(supports):
    div = np.exp(supports).sum()
    softmax_scores = np.exp(supports) / div
    softmax_scores = softmax_scores.reshape((len(softmax_scores), 1))
    return softmax_scores

# classification functions
def place_query(query, data, dist_mat, prev_dists=None):
    distances = get_dists(query, data, dist_mat)
    # if prev_dists is not none, add previously calculated differences
    if not prev_dists is None:
        distances += prev_dists
    return distances

def sort_classify(distances, tax_tab, k, mode):
    # order neighbours by proximity to queries
    # classify each row of neighbours using the data in tax_tab
    result = []
    
    # sort neighbours
    neighbours = np.argsort(distances)
    sorted_dists = np.array([d[n] for d,n in zip(distances, neighbours)])
    
    # classify for each value of k
    for _k in k:
        dists = sorted_dists[:,:_k]
        # get supports
        if mode == 'w':
            supports = wknn(dists)
        elif mode == 'd':
            supports = dwknn(dists)
        
        # classify instances
        for idx, n in enumerate(neighbours[:,:_k]):
            # select neighbour taxonomies
            sub_tax = tax_tab.iloc[n].to_numpy().T
            # assign classification for each rank
            for rk, row in enumerate(sub_tax):
                # count reperesentatives of each taxon amongst the k neighbours
                taxs, counts = np.unique(row, return_counts = True)
                # for each represented taxon get representatives, average distances and std
                for tax, count in zip(taxs, counts):
                    average_dists = dists[idx, row == tax].mean()
                    std_dists = dists[idx, row == tax].std()
                    report = [idx, rk, tax, count, _k, average_dists, std_dists]
                    if mode != 'm':
                        # add total supports to the report
                        tax_supports = supports[idx, row == tax]
                        report.append(tax_supports.sum())
                    result.append(report)
    return np.array(result, dtype=np.float32)

def build_report(result, mode):
    # extract the winning classification from the knn results
    # get classification for each query/k/rank combination
    # in the case of majority vote (mode = 'm'), take the taxon(s) with the highest count in a given k/rank
    # in the case of weighted modes (mode = 'w' o 'd'), take the taxon(s) with the highest support score (softmax result) in a given k/rank
    report = []
    indexes = np.unique(result[:,0])
    k_vals = np.unique(result[:,4])
    ranks = np.unique(result[:,1])
    # get the matrix corresponding to each idx/k/rk combination
    for idx in indexes:
        i_mat = result[result[:,0] == idx]
        for k in k_vals:
            ik_mat = i_mat[i_mat[:,4] == k]
            for rk in ranks:
                ikr_mat = ik_mat[ik_mat[:,1] == rk]
                # ikr_mat (idx, k, rk) built this way, array searches get smaller in size for every loop
                if mode == 'm':
                    # winners are the ones with the highest count
                    max_count = ikr_mat[:,3].max()
                    max_rows = ikr_mat[ikr_mat[:,3] == max_count]
                else:
                    # winners are the ones with the highest support
                    # add scores (softmax)
                    scores = softmax(ikr_mat[:,7])
                    score_mat = np.append(ikr_mat, scores, axis=1)
                    max_rows = score_mat[(scores == scores.max()).flatten()]
                report.append(max_rows)
    return np.concatenate(report)

def parse_report(report, mode):
    columns = ['idx', 'rank', 'taxon', 'count', 'k', 'mean distance', 'std distance', 'support', 'score']
    cols = columns[:report.shape[1]]
    table = pd.DataFrame(report, columns = cols)
    table['mode'] = mode
    return table

def classify(query, data, dist_mat, tax_tab, k=0, modes='mwd', prev_dists=None):
    # classification director, handles distance & support calculation, & neighbour selection
    # query : query sequence
    # data : reference sequences
    # tax_tab : reference taxonomy table
    # dist_mat : distance matrix to be used
    # k : number of neighbours, int or list
    # mode : 'm' : majority, 'w' : wKNN, 'd': dwKNN
    # prev_dists : used when trying multiple n_sites
    
    # k defaults to all neighbours (this kills the majority vote)
    k = list(k)
    if k[0] == 0:
        k[0] = data.shape[0]
    
    # get distances from each query to each reference
    distances = place_query(query, data, dist_mat, prev_dists)
    reports = []
    for mode in modes:
        # classify and build a report for each mode
        pre_classif = sort_classify(distances, tax_tab, k, mode)
        pre_report = build_report(pre_classif, mode)
        reports.append(parse_report(pre_report, mode))
    report = pd.concat(reports, ignore_index=True)
    return report, distances
