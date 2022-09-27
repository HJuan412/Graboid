#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 10:21:15 2022

@author: hernan
"""
#%% libraries
import logging
import numba as nb
import numpy as np
import pandas as pd

#%% set logger
logger = logging.getLogger('Graboid.Classification')
logger.setLevel(logging.DEBUG)
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
    # if len(query.shape) == 1:
    #     query = query.reshape((1,-1))
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

@nb.njit
def wknn_nb(dists):
    # weighted KNN, weight of an instance linearly decreases with distance along a range of [0, 1]
    d1 = np.array([d.min() for d in dists]).reshape((len(dists), -1)) # axis=1 get the min/max value per row. Reshape to turn into a column vector
    dk = np.array([d.max() for d in dists]).reshape((len(dists), -1))
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

@nb.njit
def dwknn_nb(dists):
    # double weighted KNN, weight of an instance decreases "exponentially" with distance
    d1 = np.array([d.min() for d in dists]).reshape((len(dists), -1)) # axis=1 get the min/max value per row. Reshape to turn into a column vector
    dk = np.array([d.max() for d in dists]).reshape((len(dists), -1))
    rang = dk + d1
    term1 = wknn_nb(dists)
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

#%% classifier functions family
# if a new classification mode is added, name it as classify_<mode> with arguments neighs, sorted_dists, tax_tab, k (even if some are not used)
# should return a numpy array and a list of column names
# add corresponding key values to classif_funcs, classif_modes, classif_longnames
def classify_m(neighs, sorted_dists, tax_tab, k):
    result = []
    # classify for each value of k
    for _k in k:
        dists = sorted_dists[:,:_k]
        # classify instances
        for idx, n in enumerate(neighs[:,:_k]):
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
                    result.append(report)
    columns = ['idx', 'rk', 'tax', 'count', '_k', 'average_dists', 'std_dists']
    return np.array(result, dtype=np.float32), columns

# TODO: tax_tab should be turned to numpy array before running numba functions
@nb.njit
def classify_m_nb(neighs, sorted_dists, tax_tab, k):
    result = [[0.0] for i in range(0)]
    # classify for each value of k
    for _k in k:
        dists = sorted_dists[:,:_k]
        # classify instances
        for idx, n in enumerate(neighs[:,:_k]):
            # select neighbour taxonomies
            sub_tax = tax_tab[n].T
            # assign classification for each rank
            for rk, row in enumerate(sub_tax):
                # count reperesentatives of each taxon amongst the k neighbours
                taxa = np.unique(row)
                # for each represented taxon get representatives, average distances and std
                for tax in taxa:
                    tax_idxs = row == tax
                    count = tax_idxs.sum()
                    average_dists = dists[idx][tax_idxs].mean()
                    std_dists = dists[idx][tax_idxs].std()
                    report = [idx, rk, tax, count, _k, average_dists, std_dists]
                    result.append(report)
    columns = ['idx', 'rk', 'tax', 'count', '_k', 'average_dists', 'std_dists']
    return np.array(result, dtype=np.float32), columns

def classify_w(neighs, sorted_dists, tax_tab, k):
    result = []
    # classify for each value of k
    for _k in k:
        dists = sorted_dists[:,:_k]
        # get supports
        supports = wknn(dists)
        
        # classify instances
        for idx, n in enumerate(neighs[:,:_k]):
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
                    # add total supports to the report
                    tax_supports = supports[idx, row == tax]
                    report.append(tax_supports.sum())
                    report.append(np.median(tax_supports))
                    result.append(report)
    columns = ['idx', 'rk', 'tax', 'count', '_k', 'average_dists', 'std_dists', 'total_support', 'median_support']
    return np.array(result, dtype=np.float32), columns

@nb.njit
def classify_w_nb(neighs, sorted_dists, tax_tab, k):
    result = [[0.0] for i in range(0)]
    # classify for each value of k
    for _k in k:
        dists = sorted_dists[:,:_k]
        # get supports
        supports = wknn_nb(dists)
        # classify instances
        for idx, n in enumerate(neighs[:,:_k]):
            # select neighbour taxonomies
            sub_tax = tax_tab[n].T
            # assign classification for each rank
            for rk, row in enumerate(sub_tax):
                # count reperesentatives of each taxon amongst the k neighbours
                taxa = np.unique(row)
                # for each represented taxon get representatives, average distances and std
                for tax in taxa:
                    tax_idxs = row == tax
                    count = tax_idxs.sum()
                    average_dists = dists[idx][tax_idxs].mean()
                    std_dists = dists[idx][tax_idxs].std()
                    report = [idx, rk, tax, count, _k, average_dists, std_dists]
                    # add total supports to the report
                    tax_supports = supports[idx][tax_idxs]
                    report.append(tax_supports.sum())
                    report.append(np.median(tax_supports))
                    result.append(report)
    columns = ['idx', 'rk', 'tax', 'count', '_k', 'average_dists', 'std_dists', 'total_support', 'median_support']
    return np.array(result, dtype=np.float32), columns

def classify_d(neighs, sorted_dists, tax_tab, k):
    result = []
    # classify for each value of k
    for _k in k:
        dists = sorted_dists[:,:_k]
        # get supports
        supports = dwknn(dists)
        
        # classify instances
        for idx, n in enumerate(neighs[:,:_k]):
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
                    # add total supports to the report
                    tax_supports = supports[idx, row == tax]
                    report.append(tax_supports.sum())
                    report.append(np.median(tax_supports))
                    result.append(report)
    columns = ['idx', 'rk', 'tax', 'count', '_k', 'average_dists', 'std_dists', 'total_support', 'median_support']
    return np.array(result, dtype=np.float32), columns

@nb.njit
def classify_d_nb(neighs, sorted_dists, tax_tab, k):
    result = [[0.0] for i in range(0)]
    # classify for each value of k
    for _k in k:
        dists = sorted_dists[:,:_k]
        # get supports
        supports = dwknn_nb(dists)
        
        # classify instances
        for idx, n in enumerate(neighs[:,:_k]):
            # select neighbour taxonomies
            sub_tax = tax_tab[n].T
            # assign classification for each rank
            for rk, row in enumerate(sub_tax):
                # count reperesentatives of each taxon amongst the k neighbours
                taxa = np.unique(row)
                # for each represented taxon get representatives, average distances and std
                for tax in taxa:
                    tax_idxs = row == tax
                    count = tax_idxs.sum()
                    average_dists = dists[idx][tax_idxs].mean()
                    std_dists = dists[idx][tax_idxs].std()
                    report = [idx, rk, tax, count, _k, average_dists, std_dists]
                    # add total supports to the report
                    tax_supports = supports[idx][tax_idxs]
                    report.append(tax_supports.sum())
                    report.append(np.median(tax_supports))
                    result.append(report)
    columns = ['idx', 'rk', 'tax', 'count', '_k', 'average_dists', 'std_dists', 'total_support', 'median_support']
    return np.array(result, dtype=np.float32), columns

# classification functions paired to each mode
classif_funcs = {'m':classify_m,
                 'w':classify_w,
                 'd':classify_d}
classif_funcs_nb = {'m':classify_m_nb,
                    'w':classify_w_nb,
                    'd':classify_d_nb}
# attribute used to get winner classification paired to each mode
classif_modes = {'m':3,
                 'w':7,
                 'd':7}
# full name for each classification mode
classif_longnames = {'m':'majority',
                     'w':'wKNN',
                     'd':'dwKNN'}
#%%
def get_classif(result, attr):
    # extract the winning classification from the knn results
    # get classification for each query/k/rank combination using the given attr
    # in the case of majority vote (mode = 'm'), take the taxon(s) with the highest count in a given k/rank
    # in the case of weighted modes (mode = 'w' o 'd'), take the taxon(s) with the highest support in a given k/rank
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
                max_attr_val = ikr_mat[:, attr].max()
                max_rows = ikr_mat[ikr_mat[:, attr] == max_attr_val]
                report.append(max_rows)
    return np.concatenate(report)

def parse_report(report, cols, mode):
    table = pd.DataFrame(report, columns = cols)
    table['mode'] = mode
    return table

def classify(query, data, dist_mat, tax_tab, k=0, modes='mwd', prev_dists=None, get_winners=False):
    # classification director, handles distance & support calculation, & neighbour selection
    # query : query sequence
    # data : reference sequences
    # tax_tab : reference taxonomy table
    # dist_mat : distance matrix to be used
    # k : number of neighbours, int or list
    # mode : 'm' : majority, 'w' : wKNN, 'd': dwKNN
    # prev_dists : used when trying multiple n_sites
    # get_winners : used to select only the winning classification, otherwise, return all results
    
    # k defaults to all neighbours (this kills the majority vote)
    k = list(k)
    if k[0] == 0:
        k[0] = data.shape[0]
    
    # get distances from each query to each reference
    distances = place_query(query, data, dist_mat, prev_dists)
    # sort neighbours
    neighbours = np.argsort(distances)
    sorted_dists = np.array([d[n] for d,n in zip(distances, neighbours)])
    
    reports = []
    for mode in modes:
        try:
            classifier = classif_funcs[mode]
        except KeyError:
            print(f'Error: Mode {mode} not valid')
            continue
        # classify and build a report for each mode
        pre_classif, columns = classifier(neighbours, sorted_dists, tax_tab, k)
        if get_winners:
            pre_classif = get_classif(pre_classif, classif_modes[mode])
            
        reports.append(parse_report(pre_classif, columns, classif_longnames[mode]))
        
    report = pd.concat(reports, ignore_index=True)
    return report, distances
