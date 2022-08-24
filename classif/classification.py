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
def classify(query, data, tax_tab, dist_mat, k=0, mode='mwd', prev_dists=None):
    # query : query sequence
    # data : reference sequences
    # tax_tab : reference taxonomy table
    # dist_mat : distance matrix to be used
    # k : number of neighbours
    # mode : 'm' : majority, 'w' : wKNN, 'd': dwKNN
    # prev_dists : used when trying multiple n_sites
    # classification director, handles distance & support calculation, & neighbour selection
    
    # k defaults to all neighbours (this kills the majority vote)
    k = list(k)
    if k[0] == 0:
        k[0] = data.shape[0]
    
    # calculate distances
    distances = get_dists(query, data, dist_mat)
    # if prev_dists is not none, add previously calculated differences
    if not prev_dists is None:
        distances += prev_dists
    
    # sort neighbours
    neighbours = np.argsort(distances)
    neigh_dists = np.array([d[n] for d,n in zip(distances, neighbours)])
    
    # generate the result tables
    results = {md:classifier(neighbours, tax_tab, k, md, neigh_dists) for md in mode}
    return results, distances

def classifier(neighs, tax_tab, k, mode, distances):
    # classify each row of neighs using the data in tax_tab
    result = []
    
    # classify using every value of k
    for _k in k:
        dists = distances[:,:_k]
        # compute support scores if needed
        if mode == 'w':
            supports = wknn(dists)
        elif mode == 'd':
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
                    
                    report = [idx,rk,tax,count,_k,average_dists,std_dists]
                    if mode != 'm':
                        tax_supports = supports[idx, row == tax]
                        report.append(tax_supports.sum())
                    result.append(report)
    return np.array(result, dtype=np.float32)

def get_classification(results):
    # extract the winning classification from the knn results
    # takes a dict with keys 'm', 'w', 'd'
    # get classification for each query/k/rank combination
    # in the case of majority vote, take the taxon(s) with the highest count in a given k/rank
    # in the case of weighted modes, take the taxon(s) with the highest support score (softmax result) in a given k/rank
    classifications = {}
    for mode, report in results.items():
        mode_classifs = []
        indexes = np.unique(report[:,0])
        k_vals = np.unique(report[:,4])
        ranks = np.unique(report[:,1])
        # get the matrix corresponding to each idx/k/rk combination
        for idx in indexes:
            idx_mat = report[report[:,0] == idx]
            for k in k_vals:
                k_mat = idx_mat[idx_mat[:,4] == k]
                for rk in ranks:
                    rk_mat = k_mat[k_mat[:,1] == rk]
                    if mode == 'm':
                        # winners are the ones with the highest count
                        max_count = rk_mat[:,3].max()
                        max_rows = rk_mat[rk_mat[:,3] == max_count]
                    else:
                        # winners are the ones with the highest support
                        # add scores (softmax)
                        scores = softmax(rk_mat[:,7])
                        score_mat = np.append(rk_mat, scores, axis=1)
                        max_rows = score_mat[(scores == scores.max()).flatten()]
                    mode_classifs.append(max_rows)
        classifications[mode] = np.concatenate(mode_classifs)
    return classifications

def parse_report(report):
    cols_dict = {'m':['idx', 'rank', 'taxon', 'count', 'k', 'mean distance', 'std distance'],
                 'w':['idx', 'rank', 'taxon', 'count', 'k', 'mean distance', 'std distance', 'support', 'score'],
                 'd':['idx', 'rank', 'taxon', 'count', 'k', 'mean distance', 'std distance', 'support', 'score']}
    tables = {mode:pd.DataFrame(matrix, columns = cols_dict[mode]) for mode, matrix in report.items()}
    for mode, table in tables.items():
        table['mode'] = mode
    result_tab = pd.concat(tables.values())
    result_tab.reset_index(drop=True, inplace=True)
    return result_tab

# DEPRECATED beyond this point
def get_neighs(query, data, dist_mat):
    if len(query.shape) == 1:
        query = query.reshape((1, -1))
    # get the neighbours to q sorted by distance
    dists = get_dists(query, data, dist_mat)
    neighs = np.argsort(dists)
    neigh_dists = np.array([d[n] for d,n in zip(dists, neighs)])
    
    return neighs, neigh_dists

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
