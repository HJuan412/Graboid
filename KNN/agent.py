#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:10:17 2025

@author: hernan

This script contains the knn agent class, handles data loading,
learning (calibration), and classification.
"""

#%% modules
import numpy as np

from Graboid.database import data_holder
#%% functions
#%% classes

class KNNagent:
    def __init__(self, transition=1, transversion=2):
        self.transition = 1
        self.transversion = 2
        self.data = data_holder.DataHolder()
        self.calibrator = None
        
    def load_data(self, db_dir, min_coverage=0, required_rank='family'):
        self.data.load_reference(db_dir, min_coverage, required_rank)
    
    def learn(self,
              max_n,
              step_n,
              max_k,
              step_k,
              row_thresh,
              col_thresh,
              min_seqs,
              rank,
              min_n,
              min_k,
              criterion,
              threads=1):
        self.calibrator.grid_search()
        
    def classify(self):
        pass

#%%
agent = KNNagent()
agent.load_data('test/nem_18s/')

#%%
cost_mat = np.zeros((5,5))
cost_mat[0] = 1.25
cost_mat[:,0] = 1.25
cost_mat[[1,3], [3,1]] = 1
cost_mat[[2,4], [4,2]] = 1
cost_mat[[1,1,3,3],[2,4,2,4]] = 2
cost_mat[[2,2,4,4],[1,3,1,3]] = 2

#%%
from Graboid.KNN import calibrator

result = calibrator.calibrate_sliding(agent.data.R, 300, 100, 25, 5, 3, 1, cost_mat, threads = 6)
#%%
import functools
import pandas as pd


grids = result[1]

def get_n_matrix(grids, n_range):
    n_matrix = pd.DataFrame(False, index=np.arange(len(grids)), columns=n_range)
    for idx, g in enumerate(grids):
        n_matrix.loc[idx, g.n_range.keys()] = True
    return n_matrix

res = calibrator.GridFinal(result[0], result[1], np.arange(5,26,5), np.arange(1,4,1))













#%%
from Graboid.preprocess import feature_selection

agent.data.select_region(400, 600)
gain, counts = feature_selection.get_information_gain(agent.data.R.collapsed, agent.data.R.lineage_collapsed)

cost_mat = np.zeros((5,5))
cost_mat[0] = 1.25
cost_mat[:,0] = 1.25
cost_mat[[1,3], [3,1]] = 1
cost_mat[[2,4], [4,2]] = 1
cost_mat[[1,1,3,3],[2,4,2,4]] = 2
cost_mat[[2,2,4,4],[1,3,1,3]] = 2

#%%
from Graboid.calibration import cal_main
#%% distance calculations
sites = cal_main.get_sites(gain, np.arange(5,26,5))
dists = cal_main.get_distances(agent.data.R.collapsed, sites, cost_mat)
n_range = np.arange(5,26,5)
k_range = np.arange(1,5)

lin = agent.data.R.lineage_collapsed
classifications, supports = cal_main.classify(dists, lin, k_range, criterion='orbit', threads=1)

grid_classif = cal_main.GridResult(n_range, k_range, classifications, lin.columns.values, sites)
grid_supp = cal_main.GridResult(n_range, k_range, supports, lin.columns.values)


#%%
import numba as nb

@nb.njit
def build_confusion_single(res, lin, taxa):
    # initialize confusion matrix
    confusion = np.zeros((res.shape[0], res.shape[1], len(taxa), len(taxa)), dtype=np.int32)
    # axis 2 holds real taxa, axis 3 holds predicted taxa
    for idx0, tax0 in enumerate(taxa):
        for idx1, tax1 in enumerate(taxa):
            # count number of instances of tax0 predicted as tax1
            confusion[:,:,idx0, idx1] = np.sum((lin == tax0) & (res == tax1), axis=2)
    return confusion

def build_rank_confusion(rk_results, rk_lineage):
    """
    Build confusion matrix for a given rank.

    Parameters
    ----------
    rk_results : numpy.array
        3d array of shape [range_k, #methods, #queries] containing predicted classifications at a given rank for each query and each combination of k/method.
    rk_lineage : numpy.array
        Array containing the real taxonomic classification of each query sequence at the given rank.

    Returns
    -------
    confusion : numpy.array
        4d array [range_k, #methods, #taxa_in_rank, #taxa_in_rank]. Contains confusion for each taxa in rank
    uniq_tax : numpy.array
        Array containing taxonomic IDs of every taxa in rank.

    """
    # reshape lineage array to match results -> rsulting shape is [range_k, #methods, # queries]
    reshaped_lin = np.tile(rk_lineage, (rk_results.shape[0], rk_results.shape[1], 1))
    # list unique taxa in the real lineage
    uniq_tax = np.unique(np.append(rk_lineage.values, 0))
    # build confusion matrix
    confusion = build_confusion_single(rk_results, reshaped_lin, uniq_tax)
    return confusion, uniq_tax

def build_n_confusion(results, lineage):
    """
    Build confusion matrix for the classification results for a given value of n

    Parameters
    ----------
    results : numpy.array
        Array of shape [range k, #methods, #queries, #ranks].
    lineage : pandas.DataFrame
        Dataframe containing the taxonomic information of each reference sequence.

    Returns
    -------
    n_confusion : list
        List of 4d arrays of shape [range_k, #methods, #taxa_in_rank, #taxa_in_rank].
        Each array contains the confusion matrices for each k-method combination for a given taxonomic rank.
        Axis 2 holds real taxa, axis 3 holds predicted taxa
    n_taxa : list
        List of arrays. Each array contains the unique taxa contained in each taxonomic rank (used to index columns and rows (axes 2 & 3) of the confusion matrices).

    """
    # initialize lists
    n_confusion = []
    n_taxa = []
    # generate confusion matrices for each rank
    for rk_idx, rk in enumerate(lineage.columns):
        # extract rank predictions & real lineage
        rk_results = results[:,:,:,rk_idx]
        rk_lineage = lineage.iloc[:,rk_idx]
        # build confusion matrix of rank
        rk_confusion, rk_taxa = build_rank_confusion(rk_results, rk_lineage)
        n_confusion.append(rk_confusion)
        n_taxa.append(rk_taxa)
    return n_confusion, n_taxa

def build_confusion(results_grid, lineage):
    """
    Builds the confusion matrices of each parameter combination for each taxonomic range.

    Parameters
    ----------
    results_grid : cal_main.GridResult
        Object containing the predicted taxonomic classification of each query for each parameter combination in each rank.
    lineage : pandas.DataFrame
        Dataframe containing the taxonomic information of each reference sequence.

    Returns
    -------
    result : cal_main.GridConfusion
        Object containing the generated confusion matrices for each parameter combination in each taxon of each rank.

    """
    confusion = []
    taxa = [] # taxa is the same for each value of n, is overwritten every time
    # build confusion tables for each value of n
    for n_results in results_grid.grid:
        n_confusion, taxa = build_n_confusion(n_results, lineage)
        confusion.append(n_confusion)
    n_ranks = len(taxa)
    
    # merge n confusion matrices for each rank, building the 5d arrays of shape [range_n, range_k, #methods, #taxa, #taxa]
    confusion2 = []
    for rk in range(n_ranks):
        confusion2.append(np.array([n_conf[rk] for n_conf in confusion]))
    
    # build Grid class
    result = GridConfusion(results_grid.n_range, results_grid.k_range, confusion2, taxa, results_grid.ranks, lineage.shape[0])
    return result
    
class GridConfusion:
    def __init__(self, n_range, k_range, confusion, taxa, ranks, n_seqs):
        self.n_range = n_range
        self.k_range = k_range
        self.mth_range = {'unweighted':0, 'u':0, 'wknn':1, 'w':1, 'dwknn':2, 'd':2}
        self.confusion = {rk:rk_confusion for rk, rk_confusion in zip(ranks, confusion)}
        self.taxa = {rk:rk_taxa for rk, rk_taxa in zip(ranks, taxa)}
        self.ranks = ranks
        self.n_seqs = n_seqs
    
    def __getitem__(self, rank):
        return self.confusion[rank]
    
    def cell(self, n, k, method):
        n_idx = self.n_range[n]
        k_idx = self.k_range[k]
        mth_idx = self.mth_range[method]
        result = {rk:conf[n_idx, k_idx, mth_idx] for rk, conf in self.confusion.items()}
        return result
    
    def icell(self, n, k, method):
        result = {rk:conf[n, k, method] for rk, conf in self.confusion.items()}
        return result

#%%
conf = build_confusion(grid_classif, lin)
#%%
import pandas as pd
# result = grid_classif.grid[0]

# conf, tx = build_n_confusion(result, lin)
# total_seqs = result.shape[-2]
# rk_confusion = conf[2]

def get_rk_metrics(rk_confusion, total_seqs):
    diag = rk_confusion[:,:,:, np.arange(rk_confusion.shape[3]), np.arange(rk_confusion.shape[4])]
    sum_pred = np.sum(rk_confusion, axis=3)
    sum_real = np.sum(rk_confusion, axis=4)

    tp = diag
    fp = sum_pred - tp
    fn = sum_real - tp
    tn = total_seqs - (tp + fp + fn)
    
    acc = (tp + tn) / (tp + tn + fp + fn)
    prc = np.nan_to_num(tp / (tp + fp), 0)
    rec = np.nan_to_num(tp / (tp + fn), 0)
    f1 = np.nan_to_num((2 * prc * rec) / (prc + rec), 0)
    return acc, prc, rec, f1

def get_metrics(grid_confusion, total_sequences):
    acc = []
    prc = []
    rec = []
    f1 = []
    taxa = []
    
    for rk in grid_confusion.ranks:
        rk_acc, rk_prc, rk_rec, rk_f1 = get_rk_metrics(grid_confusion[rk], total_sequences)
        acc.append(rk_acc)
        prc.append(rk_prc)
        rec.append(rk_rec)
        f1.append(rk_f1)
        taxa.append(grid_confusion.taxa[rk])
    
    acc = np.concatenate(acc, axis=3)
    prc = np.concatenate(prc, axis=3)
    rec = np.concatenate(rec, axis=3)
    f1 = np.concatenate(f1, axis=3)
    
    ranks = np.concatenate([np.full(len(tx), rk) for tx, rk in zip(taxa, grid_confusion.ranks)])
    taxa = np.concatenate(taxa)
    
    acc_grid = GridMetricsInd(grid_confusion.n_range, grid_confusion.k_range, acc, taxa, ranks)
    prc_grid = GridMetricsInd(grid_confusion.n_range, grid_confusion.k_range, prc, taxa, ranks)
    rec_grid = GridMetricsInd(grid_confusion.n_range, grid_confusion.k_range, rec, taxa, ranks)
    f1_grid = GridMetricsInd(grid_confusion.n_range, grid_confusion.k_range, f1, taxa, ranks)
    
    result = GridMetrics(acc_grid, prc_grid, rec_grid, f1_grid)
    return result

class GridMetricsInd:
    def __init__(self, n_range, k_range, grid, taxa, ranks):
        self.n_range = {n:idx for idx, n in enumerate(n_range)}
        self.k_range = {k:idx for idx, k in enumerate(k_range)}
        self.mth_range = {'unweighted':0, 'u':0, 'wknn':1, 'w':1, 'dwknn':2, 'd':2}
        self.grid = grid
        self.taxa = taxa
        self.ranks = ranks
        
    def get_best(self):
        # combine first 3 dimensions (n, k and method)
        grid_reshaped = self.grid.reshape(-1, self.grid.shape[3])

        # find maximum values along the combined dimension
        max_vals = np.max(grid_reshaped, axis=0)

        # find positions of maximum values in the combined dimension
        max_indices_flat = np.argmax(grid_reshaped, axis=0)

        # convert flat indices to original 3d indices
        max_indices = np.unravel_index(max_indices_flat, self.grid.shape[:3])
        
        index = pd.MultiIndex.from_arrays([self.ranks, self.taxa])
        self.best = pd.DataFrame({'Score':pd.Series(max_vals, index=index),
                                  'n':pd.Series(np.array(list(self.n_range.keys()))[max_indices[0]], index=index),
                                  'k':pd.Series(np.array(list(self.k_range.keys()))[max_indices[1]], index=index),
                                  'Method':pd.Series(np.array(['u', 'w', 'd'])[max_indices[2]], index=index)}).drop(index=0, level=1)
    
    def get_rank_best(self, rank=None):
        if rank is None:
            return self.best.groupby(['n', 'k', 'Method']).agg(lambda x : x.shape[0]).sort_values('Score', ascending=False)
        return self.best.loc[rank].groupby(['n', 'k', 'Method']).agg(lambda x : x.shape[0]).sort_values('Score', ascending=False)
    
class GridMetrics:
    def __init__(self, acc_grid, prc_grid, rec_grid, f1_grid):
        acc_grid.get_best()
        prc_grid.get_best()
        rec_grid.get_best()
        f1_grid.get_best()
        self.accuracy = acc_grid
        self.precision = prc_grid
        self.recall = rec_grid
        self.f1 = f1_grid
    

mets = get_metrics(conf, agent.data.R.lineage_collapsed.shape[0])
