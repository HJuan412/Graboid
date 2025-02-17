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
from Graboid.classification import cls_classify
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
# #%% orbital definitions
# sorted_distances = np.sort(dists, axis=2)[:,:,1:]
# sorted_distances_idxs = np.argsort(dists, axis=2)[:, :, 1:]


# #%% classification functions

# # calculate weights
# def build_weights_mat(weights, sizes):
#     # builds a 2d array of shape [ #queries, max number of neighbours among all queries]
#     # each row contains the weights of the neighbours of a given query
#     weight_lists = []
#     for seq_weights, seq_sizes in zip(weights, sizes):
#         weight_lists.append(np.concatenate([np.full(size, weight) for size, weight in zip(seq_sizes, seq_weights)]))
    
#     neighs = [len(seq) for seq in weight_lists]
#     n_cols = np.max(neighs)
#     weights_mat = np.zeros((len(weights), n_cols))
#     for idx, (seq, nghs) in enumerate(zip(weight_lists, neighs)):
#         weights_mat[idx, :nghs] = seq
#     return weights_mat

# def get_orbital_weights(orbitals, orbital_sizes, k_range, weight_func):
#     k_weights = [weight_func(orbitals[:,:k]) for k in k_range]
#     k_sizes = [orbital_sizes[:,:k] for k in k_range]
#     weights = [build_weights_mat(w, s) for w, s in zip(k_weights, k_sizes)]
#     return weights

# # build support tables
# def replace_vals(matrix):
#     vals = np.unique(matrix)
#     new_mat = np.zeros(matrix.shape, dtype=np.int32)
#     for idx, v in enumerate(vals):
#         new_mat[matrix == v] = idx
#     return new_mat

# def get_supports_mat(weights, sorted_distances_idxs):
#     clipped_idxs = sorted_distances_idxs[:, :weights[-1].shape[1]]
#     neigh_idxs = np.unique(clipped_idxs)
#     supports_mat = np.zeros((len(weights), sorted_distances_idxs.shape[0], len(neigh_idxs)), dtype=np.float32)
    
#     replaced_idxs = replace_vals(clipped_idxs)
    
#     for k, k_weights in enumerate(weights):
#         k_dist_idxs = replaced_idxs[:, :k_weights.shape[1]]
#         for idx, (w, d) in enumerate(zip(k_weights, k_dist_idxs)):
#             supports_mat[k, idx, d] = w
#     return supports_mat, neigh_idxs

# # calculate taxon supports
# def get_tax_support(lineage, supports, neigh_idxs):
#     clipped_lineage = lineage.iloc[neigh_idxs].copy()
#     clipped_lineage['neigh_idxs'] = np.arange(clipped_lineage.shape[0])
    
#     taxa_supports = []
#     taxa_idxs = []
#     for rk in clipped_lineage.drop(columns='neigh_idxs').columns:
#         rk_taxa = np.unique(clipped_lineage[rk])
#         rk_lineage = clipped_lineage.set_index(rk)['neigh_idxs']
#         rk_supp = np.zeros((supports.shape[0], supports.shape[1], len(rk_taxa)))
        
#         for tax_idx, tax in enumerate(rk_taxa):
#             tax_neighs = rk_lineage.loc[[tax]].values
#             rk_supp[:,:, tax_idx] = supports[:,:,tax_neighs].sum(axis=2)
#         taxa_supports.append(rk_supp)
#         taxa_idxs.append(rk_taxa)
#     return taxa_supports, taxa_idxs

# def norm_supports(tax_suports):
#     normalized = []
#     for rk in tax_suports:
#         # softmax supports
#         exp_supports = np.exp(rk)
#         exp_sum = exp_supports.sum(axis=2)
#         exp_sum = exp_sum[:,:, np.newaxis]
#         normalized.append(exp_supports / exp_sum)
#     return normalized


# # classify
# def get_classification(normalized_support, tax_ids):
#     classif_support = np.array([np.max(rk, axis=2) for rk in normalized_support]).transpose(1,2,0)
#     best_pos = np.array([np.argmax(rk, axis=2) for rk in normalized_support]).transpose(1,2,0)
    
#     classif = np.array([tax_ids[idx][best_pos[:,:,idx]] for idx in range(best_pos.shape[2])]).transpose(1,2,0)
    
#     return classif, classif_support

# def n_classify(sorted_dists, sorted_indexes, k_range, lineage):
    
#     # get orbitals + orbital sizes
#     orbitals, orbital_sizes = cal_main.find_orbitals(sorted_dists, k_range.max())
    
#     # calculate orbital weights using the three methods
#     u_weights = get_orbital_weights(orbitals, orbital_sizes, k_range, cls_classify.unweighted)
#     w_weights = get_orbital_weights(orbitals, orbital_sizes, k_range, cls_classify.wknn)
#     d_weights = get_orbital_weights(orbitals, orbital_sizes, k_range, cls_classify.dwknn)
    
#     # build neighbour support matrixes (neigh indexes are the same for all)
#     u_supports, neigh_idxs = get_supports_mat(u_weights, sorted_indexes)
#     w_supports, neigh_idxs = get_supports_mat(w_weights, sorted_indexes)
#     d_supports, neigh_idxs = get_supports_mat(d_weights, sorted_indexes)
    
#     # calcualte taxon supports
#     u_tax_supports, u_tax_ids = get_tax_support(lineage, u_supports, neigh_idxs)
#     w_tax_supports, w_tax_ids = get_tax_support(lineage, w_supports, neigh_idxs)
#     d_tax_supports, d_tax_ids = get_tax_support(lineage, d_supports, neigh_idxs)
#     # normalize supports
#     u_norm = norm_supports(u_tax_supports)
#     w_norm = norm_supports(w_tax_supports)
#     d_norm = norm_supports(d_tax_supports)
    
#     # get classifications (+ support of winner taxon)
#     u_classif, u_classif_support = get_classification(u_norm, u_tax_ids)
#     w_classif, w_classif_support = get_classification(w_norm, w_tax_ids)
#     d_classif, d_classif_support = get_classification(d_norm, d_tax_ids)
    
#     classifications = np.array([u_classif, w_classif, d_classif]).transpose(1,0,2,3)
#     classification_supports = np.array([u_classif_support, w_classif_support, d_classif_support]).transpose(1,0,2,3)
    
#     return classifications, classification_supports

# k_range=np.arange(1,5)
# result, result_supps = n_classify(sorted_distances[0], sorted_distances_idxs[0], k_range, agent.data.R.lineage_collapsed)

#%%
import numba as nb

@nb.njit
def build_confusion_single(res, lin, taxa):
    confusion = np.zeros((res.shape[0], res.shape[1], len(taxa), len(taxa)), dtype=np.int32)
    for idx0, tax0 in enumerate(taxa):
        for idx1, tax1 in enumerate(taxa):
            confusion[:,:,idx0, idx1] = np.sum((lin == tax0) & (res == tax1), axis=2)
    return confusion

def build_rank_confusion(rk_results, rk_lineage):
    reshaped_lin = np.tile(rk_lineage, (rk_results.shape[0], rk_results.shape[1], 1))
    uniq_tax = np.unique(np.append(rk_lineage.values, 0))
    confusion = build_confusion_single(rk_results, reshaped_lin, uniq_tax)
    return confusion, uniq_tax

def build_n_confusion(results, lineage):
    n_confusion = []
    n_taxa = []
    for rk_idx, rk in enumerate(lineage.columns):
        rk_results = results[:,:,:,rk_idx]
        rk_lineage = lineage.iloc[:,rk_idx]
        rk_confusion, rk_taxa = build_rank_confusion(rk_results, rk_lineage)
        n_confusion.append(rk_confusion)
        n_taxa.append(rk_taxa)
    return n_confusion, n_taxa

def build_confusion(results_grid, lineage):
    confusion = []
    taxa = []
    for n_results in results_grid.grid:
        n_confusion, taxa = build_n_confusion(n_results, lineage)
        confusion.append(n_confusion)
    n_ranks = len(taxa)
    confusion2 = []
    for rk in range(n_ranks):
        confusion2.append(np.array([n_conf[rk] for n_conf in confusion]))
    result = GridConfusion(results_grid.n_range, results_grid.k_range, results_grid.mth_range, confusion2, taxa, results_grid.ranks, lineage.shape[0])
    return result
    
class GridConfusion:
    def __init__(self, n_range, k_range, mth_range, confusion, taxa, ranks, n_seqs):
        self.n_range = n_range
        self.k_range = k_range
        self.mth_range = mth_range
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

#b.groupby(['n', 'k', 'Method']).agg(lambda x : x.shape[0])