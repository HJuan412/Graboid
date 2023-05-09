#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:36:58 2023

@author: nano

Classify test instances during calibration
"""

#%% libraries
import concurrent.futures
import numpy as np

#%% functions
def compress_dists(distance_array):
    # for every (sorted) distance array, return unique values with counts (counts are used to group neighbours)
    return [np.stack(np.unique(dists, return_counts=True)) for dists in distance_array]

def wknn(distance_array):
    d1 = distance_array[:, [0]]
    dk = distance_array[:, [-1]]
    return (dk - distance_array) / (dk - d1)

def dwknn(distance_array, weighted):
    d1 = distance_array[:, [0]]
    dk = distance_array[:, [-1]]
    penal = (dk + d1) / (dk + distance_array)
    return weighted * penal

class Neighbours:
    # this class is used to contain and handle the data after distance collapsing
    def __init__(self, sorted_neighs, max_k):
        self.max_k = max_k
        # group neighbours by distance to query
        compressed = compress_dists(sorted_neighs[1])
        distances = []
        positions = []
        indexes = []
        for neighs, comp in zip(sorted_neighs[0], compressed):
            neigh_dists = np.full(max_k, max(comp[0]))
            neigh_poss = np.full(max_k, 0, dtype=np.int16)
            _k = min(len(comp[0]), max_k)
            neigh_dists[:_k] = comp[0, :_k]
            neigh_poss[:_k] = comp[1, :_k]
            distances.append(neigh_dists)
            positions.append(neigh_poss)
            indexes.append(neighs[:neigh_poss.sum()].astype(int))
        self.distances = np.array(distances) # contains collapsed distances up to the max_k position
        self.positions = np.array(positions) # contains count of unique distance values (used to select indexes at a given orbit)
        self.indexes = indexes # contains #sequences arrays (of varying lengths) with the indexes of the neighbours that populate the orbits up to the k_max position
    
    def get_weights(self, k_range):
        self.k_range = k_range
        # weight and double weight distances for multiple values of k
        weighted = []
        double_weighted = []
        
        for k in k_range:
            k_weighted = wknn(self.distances[:,:k])
            weighted.append(k_weighted)
            double_weighted.append(dwknn(self.distances[:,:k], k_weighted))
        # both weighted and double weighted are lists containing len(k_range) arrays of weights of increasing size (from min(k_range) to max(k_range) by step_k)
        
        # both weighted and double_weighted are lists of len(k_range) elements corresponding to the weighted distances for the different values of K
        self.weighted = weighted
        self.double_weighted = weighted

# both classify functions return a list of #sequences elements containing #ranks tuples with an unweighted, weighted and double_weighted classification
# tax_tab should be the window's extended taxonomy passed as a numpy array
def orbit_classify(positions, indexes, weights, double_weights, tax_tab):
    # get all neighbours in the first k orbits
    k = weights.shape[1] # infer k from the number of calculated weights
    k_pos = positions[:,:k] # get position locators
    # get the weights for each orbit
    orbit_arrays = [] # use this as base to get the corresponding weight/double_weight for each neighbour, depending on the orbit it occupies
    for poss in k_pos:
        orbit_arrays.append(np.concatenate([[idx]*pos for idx, pos in enumerate(poss)]).astype(int))
    weighted_arrays = [wghts[orbt] for wghts, orbt in zip(weights, orbit_arrays)]
    double_weighted_arrays = [dwghts[orbt] for dwghts, orbt in zip(double_weights, orbit_arrays)]
    # retrieve indexes of neighbours located within the k-th orbit
    k_indexes = [idxs[:len(orbt)] for idxs, orbt in zip(indexes, orbit_arrays)]
    
    classifs = []
    # generate a classification for each sequence
    for idxs, w_arr, dw_arr in zip(k_indexes, weighted_arrays, double_weighted_arrays):
        sub_tax = tax_tab[idxs] # retrieve neighbour indexes
        rk_classifs = []
        # classify for each rank
        for rk in sub_tax.T:
            # count unique taxa in rank, filter out unknown taxa
            taxa, counts = np.unique(rk, return_counts=True)
            counts = counts[~np.isnan(taxa)]
            taxa = taxa[~np.isnan(taxa)]
            if len(taxa) == 0:
                # no valid taxa in rank, assign as unknown and continue
                rk_classifs.append(np.array([-1,-1,-1]))
                continue
            # get the locations of each member for each taxa (use it to retrieve a taxon's weights)
            tax_locs = [rk == tax for tax in taxa]
            
            # for all classifs: if the _classif array has multiple elements, there is a conflict, replace it for [-1] to indicat an unknown
            
            # unweighted classify, majority_vote
            u_classif = taxa[counts == max(counts)]
            if len(u_classif) > 1:
                u_classif = [-1]
            
            # for weighted and double_weighted classification:
                # get the cummulative weights for all taxa
                # select taxon with the highest weight
            
            # weighted classify
            tax_supp_weighted = np.array([w_arr[tax].sum() for tax in tax_locs])
            w_classif = taxa[tax_supp_weighted == max(tax_supp_weighted)]
            if len(w_classif) > 1:
                w_classif = [-1]
            
            # double weighted classify
            tax_supp_double_weighted = [dw_arr[tax].sum() for tax in tax_locs]
            d_classif = taxa[tax_supp_double_weighted == max(tax_supp_double_weighted)]
            if len(d_classif) > 1:
                d_classif = [-1]
            
            rk_classifs.append(np.array([u_classif[0], w_classif[0], d_classif[0]]))
        classifs.append(np.array(rk_classifs))
    return np.array(classifs, dtype=int)

def neigh_classify(positions, indexes, weights, double_weights, tax_tab):
    k = weights.shape[1] # infer k from the number of calculated weights
    # for each sequence, get the orbit containing the k-th neighbour
    selected_orbits = np.argmax(np.cumsum(positions, axis=1) >= k, axis=1) + 1
    selected_positions = [pos[:orbt] for pos, orbt in zip(positions, selected_orbits)]
    orbit_arrays = [] # use this as base to get the corresponding weight/double_weight for each neighbour, depending on the orbit it occupies
    for poss in selected_positions:
        orbit_arrays.append(np.concatenate([[idx]*pos for idx, pos in enumerate(poss)]))
    weighted_arrays = [wghts[orbt] for wghts, orbt in zip(weights, orbit_arrays)]
    double_weighted_arrays = [dwghts[orbt] for dwghts, orbt in zip(double_weights, orbit_arrays)]
    # retrieve indexes of neighbours located within the k-th orbit
    k_indexes = [idxs[:len(orbt)] for idxs, orbt in zip(indexes, orbit_arrays)]
    
    classifs = []
    # generate a classification for each sequence
    for idxs, w_arr, dw_arr in zip(k_indexes, weighted_arrays, double_weighted_arrays):
        sub_tax = tax_tab[idxs] # retrieve neighbour indexes
        rk_classifs = []
        # classify for each rank
        for rk in sub_tax.T:
            # count unique taxa in rank, filter out unknown taxa
            taxa, counts = np.unique(rk, return_counts=True)
            counts = counts[~np.isnan(taxa)]
            taxa = taxa[~np.isnan(taxa)]
            if len(taxa) == 0:
                # no valid taxa in rank, assign as unknown and continue
                rk_classifs.append(np.array([-1,-1,-1]))
                continue
            # get the locations of each member for each taxa (use it to retrieve a taxon's weights)
            tax_locs = [rk == tax for tax in taxa]
            
            # for all classifs: if the _classif array has multiple elements, there is a conflict, replace it for [-1] to indicat an unknown
            
            # unweighted classify, majority_vote
            u_classif = taxa[counts == max(counts)]
            if len(u_classif) > 1:
                u_classif = [-1]
            
            # for weighted and double_weighted classification:
                # get the cummulative weights for all taxa
                # select taxon with the highest weight
            
            # weighted classify
            tax_supp_weighted = np.array([w_arr[tax].sum() for tax in tax_locs])
            w_classif = taxa[tax_supp_weighted == max(tax_supp_weighted)]
            if len(w_classif) > 1:
                w_classif = [-1]
            
            # double weighted classify
            tax_supp_double_weighted = [dw_arr[tax].sum() for tax in tax_locs]
            d_classif = taxa[tax_supp_double_weighted == max(tax_supp_double_weighted)]
            if len(d_classif) > 1:
                d_classif = [-1]
            
            rk_classifs.append(np.array([u_classif[0], w_classif[0], d_classif[0]]))
        classifs.append(np.array(rk_classifs))
    return np.array(classifs, dtype=int)

def generate_classifications(sorted_neighbours, k_max, k_step, k_min, tax_tab, threads, criterion='orbit'):
    # criterion: orbit or neighbours
    # if orbit: get first k orbits, select all sequences from said orbits
    # if neighbours: set cutoff at the orbit that includes the first k neighs
    
    k_range = np.arange(k_min, k_max, k_step)
    # get max neighbours per level
    lvl_neighbours = [Neighbours(sorted_neigh, k_max) for sorted_neigh in sorted_neighbours]
    for lvl in lvl_neighbours:
        lvl.get_weights(k_range)
    
    grid_indexes = [(n_idx, k_idx) for n_idx in np.arange(len(lvl_neighbours)) for k_idx in np.arange(len(k_range))]
    classifs = {nk:None for nk in grid_indexes}
    if criterion == 'orbit':
        classif_func = orbit_classify
    elif criterion == 'neigh':
        classif_func = neigh_classify
    else:
        raise Exception(f'Invalid criterion value: {criterion}, must be "orbit" or "neigh"')
    with concurrent.futures.ProcessPoolExecutor(max_workers = threads) as executor:
        future_classifs = {executor.submit(classif_func,
                                           lvl_neighbours[n_idx].positions,
                                           lvl_neighbours[n_idx].indexes,
                                           lvl_neighbours[n_idx].weighted[k_idx],
                                           lvl_neighbours[n_idx].double_weighted[k_idx],
                                           tax_tab): (n_idx, k_idx) for n_idx, k_idx in grid_indexes}
        for future in concurrent.futures.as_completed(future_classifs):
            cell = future_classifs[future]
            classifs[cell] = future.result()
            # print(f'Done with cell {cell}')
    
    # return dictionary of the form (n_index, k_index):classifications
    return classifs

def classify_windows(win_list, sorted_win_neighbours, tax_ext, min_k, max_k, step_k, criterion='orbit', threads=1, win_idxs=None):
    win_classifs = []
    
    # win_idxs is used to signal when calculations are finished for a given window
    if win_idxs is None:
        win_idxs = np.arange(len(win_list))
    
    for idx, (sorted_neighs, window) in enumerate(zip(sorted_win_neighbours, win_list)):
        window_tax = tax_ext.loc[window.taxonomy].to_numpy()
        win_classifs.append(generate_classifications(sorted_neighs, max_k, step_k, min_k, window_tax, threads = threads, criterion = criterion))
        print(f'Classified window {win_idxs[idx]}')
    
    return win_classifs