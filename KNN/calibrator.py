#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 09:27:49 2025

@author: hernan
"""
#%% libraries
import concurrent.futures
import functools
import numba as nb
import numpy as np
import pandas as pd
import warnings

from Graboid.classification import cls_distance, cls_classify
from Graboid.preprocess import feature_selection

#%%
warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')
#%% classes
class GridResult:
    def __init__(self, n_range, k_range, grid, ranks, sites=None):
        self.n_range = {n:idx for idx, n in enumerate(n_range)}
        self.k_range = {k:idx for idx, k in enumerate(k_range)}
        self.mth_range = {'unweighted':0, 'u':0, 'wknn':1, 'w':1, 'dwknn':2, 'd':2}
        self.sites = sites
        self.grid = grid
        self.ranks = ranks
    
    def cell(self, n, k, method):
        n_idx = self.n_range[n]
        k_idx = self.k_range[k]
        mth_idx = self.mth_range[method]
        result = pd.DataFrame(self.grid[n_idx, k_idx, mth_idx], columns=self.ranks)
        return result
    
    def icell(self, n, k, method):
        result = pd.DataFrame(self.grid[n, k, method], columns=self.ranks)
        return result
    
    def n_sites(self, n):
        try:
            n_idx = self.n_range[n]
            return self.sites[n_idx]
        except:
            pass

class GridConfusion:
    def __init__(self, n_range, k_range, confusion, taxa, tax_reprs, ranks, n_seqs):
        self.n_range = n_range
        self.k_range = k_range
        self.mth_range = {'unweighted':0, 'u':0, 'wknn':1, 'w':1, 'dwknn':2, 'd':2}
        self.confusion = {rk:rk_confusion for rk, rk_confusion in zip(ranks, confusion)}
        self.taxa = {rk:rk_taxa for rk, rk_taxa in zip(ranks, taxa)}
        self.tax_reprs = tax_reprs
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
    def __init__(self, acc_grid, prc_grid, rec_grid, f1_grid, tax_reprs):
        acc_grid.get_best()
        prc_grid.get_best()
        rec_grid.get_best()
        f1_grid.get_best()
        self.accuracy = acc_grid
        self.precision = prc_grid
        self.recall = rec_grid
        self.f1 = f1_grid
        self.tax_reprs = tax_reprs
        
    @property
    def taxa(self):
        return self.f1.taxa
    @property
    def ranks(self):
        return self.f1.ranks
    @property
    def n_range(self):
        return self.f1.n_range
    @property
    def k_range(self):
        return self.f1.k_range

class GridFinal:
    def __init__(self, windows, grids, n_range, k_range):
        self.windows = windows
        self.grids = grids
        self.ranks = grids[0].ranks
        self.taxa = get_taxa_matrices(grids)
        self.n_vals = get_n_matrix(grids, n_range)
        self.n_range = n_range
        self.k_range = k_range

    def get_best(self, metric):
        score = pd.DataFrame(.0, index=self.taxa.index, columns=np.arange(len(self.windows)))
        n = pd.DataFrame(0, index=self.taxa.index, columns=np.arange(len(self.windows)))
        k = pd.DataFrame(0, index=self.taxa.index, columns=np.arange(len(self.windows)))
        meth = pd.DataFrame(0, index=self.taxa.index, columns=np.arange(len(self.windows)))
        
        for idx, win_grid in enumerate(self.grids):
            grid_met = getattr(win_grid, metric)
            score[idx] = grid_met.best.Score
            n[idx] = grid_met.best.n
            k[idx] = grid_met.best.k
            meth[idx] = grid_met.best.Method
        self.score = score
        self.n = n.fillna(0).astype(int)
        self.k = k.fillna(0).astype(int)
        self.Method = meth
        
#%% preparation steps, site selection and distance calculation
def get_sites(gain, n_range):
    sorted_gain = np.flip(np.argsort(gain, axis=1), axis=1)
    # get sites for each n level
    sites = [np.unique(sorted_gain[:, :n]) for n in n_range]
    # clear contained sites
    for i in np.flip(np.arange(1, len(n_range))):
        sites[i] = np.setdiff1d(sites[i], sites[i-1])
    
    # filter out values of n with no unique sites
    keep = [idx for idx, n_sites in enumerate(sites) if len(n_sites) > 0]
    n_range = n_range[keep]
    sites = [sites[idx] for idx in keep]
    return sites, n_range

def get_distances(matrix, sites, cost_mat):
    """
    Calcuate paired distances for every sub window, for every level of n

    Parameters
    ----------
    matrix : numpy.array
        Alignment array.
    sites : list
        List of numpy arrays containing the unique sites found at each level
        of n.
    cost_mat : numpy.array
        2d matrix detailing the distance cost of every substitution.

    Returns
    -------
    distances : numpy.array
        3d array of shape (# levels of n, # seqs in window, # seqs in window),
        diagonal elements are -1.

    """
    
    distances = []
    # get distances for each value of n, use cumsum to include the distance of all previous levels of n
    for n_sites in sites:
        n_cols = matrix[:, n_sites]
        distances.append(cls_distance.get_distances(n_cols, n_cols, cost_mat))
    distances = np.cumsum(distances, 0) # some elements in the diagonal have distance over 0 because of unknown sites
    distances[:, np.arange(distances.shape[1]), np.arange(distances.shape[2])] = -1 # diagonal elements to -1 ensures distance vs self is always first place when sorting
    return distances

#%% grid classification
def classify(distances, lineage, n_range, k_range, sites, criterion='orbit', threads=1):
    """
    Generates KNN classifications for the provided distance arrays.

    Parameters
    ----------
    distances : numpy.array
         3d array of shape (# levels of n, # seqs in window, # seqs in window),
         diagonal elements are -1.
    lineage : pandas.DataFrame
         Dataframe containing the taxonomic IDs for each reference sequence.
         Each column corresponds to the sequence's classification at a given
         rank.
    n_range : TYPE
        DESCRIPTION.
    k_range : numpy.array
         Range of values of K to be used in the classification.
    sites : TYPE
        DESCRIPTION.
    criterion : string, optional
         Neighbour selection criterion, possible values are "orbit"/"neighbour". The default is 'orbit'.
    threads : int, optional
         Number of parallel tasks. The default is 1.

    Returns
    -------
    grid_classifications : GridResult
        Contains classifications for each query/rank and each combination of n/k/method.
    grid_supports : GridResult
        Contains classification supports for each query/rank and each combination of n/k/method.

    """
    
    # sort distances (remove first column from each layer (it's always distance to self))
    sorted_distances = np.sort(distances, axis=2)[:, :, 1:]
    sorted_distances_idxs = np.argsort(distances, axis=2)[:,:, 1:]
    
    # get classification (& support) for each n layer
    classifications = []
    classification_supports = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(n_classify, n_dists, n_sort_idxs, k_range, lineage) for n_dists, n_sort_idxs in zip(sorted_distances, sorted_distances_idxs)]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            classifications.append(res[0])
            classification_supports.append(res[1])
    
    classifications = np.array(classifications)
    classification_supports = np.array(classification_supports)
    
    grid_classifications = GridResult(n_range, k_range, classifications, lineage.columns.values, sites)
    grid_supports = GridResult(n_range, k_range, classification_supports, lineage.columns.values)
    return grid_classifications, grid_supports

def n_classify(sorted_dists, sorted_indexes, k_range, lineage):
    
    # 0. get orbitals + orbital sizes
    orbitals, orbital_sizes = find_orbitals(sorted_dists, k_range.max())
    
    # 1. calculate orbital weights using the three methods
    u_weights = get_orbital_weights(orbitals, orbital_sizes, k_range, cls_classify.unweighted)
    w_weights = get_orbital_weights(orbitals, orbital_sizes, k_range, cls_classify.wknn)
    d_weights = get_orbital_weights(orbitals, orbital_sizes, k_range, cls_classify.dwknn)
    
    # 2. build neighbour support matrixes (neigh indexes are the same for all)
    u_supports, neigh_idxs = get_supports_mat(u_weights, sorted_indexes)
    w_supports, neigh_idxs = get_supports_mat(w_weights, sorted_indexes)
    d_supports, neigh_idxs = get_supports_mat(d_weights, sorted_indexes)
    
    # 3. calcualte taxon supports & normalize
    u_tax_supports, u_tax_ids = get_tax_support(lineage, u_supports, neigh_idxs)
    w_tax_supports, w_tax_ids = get_tax_support(lineage, w_supports, neigh_idxs)
    d_tax_supports, d_tax_ids = get_tax_support(lineage, d_supports, neigh_idxs)
    
    u_norm = norm_supports(u_tax_supports)
    w_norm = norm_supports(w_tax_supports)
    d_norm = norm_supports(d_tax_supports)
    
    # 4. get classifications (+ support of winner taxon)
    u_classif, u_classif_support = get_classification(u_norm, u_tax_ids)
    w_classif, w_classif_support = get_classification(w_norm, w_tax_ids)
    d_classif, d_classif_support = get_classification(d_norm, d_tax_ids)
    
    classifications = np.array([u_classif, w_classif, d_classif]).transpose(1,0,2,3)
    classification_supports = np.array([u_classif_support, w_classif_support, d_classif_support]).transpose(1,0,2,3)
    
    return classifications, classification_supports

# 0. find orbitals ###############################################################
@nb.njit
def find_orbitals(sorted_distances, k=1):
    """
    Builds two 2d arrays containing the radii and population size of each
    orbital (up to the kth) of each query sequence.

    Parameters
    ----------
    sorted_distances : numpy.array
        2d array containig sorted distance values.
    k : int
        Number of orbitals to select for each query.

    Returns
    -------
    orbital_radii : numpy.array
        2d array of shape (#queries, k) containing the radii of the k first
        orbitals of each query.
    orbital_sizes : numpy.array
        2d array of shape (#queries, k) containing the population sizes of the
        k first orbitals of each query.

    """
    orbital_radii = np.full((sorted_distances.shape[0], k), -1, dtype=np.float32)
    orbital_sizes = np.full((sorted_distances.shape[0], k), -1, dtype=np.int32)
    
    for idx, row in enumerate(sorted_distances):
        radii = np.unique(row).astype(np.float32)
        # pad radii if needed
        if len(radii) < k:
            radii0 = np.full(k, np.inf, dtype=np.float32)
            radii0[:len(radii)] = radii
            radii = radii0
        orbital_radii[idx] = radii[:k]
        for r_idx, r in enumerate(radii[:k]):
            orbital_sizes[idx, r_idx] = np.sum(row == r)
    return orbital_radii, orbital_sizes

# 1.0 calculate orbital weights ###################################################

def build_weights_mat(weights, sizes):
    """
    Builds a matrix containing the weights of all neighbours for all queries
    (for a single value of k).
    Returns a 2d array of shape [ #queries, max # of neighbours among all queries]
    each row contains the weights of the neighbours of a given query.

    Parameters
    ----------
    weights : numpy.array
        2d array containing the calculated weights for each orbital (up to the
        k-th).
    sizes : numpy.array
        2d array of shape (#queries, k) containing the population sizes of the
        k first orbitals of each query.

    Returns
    -------
    weights_mat : numpy.array
        2d array containing the calculated weights for each neighbour of each
        query (up to the k-th orbital).

    """
    
    weight_lists = []
    for seq_weights, seq_sizes in zip(weights, sizes):
        weight_lists.append(np.concatenate([np.full(size, weight) for size, weight in zip(seq_sizes, seq_weights)]))
    
    # count the number of neighbours for each query (depends on the population sizes of their orbitals)
    neighs = [len(seq) for seq in weight_lists]
    
    # initialize weights matrix (must acomodate up to the maximum number of neighbours)
    n_cols = np.max(neighs)
    weights_mat = np.zeros((len(weights), n_cols))
    # populate weights matrix
    for idx, (seq, nghs) in enumerate(zip(weight_lists, neighs)):
        weights_mat[idx, :nghs] = seq
    return weights_mat

def get_orbital_weights(orbital_radii, orbital_sizes, k_range, weight_func):
    """
    Builds a list of 2d arrays containing the weights for each neighbour of
    each query each 2d array corresponds to a value of k and shape
    [ #queries, max # of neighbours among all queries (for that value of k)]
    this is performed for the orbitals of a single value of n.

    Parameters
    ----------
    orbital_radii : numpy.array
        2d array containing the radii of the orbitals of each query.
    orbital_sizes : numpy.array
        2d array containing the population sizes of the orbitals of each query.
    k_range : numpy.array
        Range of values of k.
    weight_func : func
        Weighting function.

    Returns
    -------
    weights : list
        List of 2d arrays (one per value of k) containing calculated neighbour
        weights. Each array has shape [ #queires, max #neighbours for k ]

    """
    
    # calculate orbital weights for each value of k
    k_weights = [weight_func(orbital_radii[:,:k]) for k in k_range]
    # build weights matrix
    k_sizes = [orbital_sizes[:,:k] for k in k_range]
    weights = [build_weights_mat(w, s) for w, s in zip(k_weights, k_sizes)]
    return weights

# 2. build neighbour supports matrix #############################################

def replace_vals(matrix):
    # replace values in matrix for their sorted equivalent (eg: matrix with values 6,15,32, values are replaced for 0,1,2)
    # used to build the supports matrix
    vals = np.unique(matrix)
    new_mat = np.zeros(matrix.shape, dtype=np.int32)
    for idx, v in enumerate(vals):
        new_mat[matrix == v] = idx
    return new_mat

def get_supports_mat(weights, sorted_distances_idxs):
    """
    Builds set of arrays assigning the weight of each reference sequence to
    each query sequence depending on their distance to said query.

    Parameters
    ----------
    weights : list
        List of arrays containing neighbour weights for each value of k.
    sorted_distances_idxs : numpy.array
        2d array containing the sorted by distance indexes of neighbouring
        sequences to each query sequence.

    Returns
    -------
    supports_mat : numpy.array
        3d array of shape [ range_k, #queries, max neighbours ]
            # each layer of the array contains the weights of all neighbours (cols) to all queries (rows) for a given value of k
    neigh_idxs : numpy.array
        Array containing the neighbour column indexes (inidcate their relative position to their respective queries).

    """
        
    # get involved neighbours, clip sorted_disstances_idxs at the size of the largest weight matrix
    clipped_idxs = sorted_distances_idxs[:, :weights[-1].shape[1]]
    # get unique neighbout indexes
    neigh_idxs = np.unique(clipped_idxs)
    
    # replace indxes by their sorted equivalent (this is done to place weight values in the corresponding neighbour column)
    replaced_idxs = replace_vals(clipped_idxs)
    # preinitialize supports matrix
    supports_mat = np.zeros((len(weights), sorted_distances_idxs.shape[0], len(neigh_idxs)), dtype=np.float32)
    
    # populate supports matrix
    for k, k_weights in enumerate(weights):
        # get column indexes for the neighbours at the current k
        k_dist_idxs = replaced_idxs[:, :k_weights.shape[1]]
        # place weight values for the neighbours of each query
        for idx, (w, d) in enumerate(zip(k_weights, k_dist_idxs)):
            supports_mat[k, idx, d] = w
    return supports_mat, neigh_idxs

# 3. get taxa supports ###########################################################

def get_tax_support(lineage, supports, neigh_idxs):
    """
    Calculates the accumulated support (sum of weights) for each taxon.
    
    Parameters
    ----------
    lineage : pandas.DataFrame
        Dataframe contaiing the taxonomic information for each instance for a
        number of taxonomic ranks.
    supports : numpy.array
        3d array containing neighbour support for each value of k.
    neigh_idxs : numpy.array
        Array containing the neighbour indexes of each column (axis 2) in the
        supports array.

    Returns
    -------
    taxa_supports : list
        List of 3d arrays of shape [ #values of k, #queries, #taxa in rank], one element per taxonomic ranks.
    taxa_idxs : list
        List of arrays indicating the taxonomic id corresponding to each column of the taxa_supports arrays (same number of elements).

    """
    
    # extract lineages of neighbours
    clipped_lineage = lineage.iloc[neigh_idxs].copy()
    clipped_lineage['neigh_idxs'] = np.arange(clipped_lineage.shape[0])
    
    # preinitialize lists
    taxa_supports = []
    taxa_idxs = []
    
    # calculate supports for each rank
    for rk in clipped_lineage.drop(columns='neigh_idxs').columns:
        # get taxa in rank
        rk_taxa = np.unique(clipped_lineage[rk])
        rk_lineage = clipped_lineage.set_index(rk)['neigh_idxs']
        
        # calculate total support for each taxon in rank
        rk_supp = np.zeros((supports.shape[0], supports.shape[1], len(rk_taxa)))
        for tax_idx, tax in enumerate(rk_taxa):
            # get taxon representatives in supports matrix, summ their supports
            tax_neighs = rk_lineage.loc[[tax]].values
            rk_supp[:,:, tax_idx] = supports[:,:,tax_neighs].sum(axis=2)
        taxa_supports.append(rk_supp)
        taxa_idxs.append(rk_taxa)
    return taxa_supports, taxa_idxs

def norm_supports(tax_suports):
    # softmax normalize supports of each query
    normalized = []
    for rk in tax_suports:
        # softmax supports
        exp_supports = np.exp(rk)
        exp_sum = exp_supports.sum(axis=2)
        exp_sum = exp_sum[:,:, np.newaxis]
        normalized.append(exp_supports / exp_sum)
    return normalized

# 4. final classification ########################################################
def get_classification(normalized_support, tax_ids):
    """
    Assigns taxonomic classifications

    Parameters
    ----------
    normalized_support : list
        List of 3d arrays of shape [range of k, #queries, #taxa in rank]. One
        array per rank.
    tax_ids : list
        List of arrays containing the taxonomic ids for each column in each of
        the 3d arrays contained in normalized support.

    Returns
    -------
    classif : numpy.array
        3d array of shape [ range of k, #queries, #ranks ] containing the
        assigned classification for each query in each rank for each value of k.
    classif_support : numpy.array
        3d array of shape [ range of k, #queries, #ranks ] containing the
        support for each assigned classification.

    """
    # identify the taxon with the most support for each query in each rank
        # max function along axis 2 returns 2d array of shape [values of k, #queries]
        # one array generated for each rank, merged into a single 3d array of shape [#ranks, values of k, #queries]
        # array transposed into shape [values of k, #queries, #ranks]
    # same process of argmax to get classification
    classif_support = np.array([np.max(rk, axis=2) for rk in normalized_support]).transpose(1,2,0)
    best_pos = np.array([np.argmax(rk, axis=2) for rk in normalized_support]).transpose(1,2,0)
    
    classif = np.array([tax_ids[idx][best_pos[:,:,idx]] for idx in range(best_pos.shape[2])]).transpose(1,2,0)
    
    return classif, classif_support
#%% window definition
def set_sliding_windows(size, step, max_pos):
    """
    Generates coordinates for the sliding windows

    Parameters
    ----------
    size : int
        Window size.
    step : int
        Window displacement.
    max_pos : int
        Length of the marker sequence.

    Raises
    ------
    Exception
        If either the window size exceeds the length of the marker sequence or
        the displacement leaves gaps between windows.

    Returns
    -------
    windows : numpy.array
        2d array of shape [ # windows, 2 ], containing the start and end
        coordinates of each window.

    """
    if size >= max_pos:
        raise Exception(f'Given window size: {size} is equal or greater than the total length of the alignment {max_pos}, please use a smaller window size.')
    if step > size:
        raise Exception(f'Window displacement rate ({step}) must be lower or equal to the window size ({size})')
    
    # adjust window size to get uniform distribution (avoid having to use a "tail" window)
    last_position = max_pos - size
    n_windows = int(np.ceil(last_position / step))
    w_start = np.linspace(0, last_position, n_windows, dtype=int)
    windows = np.array([w_start, w_start + size]).T
    return windows

def set_custom_windows(coords, max_pos):
    # ensure coordinates are given as a n x 2 array
    if len(coords.shape) != 2 or coords.shape[1] != 2:
        raise Exception('Coordinates must be given as a 2d array of n rows and 2 columns!')
    
    # ensure all coordinate pairs are valid
    invalid = coords[:, 0] >= coords[:, 1]
    if invalid.sum() > 0:
        raise Exception(f'At least one pair of coordinates is invalid: {[list(i) for i in coords[invalid]]}')
    
    # ensure all cordinates are within bounds
    out_of_bounds = ((coords < 0) | (coords >= max_pos)).any(axis=1)
    if out_of_bounds.sum() > 0:
        raise Exception(f'At least one pair of coordinates is out of bounds [0 : {max_pos}]: {[list(i) for i in coords[out_of_bounds]]}')
    
    return coords

#%% evaluation
# 0. Confusion ###################################################################
def build_confusion(results_grid, lineage, threads=1):
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
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(build_n_confusion, n_results, lineage) for n_results in results_grid.grid]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            confusion.append(res[0])
            taxa = res[1]
    n_ranks = len(taxa)
    
    # merge n confusion matrices for each rank, building the 5d arrays of shape [range_n, range_k, #methods, #taxa, #taxa]
    confusion2 = []
    for rk in range(n_ranks):
        confusion2.append(np.array([n_conf[rk] for n_conf in confusion]))
    
    # get representatives of each taxon for later use
    tax_reprs = get_tax_reprs(lineage)
    # build Grid class
    result = GridConfusion(results_grid.n_range, results_grid.k_range, confusion2, taxa, tax_reprs, results_grid.ranks, lineage.shape[0])
    return result

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

def get_tax_reprs(lineage):
    reprs = {}
    for rk, row in lineage.T.iterrows():
        reprs[rk] = row.value_counts().sort_index()
    reprs = pd.concat(reprs)
    return reprs

# 1. Metrics #####################################################################
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
    
    result = GridMetrics(acc_grid, prc_grid, rec_grid, f1_grid, grid_confusion.tax_reprs)
    return result

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
#%% Grid merging
def get_taxa_matrices(grids):
    index = functools.reduce(lambda x, y : x.union(y), [g.tax_reprs.index for g in grids])
    tax_repr_mat = pd.DataFrame(0, index=index, columns=np.arange(len(grids)))
    for idx, g in enumerate(grids):
        tax_repr_mat[idx] = g.tax_reprs
    tax_repr_mat = tax_repr_mat.fillna(0).astype(int)
    return tax_repr_mat

def get_n_matrix(grids, n_range):
    n_matrix = pd.DataFrame(False, index=np.arange(len(grids)), columns=n_range)
    for idx, g in enumerate(grids):
        n_matrix.loc[idx, g.n_range.keys()] = True
    return n_matrix

#%% calibration funcs
def calibrate_sliding(data, w_size, w_step, max_n, step_n, max_k, step_k, cost_matrix, row_thresh=.2, col_thresh=.1, min_seqs=50, rank='genus', min_n=5, min_k=3, criterion='orbit', collapse_hm=True, threads=1):
    # get window coordinates
    windows = set_sliding_windows(w_size, w_step, data.shape[1])
    window_grids = []
    
    # prepare n, k ranges
    n_range = np.arange(min_n, max_n + 1, step_n)
    k_range = np.arange(min_k, max_k + 1, step_k)
    
    # run calibration for each window
    missed_windows = []
    for idx, win in enumerate(windows):
        print(f'Calibrating window {win} ({idx+1} of {len(windows)})')
        try:
            data.select_region(win[0], win[1])
            data.collapse(row_thresh)
        except Exception as excp:
            print(f'Window {win} ommited ({excp})')
            missed_windows.append(idx)
            continue
        gain, counts = feature_selection.get_information_gain(data.collapsed, data.lineage_collapsed)
        window_grid = grid_search(data.collapsed, data.lineage_collapsed, gain, cost_matrix, n_range, k_range)
        window_grids.append(window_grid)
    windows = np.delete(windows, missed_windows, axis=0)
    # result = GridFinal(windows, window_grids, n_range, k_range)
    # return result
    return windows, window_grids

def calibrate_custom(data, w_coords, max_n, step_n, max_k, step_k, cost_matrix, row_thresh=.2, col_thresh=.1, min_seqs=50, rank='genus', min_n=5, min_k=3, criterion='orbit', collapse_hm=True, threads=1):
    # get window coordinates
    windows = set_custom_windows(w_coords, data.shape[1])
    window_grids = []
    
    # prepare n, k ranges
    n_range = np.arange(min_n, max_n + 1, step_n)
    k_range = np.arange(min_k, max_k + 1, step_k)
    
    # run calibration for each window
    for win in windows:
        data.select_region(win[0], win[1])
        data.collapse(row_thresh)
        gain, counts = feature_selection.get_information_gain(data.collapsed, data.lineage_collapsed)
        window_grid = grid_search(data.collapsed, data.lineage_collapsed, gain, cost_matrix, n_range, k_range)
        window_grids.append(window_grid)
    
    result = GridFinal(windows, window_grids, n_range, k_range)
    return result

def grid_search(matrix, lineage, gain, cost_mat, n_range, k_range, row_thresh=.2, col_thresh=.1, min_seqs=50, rank='genus', criterion='orbit', collapse_hm=True, threads=1):    
    # get sites arrays
    sites, n_range = get_sites(gain, n_range)
    # calculate distances
    distances = get_distances(matrix, sites, cost_mat) # 3d array of shape (n, #seqs, #seqs), contains paired distances for every value of n
    
    # classify instances
    classifications, supports = classify(distances, lineage, n_range, k_range, sites, criterion, threads)
    
    # build confusion matrix
    confusion = build_confusion(classifications, lineage, threads)
    
    # get grid metrics
    metrics = get_metrics(confusion, matrix.shape[0])
    return metrics