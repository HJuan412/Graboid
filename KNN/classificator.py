#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 13:49:37 2025

@author: hernan
"""

#%% libraries
import numpy as np
import pandas as pd

from Graboid.classification import cls_classify, cls_distance
from Graboid.KNN import calibrator
from Graboid.preprocess import feature_selection
#%%
class KNNResult:
    def __init__(self, classif_tab, classif_tab_named, support_tab, ranks, start, end, n, k, method, sites, q_branches):
        self.classification = pd.DataFrame(classif_tab, columns=ranks.split())
        self.classification_named = pd.DataFrame(classif_tab_named, columns=ranks.split())
        self.supports = pd.DataFrame(support_tab, columns=ranks.split())
        self.ranks = ranks.split()
        self.start = start
        self.end = end
        self.n = n
        self.k = k
        self.method = method
        self.sites = sites + start
        self.q_branches = q_branches
    
    def expand(self):
        index = np.sort(np.concatenate(self.q_branches))
        expanded = pd.DataFrame(0, index=index, columns=self.ranks)
        for idx, branch in enumerate(self.q_branches):
            expanded.iloc[branch] = self.classification_named.loc[idx]
        return expanded
    

def get_named_classif(classif_tab, names_tab):
    names = np.array([names_tab.loc[rk_ids].values for rk_ids in classif_tab.T]).T
    return names

def classify(query, reference, cost_mat, n, k, method, criterion='orbit', threads=1):
    # select sites
    gain, counts = feature_selection.get_information_gain(reference.collapsed, reference.lineage_collapsed)
    
    n_range = np.array([n])
    sites, counts = calibrator.get_sites(gain, n_range)
    sites = sites[0]
    # get weighting function
    try:
        weight_func = {'unweighted':cls_classify.unweighted,
                       'wknn':cls_classify.wknn,
                       'dwknn':cls_classify.dwknn}[method]
    except KeyError:
        raise Exception(f'Invalid weighting method: {method}, avaliable methods are \'unweighted\'  \'wknn\'  \'dwknn\'')
    # calculate distances
    query_encoded = cls_distance.one_hot_encode(query.collapsed[:, sites])
    ref_encoded = cls_distance.one_hot_encode(reference.collapsed[:, sites])
    #distances = cls_distance.get_distances(query.collapsed[:, sites], reference.collapsed[:, sites], cost_mat)
    distances = cls_distance.get_distances3(query_encoded, ref_encoded, cost_mat)
    
    # sort distances (remove first column from each layer (it's always distance to self))
    sorted_distances = np.sort(distances, axis=1)[1:]
    sorted_distances_idxs = np.argsort(distances, axis=1)[1:]
    
    # 0. get orbitals + orbital sizes
    orbitals, orbital_sizes = calibrator.find_orbitals(sorted_distances, k)
    
    # 1. calculate orbital weights using the three methods
    weights = calibrator.get_orbital_weights(orbitals, orbital_sizes, [k], weight_func)
    
    # 2. build neighbour support matrixes (neigh indexes are the same for all)
    supports, neigh_idxs = calibrator.get_supports_mat(weights, sorted_distances_idxs)
    
    # 3. calcualte taxon supports & normalize
    tax_supports, tax_ids = calibrator.get_tax_support(reference.lineage_collapsed, supports, neigh_idxs)
    
    support_norm = calibrator.norm_supports(tax_supports)
    
    # 4. get classifications (+ support of winner taxon)
    classif, classif_support = calibrator.get_classification(support_norm, tax_ids)
    
    # this function shares the same body as the n_classify calibrator function, which is designed to work with multiple values of k
    # as a quick fix, the single value of k used here will be passed as a list of a single element
    # after all classification operations are done, retrieve the first element of the generated 3d arrays (classif & classif_support) to retrieve the data
    classif = classif[0]
    classif_support = classif_support[0]
    
    classif_named = get_named_classif(classif, reference.names_tab)
    
    result = KNNResult(classif, classif_named, classif_support, reference.ranks, reference.start, reference.end, n, k, method, sites, query.branches)
    return result