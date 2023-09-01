#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 10:39:57 2023

@author: hernan
Preprocess data for classification
"""

#%% libraries
import numpy as np
# Graboid libraries
from preprocess import feature_selection as fsele
from preprocess import sequence_collapse as sq
from preprocess import windows as wn
#%% functions
def collapse(classifier, w_start, w_end, n, rank='genus', row_thresh=0.1, col_thresh=0.1, min_seqs=50):
    """Collapse reference and query windows"""
    # collapse reference window, use the classifier's extended taxonomy table
    ref_window = wn.Window(classifier.ref_matrix,
                           classifier.tax_tab,
                           w_start,
                           w_end,
                           row_thresh=row_thresh,
                           col_thresh=col_thresh,
                           min_seqs=min_seqs)
    print('Collapsed reference window...')
    # build the collapsed reference window's taxonomy
    win_tax = classifier.tax_ext.loc[ref_window.taxonomy][[rank]] # trick: if the taxonomy table passed to get_sorted_sites has a single rank column, entropy difference is calculated for said column
    # sort informative sites
    sorted_sites = fsele.get_sorted_sites(ref_window.window, win_tax) # remember that you can use return_general, return_entropy and return_difference to get more information
    sites = np.concatenate(fsele.get_nsites(sorted_sites[0], n = n))
    # collapse query window, use only the selected columns to speed up collapsing time
    qry_cols = np.arange(w_start, w_end)[sites] # get selected column indexes
    qry_matrix = classifier.query_matrix[:, qry_cols]
    qry_branches = sq.collapse_window(qry_matrix)
    qry_window = qry_matrix[[br[0] for br in qry_branches]]
    print('Collapsed query window...')
    return ref_window, qry_window, qry_branches, win_tax, sites
