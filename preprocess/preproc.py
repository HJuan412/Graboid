#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:40:02 2022

@author: hernan
"""

#%%
import sys
sys.path.append('preprocess')
sys.path.append('classif')
#%% libraries
from classif import cost_matrix
from preprocess import windows
from preprocess import feature_selection as fsel
from preprocess import taxon_study as tstud
import numpy as np
import pandas as pd
import time
#%%
taxon = 'nematoda'
marker = '18s'
in_dir = '/home/hernan/PROYECTOS/nematoda_18s/out_dir'
out_dir = '/home/hernan/PROYECTOS/nematoda_18s/out_dir'
tmp_dir = '/home/hernan/PROYECTOS/nematoda_18s/tmp_dir'
warn_dir = '/home/hernan/PROYECTOS/nematoda_18s/warn_dir'

max_pos = 1730
w_len = 200
row_thresh = 0.2
col_thresh = 0.2

cost_mat = cost_matrix.cost_matrix()
id_mat = cost_matrix.id_matrix()

#%% cycle trough rank and windows
step = 15

rang = np.arange(0, max_pos - w_len, step)
if rang[-1] < max_pos - w_len:
    np.append(rang, max_pos - w_len)
ranks = ('phylum', 'class', 'order', 'family', 'genus', 'species')
if __name__ == '__main__':
    # generate window
    wl = windows.WindowLoader(taxon, marker, in_dir, out_dir, tmp_dir, warn_dir)
    # TODO: handle warning when window doesn't pass one of the thresholds
    for w_start in rang:
        w_end = w_start + w_len
        window = wl.get_window(w_start, w_end, row_thresh, col_thresh)
        
        for rank in ranks:
            # select attributes
            if len(window.cons_mat > 1):
                selector = fsel.Selector(window.cons_mat, window.cons_tax)
                selector.set_rank(rank)
                
                selector.select_taxons(minseqs = 10)
                selector.generate_diff_tab()
                selector.select_sites(10)
                t_mat, t_tax = selector.get_training_data()
                
                if len(t_mat) > 1:
                    super_c = tstud.SuperCluster(t_mat, t_tax, rank)
                    col_tax, col_mat = super_c.get_collapsed_data()

#%% single execution
w_start = 250
rank = 'family'
if __name__ == '__main__':
    # generate window
    wl = windows.WindowLoader(taxon, marker, in_dir, out_dir, tmp_dir, warn_dir)
    # TODO: handle warning when window doesn't pass one of the thresholds
    w_end = w_start + w_len
    w0 = np.array(wl.matrix[:,w_start:w_end])
    rows = windows.filter_matrix(w0, 0.2)
    w1 = w0[rows]
    cols = windows.filter_matrix(w1, 0.2, 1)
    window = w1[:,cols]
    effs = windows.get_effective_seqs(window)
    #%%
    # pre collapse happens in window construction
    window = wl.get_window(w_start, w_end, row_thresh, col_thresh)
    # select attributes
    if len(window.cons_mat > 1):
        selector = fsel.Selector(window.cons_mat, window.cons_tax)
        selector.set_rank(rank)
        
        selector.select_taxons(minseqs = 10)
        selector.generate_diff_tab()
        selector.select_sites(10)
        t_mat, t_tax = selector.get_training_data()
        # post collapse, generates the final data matrix and taxonomy table
        x, y = windows.collapse(t_mat, t_tax.reset_index())
        if len(t_mat) > 1:
            super_c = tstud.SuperCluster(t_mat, t_tax, rank)
            col_tax, col_mat = super_c.get_collapsed_data()
