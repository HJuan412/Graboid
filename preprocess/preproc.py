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
import cost_matrix
import windows
import feature_selection as fsel
import taxon_study as tstud
#%%
taxon = 'nematoda'
marker = '18s'
in_dir = 'nematoda_18s/out_dir'
out_dir = 'nematoda_18s/out_dir'
tmp_dir = 'nematoda_18s/tmp_dir'
warn_dir = 'nematoda_18s/warn_dir'

w_start = 200
w_end = 400
row_thresh = 0.2
col_thresh = 0.5

cost_mat = cost_matrix.cost_matrix()
id_mat = cost_matrix.id_matrix()

if __name__ == '__main__':
    # generate window
    wl = windows.WindowLoader(taxon, marker, in_dir, out_dir, tmp_dir, warn_dir)
    # TODO: handle warning when window doesn't pass one of the thresholds
    window = wl.get_window(w_start, w_end, row_thresh, col_thresh)
    window.build_cons_mat() # builds a matrix without repeated seqs and a consensus tax_table (cons_mat, cons_tax)
    
    # select attributes
    gain_tab = fsel.get_gain(window.cons_mat, window.cons_tax)
    diff_tab = fsel.get_ent_diff(window.cons_mat, window.cons_tax)
    
    gain_selected = fsel.select_features(gain_tab, 'family', 15, 'gain')
    diff_selected = fsel.select_features(diff_tab, 'family', 10, 'diff')

    gain_train = fsel.build_training_data(window.cons_mat, gain_selected)
    diff_train = fsel.build_training_data(window.cons_mat, diff_selected)
    
    super_c = tstud.SuperCluster(diff_train, window.cons_tax, 'family')
    col_tax, col_mat = super_c.get_collapsed_data()
