#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 10:19:17 2023

@author: hernan

Calibration preprocessing
"""

#%% libraries
import concurrent.futures
import numpy as np
# Graboid libraries
from preprocess import feature_selection as fsele
from preprocess import windows as wn

#%% functions
def collapse_windows(windows, matrix, tax_tab, row_thresh=0.1, col_thresh=0.1, min_seqs=50, threads=1):
    # collapse the selected windows in the given matrix, apply corresponding filters
    # return:
        # win_indexes : sorted array of collapsed windows indexes
        # win_list : list of Window instances, sorted
        # rej_indexes : sorted array of rejected windows indexes
        # rej_list : list of rejection messages, sorted
    collapsed_windows = {}
    rejected_windows = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        future_windows = {executor.submit(wn.Window, matrix, tax_tab, win[0], win[1], row_thresh, col_thresh, min_seqs):idx for idx, win in enumerate(windows)}
        for future in concurrent.futures.as_completed(future_windows):
            ft_idx = future_windows[future]
            try:
                collapsed_windows[ft_idx] = future.result()
                print(f'Collapsed window {ft_idx}')
            except Exception as excp:
                rejected_windows[ft_idx] = str(excp)
                print(f'Rejected window {ft_idx}')
                continue
    
    # translate the collapsed_windows and rejected_windows dicts into ordered lists
    win_indexes = np.sort(list(collapsed_windows.keys()))
    win_list = [collapsed_windows[idx] for idx in win_indexes]
    
    rej_indexes = np.sort(list(rejected_windows.keys()))
    rej_list = [rejected_windows[idx] for idx in rej_indexes]
    
    return win_indexes, win_list, rej_indexes, rej_list

def select_sites(win_list, tax_ext, rank, min_n, max_n, step_n):
    # get the arrays of selected sites for each collapsed window
    window_sites = []
    for win in win_list:
        win_tax = tax_ext.loc[win.taxonomy][[rank]] # trick: if the taxonomy table passed to get_sorted_sites has a single rank column, entropy difference is calculated for said column
        sorted_sites = fsele.get_sorted_sites(win.window, win_tax) # remember that you can use return_general, return_entropy and return_difference to get more information
        window_sites.append(fsele.get_nsites(sorted_sites[0], min_n, max_n, step_n))
    return window_sites