#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:49:19 2023

@author: hernan
Calibration director. Calibrates a given database against itself
"""

from calibration import cal_main
from classification import cost_matrix
from parsers import cal_parser as parser
        
#%%
def main(database,
         out_dir,
         row_thresh,
         col_thresh,
         min_seqs,
         rank,
         transition,
         transversion,
         max_k,
         step_k,
         max_n,
         step_n,
         min_k,
         min_n,
         criterion,
         threads=1,
         keep=False,
         w_size=None,
         w_step=None,
         w_start=None,
         w_end=None):
         # out_dir,
         # database,
         # max_n,
         # step_n,
         # max_k,
         # step_k,
         # mat_code,
         # row_thresh,
         # col_thresh,
         # min_seqs,
         # rank,
         # min_n,
         # min_k,
         # threads,
         # clear,
         # criterion,
         # collapse_hm=True,
         # **kwargs):
    
    # initialize calibrator and set database
    calibrator = cal_main.Calibrator()
    try:
        calibrator.set_database(database)
    except:
        raise
    
    # set windows
    try:
        if not w_start is None and not w_end is None:
            calibrator.set_custom_windows(w_start, w_end)
        elif not w_size is None and not w_step is None:
            calibrator.set_sliding_windows(w_size, w_step)
        else:
            print('Missing arguments to set calibration windows. Use either a sliding window size and step or matching sets of custom start and end positions')
            return
    except:
        raise
        
    # generate cost matrix
    cost_mat = cost_matrix.cost_matrix(transition, transversion)
    
    
    # grid calibrate
    calibrator.grid_search(max_n = max_n,
                           step_n = step_n,
                           max_k = max_k,
                           step_k = step_k,
                           cost_mat = cost_mat,
                           row_thresh = row_thresh,
                           col_thresh = col_thresh,
                           min_seqs = min_seqs,
                           rank = rank,
                           min_n = min_n,
                           min_k = min_k,
                           criterion = criterion,
                           collapse_hm = True,
                           threads = threads)
    return
#%%
if __name__ == '__main__':
    args = parser.parse_args()
    main(database = args.database,
         out_dir = args.out_dir,
         row_thresh = args.row_thresh,
         col_thresh = args.col_thresh,
         min_seqs = args.min_seqs,
         rank = args.rank,
         transition = args.transition,
         transversion = args.transversion,
         w_size = args.w_size,
         w_step = args.w_step,
         w_start = args.w_start,
         w_end = args.w_end,
         max_k = args.max_k,
         step_k = args.min_k,
         max_n = args.max_n,
         step_n = args.step_n,
         min_k = args.min_k,
         min_n = args.min_n,
         criterion = args.criterion,
         threads = args.threads,
         keep = args.keep)
    