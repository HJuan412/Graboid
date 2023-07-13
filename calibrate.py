#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:49:19 2023

@author: hernan
Calibration director. Calibrates a given database against itself
"""

import argparse
import logging
import os
import shutil

from DATA import DATA
from calibration import cal_main
from classification import cost_matrix
#% set logger
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

def main(out_dir,
         database,
         max_n,
         step_n,
         max_k,
         step_k,
         mat_code,
         row_thresh,
         col_thresh,
         min_seqs,
         rank,
         min_n,
         min_k,
         threads,
         clear,
         criterion,
         collapse_hm=True,
         **kwargs):
    # initialize calibrator and set database
    calibrator = cal_main.Calibrator()
    try:
        calibrator.set_database(database)
    except:
        raise
    
    # set windows
    try:
        if 'w_size' in kwargs.keys() and 'w_step' in kwargs.keys():
            calibrator.set_sliding_windows(kwargs['w_size'], kwargs['w_step'])
        elif 'w_starts' in kwargs.keys() and 'w_ends' in kwargs.keys():
            calibrator.set_custom_windows(kwargs['w_starts'], kwargs['w_ends'])
        else:
            print('Missing arguments to set calibration windows. Use either a sliding window size and step or matching sets of custom start and end positions')
    except:
        raise
    
    # set output directory
    if clear and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    try:
        calibrator.set_outdir(out_dir)
    except:
        raise
    
    # generate cost matrix
    try:
        cost_mat = cost_matrix.get_matrix(mat_code)
    except:
        raise
    
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
                           collapse_hm = collapse_hm,
                           threads = threads)
    return
#%%
parser = argparse.ArgumentParser(prog='Graboid CALIBRATE',
                                 usage='%(prog)s MODE_ARGS [-h]',
                                 description='Graboid CALIBRATE performs a grid search of the given ranges of K and n along a sliding window over the alignment matrix')
parser.add_argument('-o', '--out_dir',
                    help='Output directory',
                    type=str)
parser.add_argument('-db', '--database',
                    help='Database to calibrate',
                    type=str)
parser.add_argument('-rt', '--row_thresh',
                    default=0.2,
                    help='Empty row threshold. Default: 0.2',
                    type=float)
parser.add_argument('-ct', '--col_thresh',
                    default=0.2,
                    help='Empty column threshold. Default: 0.2',
                    type=float)
parser.add_argument('-ms', '--min_seqs',
                    default=10,
                    help='Minimum number of sequences allowed per taxon. Default: 10',
                    type=int)
parser.add_argument('-rk', '--rank',
                    default='genus',
                    help='Rank to be used for feature selection. Default: genus',
                    type=str)
parser.add_argument('-dm', '--dist_mat',
                    default='id',
                    help='Distance matrix to be used for distance calculation. Valid codes: "id" and "s<int>v<int>". Default: id',
                    type=str)
parser.add_argument('-wz', '--w_size',
                    default=200,
                    help='Sliding window size. Default: 200',
                    type=int)
parser.add_argument('-ws', '--w_step',
                    default=30,
                    help='Sliding window displacement. Default: 30',
                    type=int)
parser.add_argument('-mk', '--max_k',
                    default=15,
                    help='Max value of K. Default: 15',
                    type=int)
parser.add_argument('-sk', '--step_k',
                    default=2,
                    help='Rate of increase of K. Default: 2',
                    type=int)
parser.add_argument('-mn', '--max_n',
                    default=30,
                    help='Max value of n. Default: 30',
                    type=int)
parser.add_argument('-sn', '--step_n',
                    default=5,
                    help='Rate of increase of n. Default: 5',
                    type=int)
parser.add_argument('-nk', '--min_k',
                    default=1,
                    help='Min value of K. Default: 1',
                    type=int)
parser.add_argument('-nn', '--min_n',
                    default=5,
                    help='Min value of n. Default: 5',
                    type=int)
parser.add_argument('-c', '--criterion',
                    default = 'orbit',
                    choices=['orbit', 'neighbour'])
parser.add_argument('-t', '--threads',
                    default=1,
                    help='Number of threads to be used in the calibration. Default: 1',
                    type=int)
parser.add_argument('--clear',
                    action='store_true',
                    help='Clear previous calibration if present. If not activated, calibration process will be aborted.') # TODO: should ask for alternative location
parser.add_argument('--keep',
                    action='store_true',
                    help='Keep the generated classifiaction files. WARNING: this will use aditional disk space.')
parser.add_argument('--log_report',
                    action='store_true',
                    help='Log memory usage and time elapsed for each calibration cycle')

if __name__ == '__main__':
    args = parser.parse_args()
    main(out_dir = args.out_dir,
         database = args.database,
         max_n = args.max_n,
         step_n = args.step_n,
         max_k = args.max_k,
         step_k = args.min_k,
         mat_code = args.dist_mat,
         row_thresh = args.row_thresh,
         col_thresh = args.col_thresh,
         min_seqs = args.min_seqs,
         rank = args.rank,
         min_n = args.min_n,
         min_k = args.min_k,
         threads = args.threads,
         clear = args.clear,
         criterion = args.criterion,
         w_size = args.w_size,
         w_step = args.w_step)