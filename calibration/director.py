#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:49:19 2023

@author: hernan
Calibration director. Calibrates a given database against itself
"""

from DATA import DATA
from calibration import calibrator as cb

import argparse
import logging
import os
import shutil

#% set logger
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#%
def main(database,
         row_thresh=0.2,
         col_thresh=0.2,
         min_seqs=10,
         rank='genus',
         dist_mat='id',
         w_size=200,
         w_step=15,
         max_k=15,
         step_k=2,
         max_n=30,
         step_n=5,
         min_k=1,
         min_n=5,
         threads=1,
         clear=True):
    # check that database exists
    if not database in DATA.DBASES:
        print(f'Database {database} not found. Existing databases:')
        print('\n'.join(DATA.DBASES))
        return
    # build calibration directory in DATA dir
    db_dir = DATA.DATAPATH + '/' + database
    wrn_dir = db_dir + '/warning'
    cal_dir = db_dir + '/calibration'
    # check cal_dir (if it exists, check clear (if true, overwrite, else interrupt))
    if os.path.isdir(cal_dir):
        print(f'Database {database} has already been calibrated...')
        if not clear:
            raise Exception('Set "clear" as True to replace previous calibration')
        print(f'Removing existing calibration for database {database}...')
        shutil.rmtree(cal_dir)
    os.mkdir(cal_dir)
    
    # prepare calibration logger
    fh = logging.FileHandler(db_dir + '/log.calibration')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    cb.logger.addHandler(fh)
    # assemble calibration meta file
    # calibrate
    # update database map in DATA
    # initialize calibrator, establish windows
    calibrator = cb.Calibrator(cal_dir, wrn_dir)
    calibrator.set_database(database)
    calibrator.set_windows(size = w_size, step = w_step)
    calibrator.dist_mat = dist_mat

    # calibration
    calibrator.grid_search(max_k,
                           step_k,
                           max_n,
                           step_n,
                           min_seqs=10,
                           rank='genus',
                           row_thresh=0.2,
                           col_thresh=0.2,
                           min_k=1,
                           min_n=5,
                           threads=1,
                           keep_classif=True) # TODO: this true is to keep the generated classification file, remember to set as false

#%
main('nem_18s', threads=4)
#%%
parser = argparse.ArgumentParser(prog='Graboid CALIBRATE',
                                 usage='%(prog)s MODE_ARGS [-h]',
                                 description='Graboid CALIBRATE performs a grid search of the given ranges of K and n along a sliding window over the alignment matrix')
parser.add_argument('mode')
parser.add_argument('--db',
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
                    help='Distance matrix to be used for distance calculation. Valid codes: "id" and "s<int>v<int>"',
                    type=str)
parser.add_argument('-wz', '--w_size',
                    default=200,
                    help='Sliding window size. Default: 200',
                    type=int)
parser.add_argument('-ws', '--w_step',
                    default=15,
                    help='Sliding window displacement. Default: 15',
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
parser.add_argument('-t', '--threads',
                       default=1,
                       help='Number of threads to be used in the calibration. Default: 1',
                       type=int)