#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:10:17 2025

@author: hernan

This script contains the knn agent class, handles data loading,
learning (calibration), and classification.
"""

#%% modules
import numpy as np

from Graboid.database import data_holder
from Graboid.KNN import calibrator
from Graboid.KNN import classificator
from Graboid.KNN import cost_matrices

#%% functions
#%% classes

class KNNagent:
    def __init__(self, out_dir, transition=1, transversion=2):
        self.out_dir = out_dir
        self.transition = 1
        self.transversion = 2
        self.data = data_holder.DataHolder()
        self.calibrator = None
        
    def load_data(self, db_dir, min_coverage=0, required_rank='family'):
        self.data.load_reference(db_dir, min_coverage, required_rank)
    
    def learn(self,
              max_n,
              step_n,
              max_k,
              step_k,
              row_thresh,
              min_n=5,
              min_k=1,
              criterion='orbit',
              cost_matrix='s1v2',
              w_size=200,
              w_step=100,
              w_coords=None,
              threads=1,
              mode='sliding'):
        try:
            cost_matrix = cost_matrices.matrices[cost_matrix]
        except KeyError:
            raise Exception(f'Given cost matrix name: {cost_matrix} is not found, available names are: {list(cost_matrices.matrices.keys())}')
            
        if mode == 'sliding':
            self.calibrator = calibrator.calibrate_sliding(self.data.R, w_size, w_step, max_n, step_n, max_k, step_k, cost_matrix, row_thresh, min_n, min_k, criterion, threads)
        elif mode == 'custom':
            self.calibrator = calibrator.calibrate_custom(self.data.R, w_coords, max_n, step_n, max_k, step_k, cost_matrix, row_thresh, min_n, min_k, criterion, threads)
        self.calibrator.get_best('f1')
        
    def classify(self,
                 query,
                 w_start,
                 w_end,
                 n=5,
                 k=1,
                 method='unweighted',
                 cost_matrix='s1v2',
                 criterion='orbit',
                 threads=1,
                 evalue=0.005,
                 query_name='QUERY',
                 min_coverage=.95,
                 max_unk_thresh=.2,
                 *taxa):
        
        try:
            cost_matrix = cost_matrices.matrices[cost_matrix]
        except KeyError:
            raise Exception(f'Given cost matrix name: {cost_matrix} is not found, available names are: {list(cost_matrices.matrices.keys())}')
        
        self.data.load_query(query, self.out_dir, evalue, threads, query_name, min_coverage)
        self.data.select_region(w_start, w_end, max_unk_thresh)
        
        classif_result = classificator.classify(self.data.Q, self.data.R, cost_matrix, n, k, method, criterion, threads)
        return classif_result