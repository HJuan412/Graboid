#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 09:43:40 2022

@author: hernan
Contains tools for handling the generated matrix files
"""

#%% libraries
import numpy as np
import pandas as pd
import os

#%% functions
def process_matrix_filename(mat_filename):
    split_filename = mat_filename.split('/')[-1][:-4].split('_')
    n = int(split_filename[3][1:])
    bounds = split_filename[2].split('-')
    start = int(bounds[0])
    end = int(bounds[1])
    return n, start, end

#%% classes
class MatrixLoader():
    def __init__(self):
        self.clear()

    def load(self, matrix_path):
        if matrix_path is None:
            return
        if os.path.isfile(matrix_path):
            self.n, self.start, self.end = process_matrix_filename(matrix_path)
            matrix = pd.read_csv(matrix_path, index_col = 0)
            self.accs = matrix.index.tolist()
            self.matrix = matrix.to_numpy()
    
    def clear(self):
        self.accs = []
        self.matrix = None
        self.n = 0
        self.start = 0
        self.end = 0