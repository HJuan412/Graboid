#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 10:21:21 2025

@author: hernan

This script is for housing some default weight matrices, also contains functions for loading or creating custom matrices
"""

#%% libraries
import numpy as np
import pandas as pd

#%% default matrices

id_mat = np.array([[.75, .75, .75, .75, .75],
                   [.75,  0.,  1.,  1.,  1.],
                   [.75,  1.,  0.,  1.,  1.],
                   [.75,  1.,  1.,  0.,  1.],
                   [.75,  1.,  1.,  1.,  0.]], dtype=np.float32)
s1v2_mat = np.array([[1.25, 1.25, 1.25, 1.25, 1.25],
                     [1.25, 0.  , 2.  , 1.  , 2.  ],
                     [1.25, 2.  , 0.  , 2.  , 1.  ],
                     [1.25, 1.  , 2.  , 0.  , 2.  ],
                     [1.25, 2.  , 1.  , 2.  , 0.  ]], dtype=np.float32)

matrices = {'s1v2':s1v2_mat, 'id':id_mat}
def display_mat(mat):
    return pd.DataFrame(mat, index=['n', 'A', 'C', 'G', 'T'], columns=['n', 'A', 'C', 'G', 'T'])

def load_mat(file, sep, name):
    mat = pd.read_csv(file, sep=sep, header=None).values.astype(np.float32)
    if mat.shape[0] != 5 or mat.shape[1] != 5:
        raise Exception(f'Matrix must be of shape [5 5], loaded one is of shape {mat.shape}')
    matrices[name] = mat