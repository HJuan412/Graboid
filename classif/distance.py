#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:50:21 2022

@author: hernan
Distance calculators
"""

#%% libraries
from numba import njit
import numpy as np

#%% funtions
@njit
def calc_distance(seq1, seq2, dist_mat):
    dist = 0
    for x1, x2 in zip(seq1, seq2):
        dist += dist_mat[x1, x2]
    
    return dist

#%%
window = None
cost = None
id_mat = None
#%%
import time
nseqs = window.cons_mat.shape[0]
paired = np.zeros((nseqs, nseqs))

t0 = time.time()
for idx0, seq0 in enumerate(window.cons_mat):
    for idx1 in range(idx0 + 1, window.cons_mat.shape[0]):
        seq1 = window.cons_mat[idx1]
        paired[idx0, idx1] = calc_distance(seq0, seq1, id_mat)
t1 = time.time()

e = t1-t0
print(e)
#%%

mat = np.zeros((4,4))

for idx0 in range(mat.shape[0]):
    for idx1 in range(idx0,mat.shape[0]):
        mat[idx0, idx1] = 1