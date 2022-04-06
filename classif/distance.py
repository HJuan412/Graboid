#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:50:21 2022

@author: hernan
Distance calculators
"""

#%% libraries
from numba import njit

#%% funtions
@njit
def calc_distance(seq1, seq2, dist_mat):
    dist = 0
    for x1, x2 in zip(seq1, seq2):
        dist += dist_mat[x1, x2]
    
    return dist
