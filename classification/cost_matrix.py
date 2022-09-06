#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 13:42:38 2021

@author: hernan
"""
#%% libraries
import numpy as np

#%% functions
# replacement cost matrix
def pair_idxs(bases0, bases1):
    idx_dict = {'a':0, 'c':1, 'g':2, 't':3}
    idxs = []
    for b0 in bases0:
        for b1 in bases1:
            idxs.append((idx_dict[b0], idx_dict[b1]))
    return idxs

def cost_matrix(transition=1, transversion=2):
    # a 	A 	Adenine
    # c 	C 	Cytosine
    # g 	G 	Guanine
    # t 	T 	Thymine
    # u 	U 	Uracil
    # r 	A or G (I) 	puRine
    # y 	C, T or U 	pYrimidines
    # k 	G, T or U 	bases which are Ketones
    # m 	A or C 	bases with aMino groups
    # s 	C or G 	Strong interaction
    # w 	A, T or U 	Weak interaction
    # b 	not A (i.e. C, G, T or U) 	B comes after A
    # d 	not C (i.e. A, G, T or U) 	D comes after C
    # h 	not G (i.e., A, C, T or U) 	H comes after G
    # v 	neither T nor U (i.e. A, C or G) 	V comes after U
    # n 	A C G T U 	Nucleic acid
    
    # base_types = {'a':'pu', 'c':'py', 'g':'pu', 't':'py'}

    # dist mat:
        #   A C G T
        # A
        # C
        # G
        # T
    dist_mat = np.array([[0, transversion, transition, transversion],
                         [transversion, 0, transversion, transition],
                         [transition, transversion, 0, transversion],
                         [transversion, transition, transversion, 0]])
    
    cost_mat = np.zeros((16,16))
    fasta_codes = {'n':'acgt',
                   'a':'a',
                   'c':'c',
                   'g':'g',
                   't':'t',
                   'u':'t',
                   'r':'ag',
                   'y':'ct',
                   'k':'gt',
                   'm':'ac',
                   's':'cg',
                   'w':'at',
                   'b':'cgt',
                   'd':'agt',
                   'h':'act',
                   'v':'acg'}
    for idx0, b0 in enumerate(fasta_codes.values()):
        for idx1, b1 in enumerate(fasta_codes.values()):
            idxs = pair_idxs(b0, b1)
            dists = [dist_mat[idx[0], idx[1]] for idx in idxs]
            cost_mat[idx0, idx1] = np.mean(dists)
    return cost_mat

def id_matrix():
    mat = np.ones((16,16))
    for i in range(1,17):
        mat[i,i] = 0
    return mat
    
def get_matrix(mat_code):
    if mat_code == 'id':
        return id_matrix()
    if mat_code[0] == 's' and mat_code[2] == 'v':
        try:
            transition = int(mat_code[1])
            transversion = int(mat_code[3])
            return cost_matrix(transition, transversion)
        except ValueError:
            pass
    return None