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
def cost_matrix(transition = 1, transversion = 2):
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
    # X 	gap of indeterminate length

    bases = ['a', 'c', 'g', 't']
    base_types = {'a':'pu', 'c':'py', 'g':'pu', 't':'py'}
    short_dist = np.zeros((5,5))
    
    for idx1, base1 in enumerate(bases):
        for idx2, base2 in enumerate(bases):
            base1_type = base_types[base1]
            base2_type = base_types[base2]
            
            if base1 == base2:
                distance = 0
            elif base1_type == base2_type:
                distance = transition
            else:
                distance = transversion
            
            short_dist[idx1, idx2] = distance
    
    index_dict = {'a':0, 'c':1, 'g':2, 't':3}
    fasta_codes = {'a':'a',
                   'c':'c',
                   'g':'g',
                   't':'t',
                   'u':'t',
                   'r':['a', 'g'],
                   'y':['c', 't'],
                   'k':['g', 't'],
                   'm':['a', 'c'],
                   's':['c', 'g'],
                   'w':['a', 't'],
                   'b':['c', 'g', 't'],
                   'd':['a', 'g', 't'],
                   'h':['a', 'c', 't'],
                   'v':['a', 'c', 'g'],
                   'n':['a', 'c', 'g', 't'],
                   '-':['a', 'c', 'g', 't']}
    
    fasta_idx = {}
    for k,v in fasta_codes.items():
        idx_list = np.array([index_dict[b] for b in v])
        fasta_idx[k] = idx_list
    
    full_dist = np.zeros((len(fasta_codes), len(fasta_codes)))
    np.set_printoptions(precision = 2)
    for idx1, k1 in enumerate(fasta_idx):
        for idx2, k2 in enumerate(fasta_idx):
            dists = []
            for b1 in fasta_idx[k1]:
                for b2 in fasta_idx[k2]:
                    dists.append(short_dist[b1, b2])
            full_dist[idx1, idx2] = np.average(dists)
    return np.array(full_dist, dtype = np.float64)

def id_matrix():
    mat = np.ones((17,17))
    for i in range(16):
        mat[i,i] = 0
    
    return mat