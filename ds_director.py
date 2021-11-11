#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 15:27:16 2021

@author: hernan
This script directs Dataset construction
"""
#%% libraries
import ds_blast as blst
import ds_window as wndw
import ds_matrix as mtrx
import os
#%% functions
def generate_subdirs(dirname):
    return f'{dirname}/BLAST_reports', f'{dirname}/Windows', f'{dirname}/Matrixes', f'{dirname}/Warnings'

def new_directories(dirname):
    blst_dir, wndw_dir, mtrx_dir, warn_dir = generate_subdirs(dirname)
    
    os.mkdir(dirname)
    for sdir in [blst_dir, wndw_dir, mtrx_dir, warn_dir]:
        os.mkdir(sdir)
    return blst_dir, wndw_dir, mtrx_dir, warn_dir

#%%
dirname = 'Datasets/13_10_2021-20_15_58'
blst_dir, wndw_dir, mtrx_dir, warn_dir = new_directories(dirname)
seq_dir = 'Databases/13_10_2021-20_15_58/Sequence_files'
ref_dir = 'Reference_data/Reference_genes'

# generate ungapped BLAST alignments
blaster = blst.Blaster(seq_dir, blst_dir, ref_dir, warn_dir)
blaster.blast()

# generate windows
width = 100
step = 15
gap_thresh = 0.1

window_director = wndw.WindowDirector(blst_dir, seq_dir, wndw_dir, warn_dir)
window_director.direct(width, step, gap_thresh)

# generate matrixes
matrix_director = mtrx.MatrixDirector(window_director.subdirs, mtrx_dir, warn_dir)
matrix_director.direct()