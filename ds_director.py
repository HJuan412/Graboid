#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 15:27:16 2021

@author: hernan
This script directs Dataset construction
"""
#%% libraries
import ds_blast as blst
import ds_windows as wndw
import ds_matrix as mtrx
import os
#%% functions
def generate_dirnames(dirname):
    return [f'{dirname}/BLAST_reports', f'{dirname}/Windows', f'{dirname}/Matrices', f'{dirname}/Warnings']
#%%
ds_dir = 'Dataset/12_11_2021-23_15_53'
db_dir = 'Databases/12_11_2021-23_15_53/Sequence_files'
seq_dir = 'Databases/12_11_2021-23_15_53/Sequence_files'
ref_dir = 'Reference_data/Reference_genes'

#%%
class Director():
    def __init__(self, ds_dir, db_dir, seq_dir, ref_dir, width = 100, step = 15, gap_thresh = 0.1):
        self.ds_dir = ds_dir
        self.db_dir = db_dir
        self.seq_dir = seq_dir
        self.ref_dir = ref_dir
        self.generate_subdirs()
        self.create_dirs()
        self.width = width
        self.step = step
        self.gap_thresh = 0.1
    
    def generate_subdirs(self):
        subdirs = generate_dirnames(self.ds_dir)
        self.blst_dir, self.wndw_dir, self.mtrx_dir, self.warn_dir = subdirs
    
    def create_dirs(self):
        for d in [self.ds_dir, self.blst_dir, self.wndw_dir, self.mtrx_dir, self.warn_dir]:
            if not os.path.isdir(d):
                os.mkdir(d)
    
    def direct_blast(self):
        self.blaster = blst.Blaster(in_dir = self.seq_dir, out_dir = self.blst_dir, ref_dir = self.ref_dir, warn_dir = self.warn_dir)
        self.blaster.blast()
    
    def direct_windows(self):
        self.window_director = wndw.WindowDirector(blast_dir = self.blst_dir, seq_dir = self.seq_dir, out_dir = self.wndw_dir, warn_dir = self.warn_dir)
        self.window_director.direct(self.width, self.step, self.gap_thresh)

    def direct_matrices(self):
        self.matrix_director = mtrx.MatrixDirector(wndw_dirs = self.window_director.subdirs, out_dir = self.mtrx_dir, warn_dir = self.warn_dir)
        self.matrix_director.direct()

    def direct(self):
        print('Generating BLAST alignments')
        self.direct_blast()
        print('Generating sequence windows')
        self.direct_windows()
        print('Building matrixes')
        self.direct_matrices()