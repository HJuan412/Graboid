#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 12:09:54 2021

@author: hernan
Transform Alignment windows into matrixes
"""
# TODO: In theory, could merge this step with the window building (skips having to store windows as fasta files)
#%% libraries
from Bio import AlignIO
from glob import glob
import numpy as np
import os
import pandas as pd
import string

#%% variables
bases = 'acgturykmswbdhvn-'

#%% functions
def get_case(string_vals):
    for val in string_vals:
        if val in string.ascii_lowercase:
            return 0 # align is lowercase
        elif val in string.ascii_uppercase:
            return 1 # align is uppercase

def build_dirtab(dirlist):
    dir_tab = pd.DataFrame(columns=['Taxon', 'Marker', 'Win dir', 'Mat dir'])
    for idx, subdir in enumerate(dirlist):
        split_dir = subdir.split('/')
        tax = split_dir[-2]
        mark = split_dir[-1]
        dir_tab.at[idx] = [tax, mark, subdir]
    return dir_tab
#%%
class MatrixBuilder():
    # Loads an alignment file in fasta format, converts it to a numpy array
    def __init__(self, aln_file):
        self.load_aln(aln_file)
        self.aln_to_array()
        self.make_trans_dict()
        self.aln_to_numeric()
    
    def load_aln(self, aln_file):
        #TODO: enable other formats. (jariola)
        with open(aln_file, 'r') as handle:
            self.alignment = AlignIO.read(handle, 'fasta')

    def aln_to_array(self):
        # self.acc_list = [seq.id for seq in self.alignment]
        self.accs = [seq.id.split('.')[0] for seq in self.alignment]
        self.seqs = np.array([list(seq.seq) for seq in self.alignment])

    def make_trans_dict(self):
        self.translation_dict = {}

        case = get_case(self.seqs[0])
        if case == 0:
            for idx, base in enumerate(bases):
                self.translation_dict[base] = idx
        elif case == 1:
            for idx, base in enumerate(bases.upper()):
                self.translation_dict[base] = idx

    def aln_to_numeric(self):
        numeric_aln = np.zeros(self.seqs.shape, dtype = int)
        
        for idx, aln in enumerate(self.seqs):
            aln_vals = np.unique(aln)
            for val in aln_vals:                
                num_val = self.translation_dict[val]
                
                numeric_aln[idx] = np.where(aln == val, num_val, numeric_aln[idx])
        
        self.matrix = numeric_aln
    
    def save(self, filename):
        save_tab = pd.DataFrame(self.matrix, index = self.accs)
        save_tab.to_csv(filename)

class MatrixDirector():
    def __init__(self, wndw_dirs, out_dir, warn_dir):
        self.dir_tab = build_dirtab(wndw_dirs)
        self.out_dir = out_dir
        self.make_subdirs()
    
    def make_subdirs(self):
        for idx, row in self.dir_tab.iterrows():
            tax = row['Taxon']
            mark = row['Marker']
            tax_dir = f'{self.out_dir}/{tax}'
            if not os.path.isdir(tax_dir):
                os.mkdir(tax_dir)
            mark_dir = f'{self.out_dir}/{tax}/{mark}'
            if not os.path.isdir(mark_dir):
                os.mkdir(mark_dir)
            self.dir_tab.at[idx, 'Mat dir'] =  mark_dir
    
    def direct(self):
        for idx, row in self.dir_tab.iterrows():
            win_dir = row['Win dir']
            files = glob(f'{win_dir}/*fasta')
            for file in files:
                filename = file.split('/')[-1].split('.fasta')[0]
                mat_builder = MatrixBuilder(file)
                mat_builder.save(f'{self.out_dir}/{filename}.mat')