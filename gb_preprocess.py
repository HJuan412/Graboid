#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 13:32:49 2021

@author: hernan

New graboid
"""

#%% libraries
import numpy as np
import pandas as pd
import os

#%% functions
def build_mat_tab(mat_dir):
    mat_tab = pd.DataFrame(columns = ['start', 'end', 'n', 'File'])
    files = os.listdir(mat_dir)
    for file in files:
        split_file = file.split('.mat')[0].split('_')
        idx = int(split_file[0])
        coords = [int(i) for i in split_file[2].split('-')]
        n = int(split_file[3][1:])
        mat_tab.at[idx] = [coords[0], coords[1], n, file]
    
    mat_tab.sort_index(inplace = True)
    
    return mat_tab

def entropy(array):
    vals, counts = np.unique(array, return_counts = True)
    freqs = counts / len(array)
    entropy = -(freqs * np.log(freqs)).sum()
    return entropy

def gain(x, c, ent):
    vals, counts = np.unique(x, return_counts = True)
    freqs = counts / len(x)
    
    G = 0
    for v,f in zip(vals, freqs):
        idx = np.where(x == v)[0]
        sub_c = c[idx]
        sub_ent = entropy(sub_c)
        G += f * sub_ent
    
    return ent - G

#%% classes
class MatrixLoader():
    def __init__(self, mat_dir, upper = 0.25):
        self.mat_dir = mat_dir
        self.mat_tab = build_mat_tab(mat_dir)
        self.max_n = self.mat_tab['n'].max()
        self.upper_p(upper)
    
    def upper_p(self, p = 0.25):
        # store the p percentile of most populated matrices
        self.p = p
        p_int = int(self.mat_tab.shape[0] * p)
        sorted_tab = self.mat_tab.sort_values(by = 'n', ascending = False)
        self.upper = sorted_tab.iloc[:p_int]
    
    def get_matrix_path(self, idx):
        # load a matrix from the matrix tab, idx is the index of the matrix
        if idx in self.mat_tab.index:
            mat_file = self.mat_tab.loc[idx, 'File']
            mat_path = f'{self.mat_dir}/{mat_file}'
            return mat_path

class AlignmentLoader():
    def __init__(self, mat_path, tax_path):
        self.load_matrix(mat_path)
        self.load_tax_tab(tax_path)
    
    def load_matrix(self, mat_path):
        matrix = pd.read_csv(mat_path, index_col = 0)
        self.matrix = matrix.to_numpy()
        self.accs = matrix.index.tolist()

    def load_tax_tab(self, tax_path):
        tax_tab = pd.read_csv(tax_path, index_col = 0, sep = '\t')
        self.tax_tab = tax_tab.loc[self.accs]
    
    def get_tax_codes(self, rank):
        # get the taxonomy code at the given rank for each sequence
        if rank in self.tax_tab.columns:
            return self.tax_tab[rank].to_numpy()

class PreProcessor():
    def __init__(self, mat_path, tax_path):
        self.aln = AlignmentLoader(mat_path, tax_path)
        self.matrix = self.aln.matrix
        self.accs = self.aln.accs
        self.set_tax_codes()
        self.get_global_entropy()
        self.get_gain_per_col()
    
    def set_tax_codes(self, rank = 'family'):
        # get the taxonomy code at the given rank for each sequence
        self.tax_codes = self.aln.get_tax_codes(rank)

    def get_global_entropy(self):
        # calculate the entropy of the taxonomy list for the entire table
        self.global_entropy = entropy(self.tax_codes)
    
    def get_gain_per_col(self):
        # calculate information gain for each column
        self.gains = np.apply_along_axis(gain, 0, self.matrix, self.tax_codes, self.global_entropy)
    
    def select_columns(self, p):
        # select the p columns with the highest information gain
        sorted_gains = np.argsort(self.gains)[::-1]
        p_idx = sorted_gains[:p]
        self.selected = self.matrix[:, p_idx]
        self.p_idx = p_idx
