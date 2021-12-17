#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 09:18:59 2021

@author: hernan
Pruebas con random forest
"""

#%% libraries
from homolog_codes import Homologuer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

#%%
class mat_loader():
    def __init__(self, mat_dir,
                 ncbi_tax,
                 ncbi_names,
                 ncbi_nodes,
                 bold_tax,
                 bold_names,
                 bold_nodes):
        self.mat_dir = mat_dir
        self.homologs = Homologuer(bold_tax = bold_tax,
                                   bold_names = bold_names,
                                   bold_nodes = bold_nodes,
                                   ncbi_tax = ncbi_tax,
                                   ncbi_names = ncbi_names,
                                   ncbi_nodes = ncbi_nodes)
        self.set_tax_tabs()
        self.list_matrixes()
    
    def set_tax_tabs(self):
        self.homologs.fix_homologs()
        self.homologs.get_homologued()
        self.tax_tab = self.homologs.homolog_tax
        self.names_tab = self.homologs.homolog_names

    def list_matrixes(self):
        mat_files = os.listdir(self.mat_dir)
        split_files = [f.split('.')[0].split('_') for f in mat_files]
        file_id = [int(s[0]) for s in split_files] # id of the alignment file
        file_n = [int(s[-1][1:]) for s in split_files] # n seqs in the alignment files
        aln_bounds = [s[2].split('-') for s in split_files] # alignment bounds for the alignment files
        file_start = [int(a[0]) for a in aln_bounds]
        file_end = [int(a[1]) for a in aln_bounds]
        self.mat_tab = pd.DataFrame({'start':file_start, 'end':file_end, 'n': file_n, 'path':mat_files}, index = file_id)
        self.mat_tab.sort_index(inplace = True)
    
    def load_matrix(self, mat_index):
        mat_file = self.mat_tab.loc[mat_index, 'path']
        mat_path = self.mat_dir + '/' + mat_file
        matrix = pd.read_csv(mat_path, index_col = 0)
        accs = matrix.index
        matrix_taxonomy = self.tax_tab.loc[accs]
        return matrix, matrix_taxonomy
    
    def get_mat(self, mat_index, tax_rank = 'family'):
        matrix, matrix_taxonomy = self.load_matrix(mat_index)
        matrix['tax'] = matrix_taxonomy[tax_rank]
        return matrix

#%% functions
def prepare_mat(mat):
    # filter unknown entries
    # filter taxons with only 1 entry
    filtered1 = mat.loc[mat['tax'] != 0]
    taxes = filtered1.iloc[:,-1]
    tax_counts = taxes.value_counts()
    filtered_taxes = tax_counts.loc[tax_counts > 1].index.tolist()
    rejected_taxes = tax_counts.loc[tax_counts == 1].index.tolist()
    filtered_mat = filtered1.loc[filtered1['tax'].isin(filtered_taxes)]
    return filtered_mat, rejected_taxes

def normalize_confusion(confusion):
    total_trues = confusion.sum(axis = 1)
    norm_confusion = confusion.copy().astype(float)
    
    for idx, true in enumerate(total_trues):
        norm_confusion[:,idx] /= true
    return norm_confusion
#%% test vars
mat_dir = 'Dataset/12_11_2021-23_15_53/Matrices/Nematoda/COI'
ncbi_tax = 'Databases/12_11_2021-23_15_53/Taxonomy_files/Nematoda_COI_tax.tsv'
bold_tax = '/home/hernan/PROYECTOS/Nematoda_BOLD_acc2taxid.tsv'
ncbi_names = 'Databases/12_11_2021-23_15_53/Taxonomy_files/cropped_names.tsv'
ncbi_nodes = 'Databases/12_11_2021-23_15_53/Taxonomy_files/cropped_nodes.tsv'
bold_names = '/home/hernan/PROYECTOS/BOLD_names.tsv'
bold_nodes = '/home/hernan/PROYECTOS/BOLD_nodes.tsv'

rank = 'family'
ml = mat_loader(mat_dir = mat_dir,
                ncbi_tax = ncbi_tax,
                ncbi_names = ncbi_names,
                ncbi_nodes = ncbi_nodes,
                bold_tax = bold_tax,
                bold_names = bold_names,
                bold_nodes = bold_nodes)
#%%
labels = ml.tax_tab[rank].unique()

top_n = ml.mat_tab.quantile(0.5)['n']
mat_idx = ml.mat_tab.loc[ml.mat_tab['n'] >= top_n].index

prc_tab = pd.DataFrame(index = mat_idx, columns = labels)
rec_tab = pd.DataFrame(index = mat_idx, columns = labels)
fsc_tab = pd.DataFrame(index = mat_idx, columns = labels)

repr_tab = pd.DataFrame(index = mat_idx, columns = labels)

for idx in mat_idx:
    print(f'Processing mat {idx}')
    mat = ml.get_mat(idx, rank)
    
    # register taxon representations in matrix
    tax_counts = mat['tax'].value_counts()
    repr_tab.at[idx, tax_counts.index] = tax_counts.values

    #% sklearn preproc
    fmat, rtax = prepare_mat(mat)
    
    x = fmat.iloc[:,:-1]
    y = fmat.iloc[:,-1]
    
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, stratify = y)
    
    #% rf train
    rand_forest = rf(n_estimators = 100,
                     criterion='entropy',
                     max_depth = 25,
                     verbose = 1)
    
    rand_forest.fit(x_train, y_train)
    #% rf test
    prediction = rand_forest.predict(x_test)
    
    acc = accuracy_score(y_test, prediction)
    prc = precision_score(y_test, prediction, labels = labels, average = None)
    rec = recall_score(y_test, prediction, labels = labels, average = None)
    fsc = f1_score(y_test, prediction, labels = labels, average = None)
    
    prc_tab.at[idx] = prc
    rec_tab.at[idx] = rec
    fsc_tab.at[idx] = fsc

repr_tab.fillna(0, inplace = True)
repr_tab.drop(columns = 0, inplace = True)
# filter columns with no representation
sum_reps = repr_tab.sum(0)
not_represented = sum_reps.loc[sum_reps <= 0].index
repr_tab.drop(columns = not_represented, inplace = True)
#%% analyze results
best_prcs = prc_tab.max(axis = 0)
best_recs = rec_tab.max(axis = 0)
best_fscs = rec_tab.max(axis = 0)

best_prcs = best_prcs.loc[best_prcs > 0]
best_recs = best_recs.loc[best_recs > 0]
best_fscs = best_fscs.loc[best_fscs > 0]

count_prcs = best_prcs.value_counts(normalize = True).sort_index()
count_recs = best_recs.value_counts(normalize = True).sort_index()
count_fscs = best_fscs.value_counts(normalize = True).sort_index()

#%% plot results
fig, ax = plt.subplots(figsize = (10, 7))

ax.plot(count_prcs.index, count_prcs.values, label = 'Best precisions')
ax.plot(count_recs.index, count_recs.values, label = 'Best recalls')
ax.plot(count_fscs.index, count_fscs.values, label = 'Best f1s')
ax.legend()