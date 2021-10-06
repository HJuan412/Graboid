#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 10:25:46 2021

@author: hernan
Test director
"""

#%% libraries
import data_preprocess as dpp
import graboid
import numpy as np
import pandas as pd
import visualizer as vis
#%% preprocessing
alnfile = 'Test_data/nem_18S_win_100_step_16/nem-18S_win_272-372_max-gap_10.fasta'
taxfile = '/home/hernan/PROYECTOS/Graboid/Taxonomy/Taxonomies.tab'
namesfile = '/home/hernan/Downloads/Taxonomy_genbank/names.tsv'
names_tab = pd.read_csv(namesfile, sep = '\t', header = None)

handler = dpp.alignment_handler(alnfile, taxfile)

rank = 'family'
ntax = 30
nbase = 30

# select data
selected_data, selected_taxonomy = handler.data_selection(ntax, rank, nbase)

M = graboid.cost_matrix()
k = 3
#%% classify
# clasificación con graboid nativo
classifications = graboid.jacknife_classify(selected_data.to_numpy(), M, k, selected_taxonomy['family'].to_numpy())
confusion = vis.make_confusion(np.array(classifications), selected_taxonomy[rank].to_numpy(), handler.selected_taxons)
metrics = vis.get_metrics(confusion)

vis.graph_confusion(classifications, selected_taxonomy[rank], handler.selected_taxons, names_tab)

#%% multithreading test
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# def jacknife_worker(query_idx):
#     # this function replaces the one in the graboid library, its only argument is the query index, as it takes the reference data, labels and cost matrix from the current namespace
#     query = data[query_idx]
#     references = np.delete(data, query_idx, 0)
#     labels = np.delete(tax_labels, query_idx)
#     return graboid.classify2(query, references, M, k, labels)

#%% serial test
time_tab = pd.DataFrame(columns = ['ntax', 'nbase', 'Mode', 'Time'])
for ntax in np.arange(10, 51, 10):
    for nbase in np.arange(3, 31, 2):
        print(f'{ntax}, {nbase}')
        selected_data, selected_taxonomy = handler.data_selection(ntax, rank, nbase)
        tax_labels = selected_taxonomy['family'].to_numpy()
        data = selected_data.to_numpy()
        
        def jacknife_worker(query_idx):
            # this function replaces the one in the graboid library, its only argument is the query index, as it takes the reference data, labels and cost matrix from the current namespace
            query = data[query_idx]
            references = np.delete(data, query_idx, 0)
            labels = np.delete(tax_labels, query_idx)
            return graboid.classify2(query, references, M, k, labels)
        # serial test
        t0 = time.time()
        serial_classif = [jacknife_worker(i) for i in range(data.shape[0])]
        t1 = time.time()
        
        elapsed = t1-t0
        time_tab = time_tab.append({'ntax':ntax, 'nbase':nbase, 'Mode':'serial', 'Time':elapsed}, ignore_index=True)

        # parallel test
        t0 = time.time()
        with ProcessPoolExecutor() as executor:
            parallel_classif = [classif for classif in executor.map(jacknife_worker, range(data.shape[0]))]
        t1 = time.time()
        elapsed = t1 - t0
        time_tab = time_tab.append({'ntax':ntax, 'nbase':nbase, 'Mode':'parallel', 'Time':elapsed}, ignore_index=True)

#%% plot results
import matplotlib.pyplot as plt

# sort by ntax
sort_by = 'nbase'
x_var = 'ntax'
n = 15
serial = time_tab.loc[(time_tab[sort_by] == n) & (time_tab['Mode'] == 'serial'), 'Time'].to_numpy()
parallel = time_tab.loc[(time_tab[sort_by] == n) & (time_tab['Mode'] == 'parallel'), 'Time'].to_numpy()
X = time_tab.loc[(time_tab[sort_by] == n) & (time_tab['Mode'] == 'serial'), x_var].to_numpy()

fig, ax = plt.subplots()
ax.plot(X, serial, label = 'serial')
ax.plot(X, parallel, label = 'parallel')
ax.legend()
ax.set_xlabel(x_var)
ax.set_ylabel('time')