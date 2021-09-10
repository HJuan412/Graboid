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

#%% paralell testing
var_dict = {}

def init_worker(data, M, label_data, k):
    # this is executed at the start of each process, stores the data matrix and its shape in global variables
    var_dict['data'] = data
    var_dict['data_shape'] = np.array(data.shape)
    var_dict['cost_mat'] = M.astype('float64')
    var_dict['mat_shape'] = np.array(M.shape)
    var_dict['labels'] = label_data
    var_dict['K'] = np.array(k)

def jacknife_worker(query_idx):
    data_shape = np.frombuffer(var_dict['data_shape'], dtype = int)
    data = np.frombuffer(var_dict['data'], dtype = int).reshape(data_shape)
    labels = np.array(var_dict['labels'], dtype = int)
    mat_shape = np.frombuffer(var_dict['mat_shape'], dtype = int)
    M = np.frombuffer(var_dict['cost_mat'], dtype = float).reshape(mat_shape)
    K = np.frombuffer(var_dict['K'], dtype = int)[0]
    
    data2 = np.delete(data, query_idx, 0)
    query = data[query_idx]
    classification = graboid.classify2(query, data2, M, K, labels)
    return query_idx, classification
#%%
qidx, classif = jacknife_worker(0)
