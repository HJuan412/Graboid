#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hernán Juan

Taxonomy classification of alignment sequences using KNN method
"""

#%% modules
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from multiprocessing import Pool, Array, RawArray, Value
from numba import njit
import numpy as np
import pandas as pd
import time

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
                   # 'u':'t',
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
                   'n':['a', 'c', 'g', 't']}
    
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
    return np.array(full_dist, dtype = np.float32)

@njit
def get_dist(query, reference, matrix):
    # TODO: dist actual es suma del vector, que tal longitud del vector (sqrt((dist_array**2).sum()))
    # calculates the distance of 2 sequences of equal length (query & reference) using the given cost matrix
    bases = np.arange(len(query))
    dist_array = np.array([matrix[query[idx], reference[idx]] for idx in bases])
    dist = dist_array.sum()
    return dist

@njit
def get_k_neighs(query, references, matrix, k):
    n_refs = references.shape[0]
    distances = np.zeros(n_refs)
    
    for idx, ref in enumerate(references):
        distances[idx] = get_dist(query, ref, matrix)
    
    nearest_neighs = np.argsort(distances)[:k]
    return nearest_neighs, distances

def classify(query, references, matrix, k, labels):
    nearest_neighs, distances = get_k_neighs(query, references, matrix, k)

    neigh_labels = np.unique(labels[nearest_neighs], return_counts = True)
    winner_idx = np.argsort(neigh_labels[1])[-1]
    classification = neigh_labels[0][winner_idx]
    return classification

def jacknife_classify(data, matrix, k, labels):
    classifications = []
    for query_idx in range(data.shape[0]):
        data2 = np.delete(data, query_idx, 0)
        labels2 = np.delete(labels, query_idx)
        query = data[query_idx]
        classifications.append(classify(query, data2, matrix, k, labels2))
    return classifications

#%% main
import data_preprocess as dp

alnfile = 'Test_data/nem_18S_win_100_step_16/nem-18S_win_272-372_max-gap_10.fasta'
taxfile = '/home/hernan/PROYECTOS/Graboid/Taxonomy/Taxonomies.tab'

handler = dp.alignment_handler(alnfile, taxfile)

M = cost_matrix()

rank = 'family'
ntax = 20
nbase = 10

# select data
selected_data, selected_taxonomy = handler.data_selection(ntax, rank, nbase)

#%%
query = selected_data.iloc[0].to_numpy()
query_lab = selected_taxonomy.iloc[0,3]
references = selected_data.iloc[1:].to_numpy()
labels = selected_taxonomy.iloc[1:, 3].to_numpy()
k = 5
#%%
classifications = jacknife_classify(selected_data.to_numpy(), M, k, selected_taxonomy['family'].to_numpy())

#%%mets
classif = np.array(classifications)
taxons = handler.selected_taxons
n_taxons = len(taxons)
true_labels = selected_taxonomy.iloc[:, 3].to_numpy()
total_inds = len(true_labels)

# wrong one
confusion_mat = np.zeros((len(taxons), 4)) # TP, TN, FP, FN

for idx, tax in enumerate(taxons):
    # true pos
    tp_idx = np.where(true_labels == tax)[0]
    tn_idx = np.where(true_labels != tax)[0]
    
    true_pos = len(np.where(classif[tp_idx] == tax)[0])
    true_neg = len(np.where(classif[tn_idx] != tax)[0])
    false_pos = len(np.where(classif[tn_idx] == tax)[0])
    false_neg = len(np.where(classif[tp_idx] != tax)[0])
    
    confusion_mat[idx,:] = [true_pos, true_neg, false_pos, false_neg]

acc_mat = (confusion_mat[:,0] + confusion_mat[:,1]) / total_inds
prec_mat = (confusion_mat[:,0]) / (confusion_mat[:,0] + confusion_mat[:,2])
rec_mat = (confusion_mat[:,0]) / (confusion_mat[:,0] + confusion_mat[:,3])

confusion = np.zeros((n_taxons, n_taxons))

for idx0, tax0 in enumerate(taxons):
    actual_idx = np.where(true_labels == tax0)[0]
    predicted_values = classif[actual_idx]
    for idx1, tax1 in enumerate(taxons):
        n_predicted = len(np.where(predicted_values == tax1)[0])
        confusion[idx0, idx1] = n_predicted
total_true = confusion.sum(1)
total_predicted = confusion.sum(0)

# metrics
accuracy = np.zeros(n_taxons)
precision = np.zeros(n_taxons)
recall = np.zeros(n_taxons)

for idx, tax in enumerate(taxons):
    true_pos = confusion[idx, idx]
    true_neg = np.delete(np.delete(confusion, idx, 0), idx, 1).sum()
    false_pos = np.delete(confusion[:,idx], idx).sum()
    false_neg = np.delete(confusion[idx], idx).sum()
    
    accuracy[idx] = (true_pos + true_neg) / total_inds
    precision[idx] = true_pos / (true_pos + false_pos)
    recall[idx] = true_pos / (true_pos + false_neg)

f1 = 2 * (precision * recall) / (precision + recall)
#%% MOVE to director
# var_dict = {}

# def init_worker(data, data_shape, M, M_shape, label_data, k):
#     # this is executed at the start of each process, stores the data matrix and its shape in global variables
#     var_dict['data'] = data
#     var_dict['data_shape'] = data_shape
#     var_dict['cost_mat'] = M
#     var_dict['mat_shape'] = M_shape
#     var_dict['labels'] = label_data
#     var_dict['K'] = k

# def jacknife_worker(query_idx):
#     data = np.frombuffer(var_dict['data']).reshape(var_dict['data_shape']).astype(int)
#     labels = np.array(var_dict['labels']).astype(int)
#     M = np.frombuffer(var_dict['cost_mat']).reshape(var_dict['mat_shape']).astype(float)
#     K = var_dict['K']
    
#     data2 = np.delete(data, query_idx, 0)
#     query = data[query_idx]
#     classification = classify(query, data2, M, K, labels)
#     return query_idx, classification

# #jacknife
# if __name__ == '__main__':
#     tables = []
#     start = time.time()

#     parallel = False # toggle parallel processing

#     if parallel:
#         for aln_idx, aln_fl in enumerate([aln_files[0]]):
#             print(f'Alginment n {aln_idx}')
#             M = cost_matrix()
        
#             pack = data_pack(aln_fl)
#             pack.select_taxons(20, 'family')
#             pack.select_bases(5)
            
#             data = pack.crop_data()
#             data_shape = data.shape
    
#             if data_shape[0] == 1:
#                 print(f'Not enough records {data_shape[0]}')
#                 continue
#             elif data_shape[1] == 1:
#                 print(f'Not enough bases {data_shape[1]}')
#                 continue
    
#             X = RawArray('d', data_shape[0] * data_shape[1]) # this array will be used to share the data between the processes (without having to copy the matrix each time)
#             M_data = RawArray('d', M.shape[0] * M.shape[1]) # this will contain the cost matrix
#             label_data = Array('d', pack.reference_labels['num'].to_numpy(dtype = np.int32))
#             k_neighs = Value('i', 5)
        
#             M_np = np.frombuffer(M_data, dtype = np.float64).reshape(M.shape)
#             np.copyto(M_np, M)
#             X_np = np.frombuffer(X, dtype = np.float64).reshape(data.shape) # these lines paste the content from data to the RawArray X
#             np.copyto(X_np, data)
            
#             with Pool(processes = 3, initializer = init_worker, initargs=(X, data.shape, M_data, M.shape, label_data, 5)) as pool:
#                 result = pool.map(jacknife_worker, range(len(data)), chunksize = 1500)
#             pack.reference_labels['clasif'] = np.array(result)[:,1]
#             tables.append(pack.reference_labels)
#     else:
#         data_mats = []
#         label_mats = []
#         for aln_fl in [aln_files[0]]:
#             pack = data_pack(aln_fl)
#             pack.select_taxons(20, 'family')
#             pack.select_bases(5)
            
#             data = pack.crop_data()
#             data_shape = data.shape
    
#             if data_shape[0] == 1:
#                 print(f'Not enough records {data_shape[0]}')
#                 continue
#             elif data_shape[1] == 1:
#                 print(f'Not enough bases {data_shape[1]}')
#                 continue
#             data_mats.append(data)
#             label_mats.append(pack.reference_labels['num'].to_numpy())
        
#         iterables = [(dmat, M, 5, lmat) for dmat, lmat in zip(data_mats, label_mats)]
#         with Pool(processes = 3) as pool:
#             result = pool.starmap(jacknife_classify, iterables)

#     end = time.time()
#     total_time = end - start