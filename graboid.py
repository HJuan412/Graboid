#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hernán Juan

Taxonomy classification of alignment sequences using KNN method
"""

#%% modules
from multiprocessing import Pool, Array, RawArray, Value
from numba import njit
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
    return np.array(full_dist, dtype = np.float64)

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

@njit
def get_sorted_neighs(query, references, matrix):
    # returns the indexes of the neighbouring references, sorted by distances
    # also returns distances
    n_refs = references.shape[0]
    distances = np.zeros(n_refs)
    
    for idx, ref in enumerate(references):
        distances[idx] = get_dist(query, ref, matrix)
    
    dist_sort = np.argsort(distances)
    return dist_sort, distances[dist_sort]

def classify(query, references, matrix, k, labels):
    nearest_neighs, distances = get_k_neighs(query, references, matrix, k)

    neigh_labels = np.unique(labels[nearest_neighs], return_counts = True)
    winner_idx = np.argsort(neigh_labels[1])[-1]
    classification = neigh_labels[0][winner_idx]
    return classification

def classify2(query, references, matrix, k, labels):
    neighs, distances = get_sorted_neighs(query, references, matrix)
    nearest_neighs = neighs[:k]

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
if __name__ == '__main__':
    import time
    import data_preprocess as dp
    
    alnfile = 'Test_data/nem_18S_win_100_step_16/nem-18S_win_272-372_max-gap_10.fasta'
    taxfile = '/home/hernan/PROYECTOS/Graboid/Taxonomy/Taxonomies.tab'
    
    handler = dp.alignment_handler(alnfile, taxfile)
    
    M = cost_matrix()
#%%

    rank = 'family'
    
    ntax_range = np.arange(10, 41, 5)
    nbase_range = np.arange(5, 21, 3)
    times_record = []
    
    # for nbase in nbase_range:
        # print(nbase)
    ntax = 20
    nbase = 10
    
    # select data
    selected_data, selected_taxonomy = handler.data_selection(ntax, rank, nbase)
    
    k = 5
    # #%
    # t0 = time.time()
    # classifications = jacknife_classify(selected_data.to_numpy(), M, k, selected_taxonomy['family'].to_numpy())
    # t1 = time.time()
    # times_record.append(t1 - t0)
    
    data_mat = selected_data.to_numpy()
    tax_mat = selected_taxonomy.to_numpy()
    query = data_mat[0]
    query_lab = tax_mat[0]
    ref = data_mat[1:]
    ref_labs = tax_mat[1:]
    
    t0 = time.time()
    nn, dist = get_k_neighs(query, ref, M, k)
    t1 = time.time()
    
    elapsed = t1 - t0
    #%%
    import matplotlib.pyplot as plt
    plt.plot(nbase_range, times_record)
#%% MOVE to director


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