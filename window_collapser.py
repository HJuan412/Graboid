# -*- coding: utf-8 -*-
"""
Spyder Editor

This scripc collapses redundant sequences in a file
"""

#%% libraries
import matrix_toolkit as mtk
import numpy as np
import pandas as pd
#%% functions
def get_val_idx(array):
    # returns a dictionary of the form {value:[idx of rows with value]} for each unique value in the array
    array_vals = set(array)
    indexes = {val:[] for val in array_vals}
    
    for idx, val in enumerate(array):
        indexes[val].append(idx)

    return indexes

def get_uniq_seqs(matrix, row_idx = None, col_idx = 0):
    if row_idx is None:
        row_idx = np.arange(matrix.shape[0])
    # stop conditions
    # one row remaining or last column reached
    if len(row_idx) == 1 or col_idx == matrix.shape[1]:
        seq = tuple(matrix[row_idx[0]])
        indexes = tuple(row_idx)
        return {seq:indexes}
    
    sub_mat = matrix[row_idx]
    uniq_vals = get_val_idx(sub_mat[:, col_idx])
    uniq_seqs = {}

    for val_idx in uniq_vals.values():
        row_idx1 = row_idx[val_idx] # esto es necesario
        # get unique sequences in the sub_mat
        uniq_seqs.update(get_uniq_seqs(matrix, row_idx1, col_idx + 1))

    return uniq_seqs

#%% classes
class WindowCollapser():
    def __init__(self, tax_file, rank = 'family'):
        self.tax_file = tax_file
        self.rank = rank
        self.mloader = mtk.MatrixLoader()
        
        self.set_matrix()
        self.collapsed = pd.DataFrame()
        self.mixed_taxes = {}
    def set_matrix(self, mat_file = None):
        # load the data matrix to be used
        self.mat_file = mat_file
        self.mloader.clear()
        self.mloader.load(mat_file)
        # MatrixLoader object contains matrix, accession list, n, start and end
        self.__set_tax()
    
    def collapse(self):
        if not self.mat_file is None:
            self.__collapse_windows()
            self.__get_repeated_seqs()
            self.__get_unique_seqs()
            self.__get_collapsed()

    # these methods are private because they should only be used by other methods of the class
    def __set_tax(self):
        if self.mloader.matrix is None:
            self.diff = None
            self.tax_tab = None
        else:
            tax_tab = pd.read_csv(self.tax_file, index_col = 0, sep = '\t')
            
            # these three lines make sure we don't ask for missing records in tax_tab
            acc_set = set(self.mloader.accs)
            index_set = set(tax_tab.index)
            intersect = acc_set.intersection(index_set)
            
            self.diff = acc_set.difference(index_set) # stores entries present in the window accessions but not in the tax_table
            self.tax_tab = tax_tab.loc[intersect]

    def __collapse_windows(self):
        # create dictionary of the form {(sequence):(instance_idxs)}
        self.uniq_dict = get_uniq_seqs(self.mloader.matrix)
    
    def __get_repeated_seqs(self):
        # checks if the instances sharing the same sequence are of the same taxon
        # for sequences with mixed taxons, generate new pseudo taxons (stored in the mixed_taxes dictionary)
        
        repeated = {k:v for k,v in self.uniq_dict.items() if len(v) != 1} # contains all sequences with more than one instance in the windows
        mixed = {}
        pure = {}
        
        mixed_taxes = {}
        n_mixed = 0
        for k,v in repeated.items():
            accs = {self.mloader.accs[i] for i in v}
            taxons = tuple(self.tax_tab.loc[accs, self.rank].unique())
            
            if len(taxons) > 1:
                # mutliple taxons share this sequence
                # check if a pseduo taxon with the same taxon combination already exists (if not create a new one)
                if taxons in mixed_taxes.keys():
                    tax_id = mixed_taxes[taxons] # select the existing pseudo-tax id for the current sequence
                else:
                    # create a new pseudo taxon for the current taxon mix
                    tax_id = f'mix_{n_mixed}'
                    mixed_taxes[taxons] = tax_id
                    n_mixed += 1
                mixed[k] = tax_id, v
            else:
                # all instances sharing the sequence are of the same taxon (it is pure)
                pure[k] = taxons[0], v
        
        self.mixed = mixed
        self.pure = pure
        self.mixed_taxes = {v:k for k,v in mixed_taxes.items()}
    
    def __get_unique_seqs(self):
        unique_seqs = {k:v for k,v in self.uniq_dict.items() if len(v) == 1}
        unique = {}
        for k,v in unique_seqs.items():
            acc = self.mloader.accs[v[0]]
            tax = self.tax_tab.loc[acc, self.rank]
            unique[k] = tax, v
        
        self.unique = unique
    
    def __get_collapsed(self):
        collapsed_mixed = {(k, v[0], v[1]) for k,v in self.mixed.items()}
        collapsed_pure = {(k, v[0], v[1]) for k,v in self.pure.items()}
        collapsed_unique = {(k, v[0], v[1]) for k,v in self.unique.items()}
        
        union = collapsed_mixed.union(collapsed_pure).union(collapsed_unique)
        self.collapsed = pd.DataFrame(union, columns = ['seq', 'tax', 'id'])
