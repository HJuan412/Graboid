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
    def __init__(self, tax_file):
        # tax_file should be a file located in the Taxonomy_files dir in a downloaded database, name should be <taxon>_<marker>_tax.tsv
        self.tax_file = tax_file
        self.mloader = mtk.MatrixLoader()
        self.set_matrix()

    def set_matrix(self, mat_file = None):
        # load the data matrix to be used, if mat_file is None clears existing matrix
        self.mat_file = mat_file
        self.mloader.clear()
        self.mloader.load(mat_file)
        # MatrixLoader object contains matrix, accession list, n, start and end
        self.__set_tax()
    
    def collapse(self):
        if not self.mat_file is None:
            self.__collapse_windows()
            self.__get_unique_repeated()
            self.__unpack()
            self.__uniq_taxes()
            self.__mixd_taxes()
            self.__build_tax_tab()

    # these methods are private because they should only be used by other methods of the class
    def __set_tax(self):
        if self.mloader.matrix is None:
            # no matrix loaded, clear records
            self.diff = None
            self.tax_tab = None
        else:
            # load taxonomy table and extract relevant entries
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
    
    def __get_unique_repeated(self):
        # sorts repeated and unique sequences
        repeated = {}
        unique = {}
        for k,v in self.uniq_dict.items():
            if len(v) != 1:
                repeated[k] = v # contains all sequences with more than one instance
            else:
                unique[k] = v
        self.repeated = repeated
        self.unique = unique
    
    def __unpack(self):
        # extracts the data matrix and generates a report of instances belonging to each sequence
        seq_list = []
        inst_list = []
        for k,v in self.unique.items():
            seq_list.append(k)
            inst_list.append([list(v), 'unique'])
        for k,v in self.repeated.items():
            seq_list.append(k)
            inst_list.append([list(v), 'repeated'])
        
        self.matrix = np.array(seq_list)
        self.report = pd.DataFrame(inst_list, columns = ['instances', 'status'])
        self.__get_accs()
    
    def __get_accs(self):
        # add accession codes to the report
        acc_list = []
        for _, item in self.report['instances'].iteritems():
            accs = self.tax_tab.index[item].tolist()
            acc_list.append(accs)
        self.report['accs'] = pd.Series(acc_list)

    def __uniq_taxes(self):
        # extract taxonomic codes of unique sequences
        idx = [v[0] for v in self.unique.values()]
        self.taxes_unique = self.tax_tab.iloc[idx,1:].reset_index(drop = True)
    
    def __mixd_taxes(self):
        # exrtract taxonomic codes of repeated sequences
        repeated = self.report.loc[self.report['status'] == 'repeated', 'instances']
        taxes_repeated = pd.DataFrame(index = repeated.index, columns = self.taxes_unique.columns)
        for idx, instances in repeated.iteritems():
            prev_col = 0
            for col in taxes_repeated.columns:
                taxes = self.tax_tab[col].iloc[instances].unique()
                if len(taxes) == 1:
                    prev_col = taxes[0]
                taxes_repeated.at[idx, col] = prev_col
        self.taxes_repeated = taxes_repeated

    def __build_tax_tab(self):
        self.tax_collapsed = pd.concat([self.taxes_unique, self.taxes_repeated])
#%% test - delete
tax_file = '/home/hernan/PROYECTOS/Graboid/Databases/12_11_2021-23_15_53/Taxonomy_files/Nematoda_18S_tax.tsv'
mat_file = 'Dataset/12_11_2021-23_15_53/Matrices/Nematoda/18S/17_aln_256-355_n12260.mat'

wc = WindowCollapser(tax_file)
wc.set_matrix(mat_file)
wc.collapse()
