#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 13:47:10 2021

@author: hernan
Director
"""

#%% libraries
from Bio import AlignIO
import numpy as np
import pandas as pd
import toolkit as tools
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

def entropy(matrix):
    n_cols = matrix.shape[1]
    n_rows = matrix.shape[0]

    entropy = np.zeros(n_cols)
    for i in range(n_cols):
        value_freqs = np.unique(matrix[:,i], return_counts = True)[1] / n_rows
        entropy[i] = -np.sum(np.log2(value_freqs) * value_freqs)
        
    return (2-entropy) / 2
#%%
class alignment_loader():
    # Loads an alignment file in fasta format, converts it to a numpy array
    def __init__(self, alnfile, taxfile):
        self.load_aln(alnfile)
        self.aln_to_array()
        self.make_trans_dict()
        self.aln_to_numeric()
        self.get_taxonomy(taxfile)
    
    def load_aln(self, alnfile):
        #TODO: enable other formats. (jariola)
        with open(alnfile, 'r') as handle:
            self.alignment = AlignIO.read(handle, 'fasta')

    def aln_to_array(self):
        # self.acc_list = [seq.id for seq in self.alignment]
        self.acc_list = [seq.id.split('.')[0] for seq in self.alignment]
        self.aln_array = np.array([list(seq.seq) for seq in self.alignment])

    def make_trans_dict(self):
        self.translation_dict = {}

        case = get_case(self.aln_array[0])
        if case == 0:
            for idx, base in enumerate(bases):
                self.translation_dict[base] = idx
        elif case == 1:
            for idx, base in enumerate(bases.upper()):
                self.translation_dict[base] = idx

    def aln_to_numeric(self):        
        numeric_aln = np.zeros(self.aln_array.shape, dtype = int)
        
        for idx, aln in enumerate(self.aln_array):
            aln_vals = np.unique(aln)
            for val in aln_vals:                
                num_val = self.translation_dict[val]
                
                numeric_aln[idx] = np.where(aln == val, num_val, numeric_aln[idx])
        
        self.numeric_aln = numeric_aln
    
    def get_taxonomy(self, taxfile):
        taxonomy_df = pd.read_csv(taxfile, index_col = 'ACC short') # table containing taxonomies of all organisms in the database
        # get intersection of organisms in alignment and organisms in table
        acc_idx = set(taxonomy_df.index.to_list())
        acc_aln = set(self.acc_list)
        self.acc_in = list(acc_aln.intersection(acc_idx))
        # select taxonomy codes
        taxes_selected = taxonomy_df.loc[self.acc_in].fillna(0)
        self.taxonomy_codes = taxes_selected.iloc[:,1:].to_numpy(dtype = int)

class alignment_handler(alignment_loader):
    def __init__(self, alnfile, taxfile):
        alignment_loader.__init__(self, alnfile, taxfile)
        self.locate_seqs()
    
    def locate_seqs(self):
        # select only sequences that have a taxonomic code
        acc_array = np.array(self.acc_list)
        indexes = [np.where(acc_array == idx)[0][0] for idx in self.acc_in]
        self.selected = self.numeric_aln[indexes]
        self.selected_taxonomy = self.taxonomy_codes
        self.selected_accs = np.array(self.acc_in)
    
    def get_orgs(self, tax, rank):
        # get a subset of organisms belonging to the given taxon at the given rank
        tax_idx = self.get_inds_idx(tax, rank)
        return self.selected[tax_idx]
    
    def get_inds_idx(self, tax, rank):
        # get the index of all organism belonging to the given taxon at the given rank
        # tax can be a single entry or an array with multiple elements
        ind_idx = np.where(np.isin(self.selected_taxonomy[:,rank], tax))
        return ind_idx

    def select_data(self, ntaxes, rank, nbases):
        # generate the data matrix to be used in the graboid pipeline
        # select the ntaxes most populated taxes at the given rank
        # select the nbases most populated bases
        self.ntaxes = ntaxes
        self.rank = rank
        self.nbases = nbases
        
        # clean slate data matrix
        self.locate_seqs()
        
        # select taxes
        self.get_taxes()
        self.org_idx = self.get_inds_idx(self.selected_tax, self.rank)
        # self.selected = self.get_orgs(self.selected_tax, self.rank)
        self.selected = self.selected[self.org_idx]
        self.selected_taxonomy = self.selected_taxonomy[self.org_idx]
        self.selected_accs = self.selected_accs[self.org_idx]
        
        # select bases
        self.get_bases()
        self.selected = self.selected[:, self.selected_base_idx]
    
    def get_taxes(self):
        # generates list of the ntaxes most populated taxes
        # count each tax
        tax_count = np.array(np.unique(self.taxonomy_codes[:,self.rank], return_counts=True))
        # delete empty clade
        to_del = np.where(tax_count[0] <= 0)
        tax_count = np.delete(tax_count, to_del, axis = 1)
        # sort remaining counts in decreasing order and retrieve the first ntaxes
        sorted_count_idx = np.argsort(tax_count[1])[::-1]
        selected_tax_idx = sorted_count_idx[:self.ntaxes]
        self.selected_tax = tax_count[0][selected_tax_idx]

    def get_bases(self):
        general_entropy = entropy(self.selected) # entropy of each base for the given matrix
        per_taxon_entropy = np.zeros((self.ntaxes, self.selected.shape[1])) # entropy for each base for each of the selected taxons
            
        # calculate entropy within each taxon
        for idx, taxon in enumerate(self.selected_tax):
            orgs = self.get_orgs(taxon, self.rank)
            per_taxon_entropy[idx] = entropy(orgs)
        
        # get entropy differences intra/inter taxon
        entropy_diff = per_taxon_entropy - general_entropy
        diff_sort = np.argsort(entropy_diff, 1)
        
        self.selected_base_idx = np.unique(diff_sort[:,-self.nbases:]) # store the column indexes of the selected bases


#%% get best file
# TODO remove this shit
if __name__ == '__main__':
    datadir = 'Test_data/nem_18S_win_100_step_16'
    bacon = tools.filetab(datadir, '*fasta', '_', '.fasta')
    bacon.split_col('-', 2, ['wstart', 'wend'])
    bacon.set_coltype('wstart', int)
    bacon.set_coltype('wend', int)
    bacon.build_filetab(['wstart', 'wend', 'File'])
    
    report_df = pd.read_csv(f'{datadir}/Report_director.csv', index_col = 0)
    report_df.sort_values(by = 'Filtered seqs', ascending = False, inplace=True)
    align_number = report_df.iloc[0].name
    obj_wstart = align_number * 16
    obj_file = bacon.filetab.loc[bacon.filetab['wstart'] == obj_wstart].iloc[0,-1]
    
    #%% test loader
    alnfile = f'{datadir}/{obj_file}'
    taxfile = '/home/hernan/PROYECTOS/Graboid/Taxonomy/Taxonomies.tab'
    
    loader = alignment_loader(alnfile, taxfile)
    handler = alignment_handler(alnfile, taxfile)
    #%%    
class alignment_handler2():
    def __init__(self, alnfile, taxfile):
        loader = alignment_loader(alnfile, taxfile)
        self.alignment_tab = pd.DataFrame(index = loader.acc_list, data = loader.numeric_aln)
        tax_tab = pd.read_csv(taxfile, index_col = 'ACC short')
        self.taxonomy_tab = tax_tab.iloc[:,1:].fillna(0).astype(int)

        aln_idx = set(self.alignment_tab.index)
        tax_idx = set(tax_tab.index)
        self.acc_in = aln_idx.intersection(tax_idx) # accessions both in alignment and taxonomy table

        self.clear_selection()        

    def clear_selection(self):
        self.selected_acc = self.acc_in
        self.selected_data = self.alignment_tab.loc[self.acc_in]
        self.selected_taxonomy = self.taxonomy_tab.loc[self.acc_in]
    
    def set_parameters(self, ntax, rank, nbase):
        self.ntax = ntax
        self.rank = rank
        self.nbase = nbase

    def select_taxons(self):
        tax_count = self.selected_taxonomy[self.rank].value_counts(ascending = False)
        if 0 in tax_count.index:
            tax_count.drop(0, inplace = True)
        self.selected_taxons = list(tax_count.iloc[:self.ntax].index)
        self.selected_accs = self.select_inds(self.selected_taxons)
    
    def select_inds(self, tax_list):
        selected_inds = list(self.selected_taxonomy.loc[self.selected_taxonomy[self.rank].isin(tax_list)].index)
        return selected_inds

    def select_bases(self):
        general_entropy = entropy(self.selected_data.loc[self.selected_accs].to_numpy()) # entropy of each base for the given matrix
        per_taxon_entropy = np.zeros((self.ntax, self.selected_data.shape[1])) # entropy for each base for each of the selected taxons
            
        # calculate entropy within each taxon
        for idx, taxon in enumerate(self.selected_taxons):
            orgs = self.select_inds([taxon])
            per_taxon_entropy[idx] = entropy(self.selected_data.loc[orgs].to_numpy())
        
        # get entropy differences intra/inter taxon
        entropy_diff = per_taxon_entropy - general_entropy
        diff_sort = np.argsort(entropy_diff, 1)
        
        self.selected_base_idx = np.unique(diff_sort[:,-self.nbase:]) # store the column indexes of the selected bases

    def data_selection(self, ntax, rank, nbase):
        self.clear_selection()
        self.set_parameters(ntax, rank, nbase)
        self.select_taxons()
        self.select_bases()
        
        return self.selected_data.loc[self.selected_accs, self.selected_base_idx]

#%%
import time

t0 = time.time()
handler = alignment_handler(alnfile, taxfile)
for i in range(100):
    handler.select_data(20, 3, 5)
t1 = time.time()
print(f'handler done in {t1 - t0}')
t0 = time.time()
handler2 = alignment_handler2(alnfile, taxfile)
for i in range(100):
    data = handler2.data_selection(20, 'family', 5)
t1 = time.time()
print(f'handler2 done in {t1 - t0}')