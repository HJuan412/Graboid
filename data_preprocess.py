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
    def __init__(self, alnfile):
        self.load_aln(alnfile)
        self.aln_to_array()
        self.make_trans_dict()
        self.aln_to_numeric()
    
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

class alignment_handler():
    def __init__(self, alnfile, taxfile):
        # loader object used to get the alignment data in
        # numeric form, stores matrix in a dataframe (using short accession as
        # index) to facilitate joint manipulation with taxonomy table.
        loader = alignment_loader(alnfile)
        self.alignment_tab = pd.DataFrame(index = loader.acc_list, data = loader.numeric_aln)
        # loads taxonomy table (converts to int) with short accessions as index
        tax_tab = pd.read_csv(taxfile, index_col = 'ACC short')
        self.taxonomy_tab = tax_tab.iloc[:,1:].fillna(0).astype(int)
        # get intersection of alignment and taxonomy tables
        aln_idx = set(self.alignment_tab.index)
        tax_idx = set(tax_tab.index)
        self.acc_in = aln_idx.intersection(tax_idx) # accessions both in alignment and taxonomy table

        self.clear_selection()        

    def clear_selection(self):
        # resets selection
        self.selected_acc = self.acc_in
        self.selected_data = self.alignment_tab.loc[self.acc_in]
        self.selected_taxonomy = self.taxonomy_tab.loc[self.acc_in]
    
    def set_parameters(self, ntax, rank, nbase):
        # set selection parameters, called from data_selection
        self.ntax = ntax
        self.rank = rank
        self.nbase = nbase

    def select_taxons(self):
        # select the ntax most populated taxons, called from data_selection
        tax_count = self.selected_taxonomy[self.rank].value_counts(ascending = False)
        if 0 in tax_count.index:
            tax_count.drop(0, inplace = True)
        self.selected_taxons = list(tax_count.iloc[:self.ntax].index)
        self.selected_accs = self.select_inds(self.selected_taxons)
    
    def select_inds(self, tax_list):
        # get a list of selected individuals (those whose taxon at the set rank are in the given tax_list), called from data_selection
        selected_inds = list(self.selected_taxonomy.loc[self.selected_taxonomy[self.rank].isin(tax_list)].index)
        return selected_inds

    def select_bases(self):
        # select the nbase most informative bases, called from data_selection, AFTER selecting taxons and individuals
        general_entropy = entropy(self.selected_data.loc[self.selected_accs].to_numpy()) # entropy of each base for the given matrix
        per_taxon_entropy = np.zeros((self.ntax, self.selected_data.shape[1])) # entropy for each base for each of the selected taxons
            
        # calculate entropy within each taxon
        for idx, taxon in enumerate(self.selected_taxons):
            orgs = self.select_inds([taxon])
            per_taxon_entropy[idx] = entropy(self.selected_data.loc[orgs].to_numpy())
        
        ############### TESTING OTHER BLOCK
        # # get entropy differences intra/inter taxon for each base, sort in each taxon
        # entropy_diff = per_taxon_entropy - general_entropy
        # diff_sort = np.argsort(entropy_diff, 1)
        
        # # select the nbase most informative columns in each taxon
        # # as the selected columns may not be the same for each taxon, the
        # # final vector is usually larger than nbase
        # self.selected_base_idx = np.unique(diff_sort[:,:self.nbase]) # store the column indexes of the selected bases
        ###############

        # get entropy differences intra/inter taxon for each base, sort in each taxon
        gain = general_entropy - per_taxon_entropy # we want per_taxon_entropy to be smaller than general entropy (bigger is better)
        gain_sort = np.argsort(gain, 1) # ascending sort, we want the largest values
        
        # select the nbase most informative columns in each taxon
        # as the selected columns may not be the same for each taxon, the
        # final vector is usually larger than nbase
        
        #TODO define which is best, last nbase or first nbase
        # self.selected_base_idx = np.unique(gain_sort[:,-self.nbase:]) # store the column indexes of the selected bases
        self.selected_base_idx = np.unique(gain_sort[:,:self.nbase])

    def data_selection(self, ntax, rank, nbase):
        """
        Crops the data matrix to select the n most populated taxons and the m
        most informative bases for each of them.

        Parameters
        ----------
        ntax : int
            Number of taxons to keep.
        rank : str
            Taxonomic rank to filter by.
        nbase : int
            Number of informative bases to keep per taxon.
        -------
        data_selected : pandas.DataFrame
            Cropped data matrix, containing the individuals belonging to the
            filtered taxons and the informative bases selected.
        taxonomy_selected : pandas.DataFrame
            Taxonomic codes for the selected individuals.

        """
        self.clear_selection()
        self.set_parameters(ntax, rank, nbase)
        self.select_taxons()
        self.select_bases()
        
        data_selected = self.selected_data.loc[self.selected_accs, self.selected_base_idx]
        taxonomy_selected = self.selected_taxonomy.loc[self.selected_accs]
        return data_selected, taxonomy_selected
    
    def list_ranks(self):
        # Display taxonomic ranks available to filter by
        return list(self.taxonomy_tab.columns)


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
    import time
    
    t0 = time.time()
    handler = alignment_handler(alnfile, taxfile)
    for i in range(100):
        handler.select_data(20, 3, 5)
    t1 = time.time()
    print(f'handler done in {t1 - t0}')