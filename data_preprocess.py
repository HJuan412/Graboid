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

#%%
class alignment_loader():
    # Loads an alignment file in fasta format, converts it to a numpy array
    def __init__(self, filename):
        self.load_aln(filename)
        self.aln_to_array()
        self.make_trans_dict()
        self.aln_to_numeric()
    
    def load_aln(self, filename):
        #TODO: enable other formats
        with open(filename, 'r') as handle:
            self.alignment = AlignIO.read(handle, 'fasta')

    def aln_to_array(self):
        self.acc_list = [seq.id for seq in self.alignment]
        self.aln_array = np.array([list(seq.seq) for seq in alignment])

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
    
    #TODO: add taxonomy codes

class alignment_handler(alignment_loader):
    # TODO: this method will handle taxon and position filtering
    def clown(self):
        print(self.numeric_aln.shape)

#%% get best file
# TODO remove this shit
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

#%%
from Bio import AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

alnfile = f'{datadir}/{obj_file}'

with open(alnfile, 'r') as handle:
    alignment = AlignIO.read(handle, 'fasta')
