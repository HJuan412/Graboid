#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 11:38:29 2022

@author: hernan
Build an alignment matrix from the blast report
"""

#%% libraries
from Bio.SeqIO.FastaIO import SimpleFastaParser as sfp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
#%% vars
bases = 'acgturykmswbdhvn-'

#%% functions
def make_transdict():
    translation_dict = {}
    translation_dict['lower'] = {base:idx for idx, base in enumerate(bases)}
    translation_dict['upper'] = {base:idx for idx, base in enumerate(bases.upper())}
    return translation_dict

# read blast file
def read_blast(blast_file, evalue = 0.005):
    colnames = 'qseqid pident length qstart qend sstart send evalue'.split(' ')
    columns = {idx:col for idx,col in enumerate(colnames)}
    blast_tab = pd.read_csv(blast_file, sep = '\t', header = None, index_col = 0)
    blast_tab.rename(columns = columns, inplace = True)
    blast_tab = blast_tab.loc[blast_tab['evalue'] <= evalue]
    return blast_tab

# read seq_file
def read_seqfile(seq_file):
    seq_dict = {}
    acc_list = []
    with open(seq_file, 'r') as handle:
        for head, seq in sfp(handle):
            acc = head.split(' ')[0]
            seq_dict[acc] = seq
            acc_list.append(acc)
    return seq_dict, acc_list

def filter_acc_list(acc_list, blast_tab):
    filtered = [acc for acc in acc_list if acc in blast_tab.index]
    return filtered

def get_mat_dims(blast_tab):
    # get total dimensions for the sequence matrix
    nrows = len(blast_tab.index.unique())
    offset = blast_tab[['sstart', 'send']].min().min() - 1 # most times this value is 0
    upper = blast_tab[['sstart', 'send']].max().max()
    ncols = upper - offset
    return (nrows, ncols, offset)

def build_coords(subtab, offset):
    # get the sorted coordinates for the given match
    # seq_coords tell what part of each sequence to take
    # mat_coords tell where the matches go in the alignment
    raw_coords = np.reshape(subtab.to_numpy() - 1, (-1,4)).astype('int64')
    seq_coords_raw = raw_coords[:,:2]
    mat_coords_raw = raw_coords[:,2:]
    seq_coords_ord = np.sort(seq_coords_raw, axis = 1)
    mat_coords_ord = np.sort(mat_coords_raw, axis = 1)
    order = np.argsort(seq_coords_ord[:,0])
    
    seq_coords = seq_coords_ord[order]
    mat_coords = mat_coords_ord[order] - offset
    return seq_coords, mat_coords

def build_row(acc, seq, seq_coords, mat_coords, rowlen, transdict):
    row = np.ones(rowlen, dtype = np.int64) * 16
    for seq_coor, mat_coor in zip(seq_coords, mat_coords):
        subseq = seq[seq_coor[0]:seq_coor[1]+1]
        numseq = np.array([transdict[base] for base in subseq], dtype = 'int64')
        row[mat_coor[0]:mat_coor[1]+1] = numseq    
    return row.astype(np.int64)

# plotting functions
def get_coverage_data(in_file):
    blast_tab = read_blast(in_file)
    rows, cols, offset = get_mat_dims(blast_tab)
    coords = build_coords(blast_tab[['qstart', 'qend', 'sstart', 'send']], offset)[1]
    cov_data = np.zeros(cols)
    for coo in coords:
        cov_data[coo[0]:coo[1]] += 1
    return cov_data, cov_data / rows

def plot_coverage_data(taxon, marker, cov_data, mode = None):
    # mode 'perc' plot is given in percentage
    tot_data = cov_data[0]
    perc_data = cov_data[1]

    view = len(tot_data)
    xticklen = int(view/10)
    xticks = np.arange(0, view + 1, xticklen)
    xlabs = np.arange(0, len(tot_data)+1, xticklen)

    # generate plot information
    # per base coverage data
    cov = tot_data
    ylabel = 'Coverabe (bases)'
    if mode == 'perc':
        # use percentages instead
        cov = perc_data
        ylabel = 'Coverage (%)'

    x = np.arange(len(cov))

    # plot
    fig, ax = plt.subplots(figsize = (12,7))
    ax.margins(x=0.005, y = 0.01)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabs)
    ax.set_xlabel('Position')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{taxon}, {marker}\nPer base coverage')
    ax.plot(x, cov, color = 'r', label = 'Per base coverage')
    # ax.legend()
    return
#%% classes
class MatBuilder():
    def __init__(self, taxon, marker, blast_file, seq_file, out_dir, warn_dir):
        self.taxon = taxon
        self.marker = marker
        self.blast_file = blast_file
        self.seq_file = seq_file
        self.out_dir = out_dir
        self.warn_dir = warn_dir
        self.warnings = []
        self.out_file = f'{out_dir}/{taxon}_{marker}.npy'
        self.out_acclist = f'{out_dir}/{taxon}_{marker}.acclist'
        self.out_dims = f'{out_dir}/{taxon}_{marker}.dims'

    def __read_blast(self):
        blast_tab = read_blast(self.blast_file)
        self.blast_tab = blast_tab
        dims = get_mat_dims(blast_tab)
        self.dims = dims[:2]
        self.offset = dims[2]
    
    def __read_seqs(self):
        seq_dict, acc_list = read_seqfile(self.seq_file)
        filtered = filter_acc_list(acc_list, self.blast_tab)
        self.acc_list = filtered
        self.seq_dict = seq_dict
    
    def __make_transdict(self):
        td = make_transdict()
        test_seq = self.seq_dict[self.acc_list[0]][0]
        if test_seq in string.ascii_lowercase:
            self.transdict = td['lower'] # align is lowercase
        elif test_seq in string.ascii_uppercase:
            self.transdict = td['upper'] # align is uppercase
    
    def __setup_matrix(self):
        self.matrix = np.memmap(self.out_file, dtype = np.int64, mode = 'w+', shape = self.dims)
        # self.matrix = np.ones((self.dims), dtype = 'int64')
    
    def __save_metafiles(self):
        with open(self.out_acclist, 'w') as list_handle:
            list_handle.write('\n'.join(self.acc_list))
        with open(self.out_dims, 'w') as dims_handle:
            dims_handle.write(f'{self.dims[0]}\t{self.dims[1]}')

    def build_matrix(self):
        self.__read_blast()
        self.__read_seqs()
        self.__make_transdict()
        self.__setup_matrix()
        for accnum, acc in enumerate(self.acc_list):
            seq = self.seq_dict[acc]
            subtab = self.blast_tab.loc[acc]
            seq_coords, mat_coords = build_coords(subtab[['qstart', 'qend', 'sstart', 'send']], self.offset)
            row = build_row(acc, seq, seq_coords, mat_coords, self.dims[1], self.transdict)
            
            self.matrix[accnum,:] = row
        self.matrix.flush()
        self.__save_metafiles()
        