#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 11:38:29 2022

@author: hernan
Build an alignment matrix from the blast report
"""

#%% libraries
from Bio.SeqIO.FastaIO import SimpleFastaParser as sfp
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#%% set logger
logger = logging.getLogger('mapping_logger.Matrix')

#%% vars
bases = '-acgturykmswbdhvn'
tr_dict = {base:idx for idx, base in enumerate(bases)} | {base:idx for idx, base in enumerate(bases.upper())}
#%% functions
def make_transdict():
    translation_dict = {}
    translation_dict['lower'] = {base:idx for idx, base in enumerate(bases)}
    translation_dict['upper'] = {base:idx for idx, base in enumerate(bases.upper())}
    return translation_dict

# read blast file
def read_blast(blast_file, evalue = 0.005):
    colnames = 'qseqid pident length qstart qend sstart send evalue'.split(' ')
    blast_tab = pd.read_csv(blast_file,
                            sep = '\t',
                            header = None,
                            names = colnames)
    # blast_tab.rename(columns = columns, inplace = True)
    blast_tab = blast_tab.loc[blast_tab['evalue'] <= evalue]
    return blast_tab

# read seq_file
def read_seqfile(seq_file):
    seq_dict = {}
    with open(seq_file, 'r') as handle:
        for head, seq in sfp(handle):
            acc = head.split(' ')[0]
            seq_dict[acc] = seq
            
    return seq_dict

def get_seqs(self, seq_file, blast_tab):
    seq_dict = read_seqfile(seq_file)
    accs = blast_tab['qseqid'].unique()
    filtered_seqs = {acc:seq_dict[acc] for acc in accs}
    return filtered_seqs

def get_mat_dims(blast_tab):
    # get total dimensions for the sequence matrix
    nrows = len(blast_tab.index.unique())
    offset = blast_tab[['sstart', 'send']].min().min() - 1 # most times this value is 0
    upper = blast_tab[['sstart', 'send']].max().max()
    ncols = upper - offset
    return nrows, ncols, offset

# TODO: replace original build_coords
def build_coords0(coord_mat):
    # get the sorted coordinates for the given match
    # seq_coords tell what part of each sequence to take
    # mat_coords tell where the matches go in the alignment
    seq_coords_raw = coord_mat[:, :2]
    mat_coords_raw = coord_mat[:, 2:]
    seq_coords_ord = np.sort(seq_coords_raw, axis = 1)
    mat_coords_ord = np.sort(mat_coords_raw, axis = 1)
    order = np.argsort(seq_coords_ord[:,0])

    seq_coords = seq_coords_ord[order]
    mat_coords = mat_coords_ord[order]
    return seq_coords, mat_coords

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

# TODO: replace original build_row
def build_row0(seq, seq_coords, mat_coords, rowlen):
    row = np.zeros(rowlen, dtype = int)
    for seq_coor, mat_coor in zip(seq_coords, mat_coords):
        subseq = subseq = seq[seq_coor[0]:seq_coor[1]+1]
        numseq = np.array([tr_dict[base] for base in subseq], dtype = 'int64')
        row[mat_coor[0]:mat_coor[1]+1] = numseq
    return row

def build_row(acc, seq, seq_coords, mat_coords, rowlen, transdict):
    row = np.ones(rowlen, dtype = np.int64) * 16
    for seq_coor, mat_coor in zip(seq_coords, mat_coords):
        subseq = seq[seq_coor[0]:seq_coor[1]+1]
        numseq = np.array([transdict[base] for base in subseq], dtype = 'int64')
        row[mat_coor[0]:mat_coor[1]+1] = numseq    
    return row.astype(np.int64)

def plot_coverage_data(blast_file, evalue = 0.005, figsize=(12,7)):
    # TODO: save plot to file
    # get coverage matrix
    blast_tab = read_blast(blast_file, evalue)
    blast_tab.set_index('qseqid', inplace = True)
    extent = blast_tab[['sstart', 'send']].max().max()
    coords = []
    for acc, subtab in blast_tab.groupby('qseqid'):
        coord_mat = subtab[['sstart', 'send']].to_numpy().reshape((-1,2))
        coords.append(np.sort(coord_mat, axis=1))
    # plot
    x = np.arange(extent)
    fig, ax = plt.subplots(figsize = figsize)
    for idx, coor in enumerate(coords):
        cov = np.zeros(extent)
        cov[:] = np.nan
        for coo in coor:
            cov[coo[0]:coo[1]] = idx+1
        ax.plot(x, cov, c='r')
    
    ax.set_xlabel('Coordinates')
    ax.set_ylabel('Sequences')
    ax.set_title(f'Sequence coverage of {blast_file}')

#%% classes
class MatBuilder:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        
        self.dims = (0, 0)
        self.acclist = []
        self.mat_file = None
        self.acc_file = None
        
    def generate_outnames(self, seq_file, out_name=None):
        file_name = seq_file.split('/')[-1].split('.')[0]
        if not out_name is None:
            self.mat_file = f'{self.out_dir}/{out_name}.npy'
            self.acc_file = f'{self.out_dir}/{out_name}.acclist'
        else:
            self.mat_file = f'{self.out_dir}/{file_name}.npy'
            self.acc_file = f'{self.out_dir}/{file_name}.acclist'
        
    def build(self, blast_file, seq_file, out_name=None, evalue=0.005):
        # load blast report
        print('Reading blast report...')
        blast_tab = read_blast(blast_file, evalue)
        if len(blast_tab) == 0:
            logger.warning('No matches in the blast report {blast_file} passed the filter (evalue <= {evalue})')
            return
        mat_dims = get_mat_dims(blast_tab)
        self.dims = mat_dims[:2]
        offset = mat_dims[2]
        
        coord_mat = blast_tab[['qstart', 'qend', 'sstart', 'send']].to_numpy()
        coord_mat[:, 2:] -= offset
        
        # get filtered sequences
        print('Retrieving sequences...')
        sequences = get_seqs(seq_file, blast_tab)
        
        # build matrix
        matrix = np.zeros(self.dims, dtype=int)
        for accnum, (acc, seq) in enumerate(sequences.items()):
            coord_submat = coord_mat[blast_tab['qseqid'] == acc]
            # seq_coords, mat_coords = build_coords(subtab[['qstart', 'qend', 'sstart', 'send']], self.offset)
            seq_coords, mat_coords = build_coords0(coord_submat)
            row = build_row0(acc, seq, seq_coords, mat_coords, self.dims[1], self.transdict)
            
            matrix[accnum,:] = row
            self.acclist.append(acc)
        
        # generate out_files
        self.generate_outnames(seq_file, out_name)
        
        # store output
        np.save(self.mat_file, matrix, allow_pickle=False)
        with open(self.acc_file, 'w') as list_handle:
            list_handle.write('\n'.join(self.acclist))
