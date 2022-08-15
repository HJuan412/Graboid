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
bases = 'nacgturykmswbdhv'
tr_dict = {base:idx for idx, base in enumerate(bases)} | {base:idx for idx, base in enumerate(bases.upper())}
tr_dict['-'] = 0
#%% functions
def make_transdict():
    translation_dict = {}
    translation_dict['lower'] = {base:idx for idx, base in enumerate(bases)}
    translation_dict['upper'] = {base:idx for idx, base in enumerate(bases.upper())}
    return translation_dict

# read blast file
def read_blast(blast_file, evalue = 0.005):
    # read blast file, flip reversed matches, sort by qseqid and qstart, subtract 1 from qstart and sstart to adjust for python indexing
    colnames = 'qseqid pident length qstart qend sstart send evalue'.split(' ')
    blast_tab = pd.read_csv(blast_file,
                            sep = '\t',
                            header = None,
                            names = colnames)
    # blast_tab.rename(columns = columns, inplace = True)
    blast_tab = blast_tab.loc[blast_tab['evalue'] <= evalue]
    qmat = blast_tab[['qstart', 'qend']].values
    smat = blast_tab[['sstart', 'send']].values
    
    # flip inverted matches
    qmat = np.sort(qmat, 1)
    smat = np.sort(smat, 1)
    # adjust for python indexing
    qmat[:,0] -= 1
    smat[:,0] -= 1
    # update values
    blast_tab[['qstart', 'qend']] = qmat
    blast_tab[['sstart', 'send']] = smat
    blast_tab.sort_values(['qseqid', 'qstart'], ignore_index=True, inplace=True)
    return blast_tab

# read seq_file
def read_seqfile(seq_file):
    seq_dict = {}
    with open(seq_file, 'r') as handle:
        for head, seq in sfp(handle):
            acc = head.split(' ')[0]
            seq_dict[acc] = seq
            
    return seq_dict

def get_seqs(seq_file, blast_tab):
    # get matched sequences and convert to numerical values
    accs = blast_tab.qseqid.unique()
    seq_dict = read_seqfile(seq_file)
    filtered_seqs = {}
    for acc in accs:
        filtered_seqs[acc] = np.array([tr_dict[base] for base in seq_dict[acc]], dtype = np.int8)
    return filtered_seqs

def get_mat_dims(blast_tab):
    # get total dimensions for the sequence matrix
    nrows = len(blast_tab.qseqid.unique())
    lower = blast_tab.sstart.min() # used to calculate the offset, most times this value is 0
    upper = blast_tab.send.max()
    ncols = upper - lower
    return nrows, ncols, lower, upper

def make_guide(values):
    # buils a guide to the position of the matches for each sequence in the blast report
    # used to speed up matrix construction
    # returns list of lists [acc, idx of first match, idx of last match]
    accs, idxs0 = np.unique(values, return_index = True)
    order = np.argsort(idxs0)
    accs = accs[order]
    idxs0 = idxs0[order]
    idxs1 = np.append(idxs0[1:], len(values))
    guide = [[acc, idx0, idx1] for acc, idx0, idx1 in zip(accs, idxs0, idxs1)]
    return guide

# def build_coords(coord_mat):
#     # get the sorted coordinates for the given match
#     # seq_coords tell what part of each sequence to take
#     # mat_coords tell where the matches go in the alignment
#     seq_coords_raw = coord_mat[:, :2]
#     mat_coords_raw = coord_mat[:, 2:]
#     seq_coords_ord = np.sort(seq_coords_raw, axis = 1)
#     mat_coords_ord = np.sort(mat_coords_raw, axis = 1)
#     order = np.argsort(seq_coords_ord[:,0])

#     seq_coords = seq_coords_ord[order]
#     mat_coords = mat_coords_ord[order]
#     # substract 1 from first column to account for python indexing
#     seq_coords[:,0] -= 1
#     mat_coords[:,0] -= 1
#     return seq_coords, mat_coords

# # TODO: replace original build_row
# def build_row0(seq, seq_coords, mat_coords, rowlen):
#     row = np.zeros(rowlen, dtype = np.int8)
#     for seq_coor, mat_coor in zip(seq_coords, mat_coords):
#         # subseq = seq[seq_coor[0]:seq_coor[1]]
#         # numseq = np.array([tr_dict[base] for base in subseq], dtype = 'int64')
#         # row[mat_coor[0]:mat_coor[1]] = numseq
#         row[mat_coor[0]:mat_coor[1]] = seq[seq_coor[0]:seq_coor[1]]
#     return row

# def build_row(acc, seq, seq_coords, mat_coords, rowlen, transdict):
#     row = np.ones(rowlen, dtype = np.int64) * 16
#     for seq_coor, mat_coor in zip(seq_coords, mat_coords):
#         subseq = seq[seq_coor[0]:seq_coor[1]+1]
#         numseq = np.array([transdict[base] for base in subseq], dtype = 'int64')
#         row[mat_coor[0]:mat_coor[1]+1] = numseq    
#     return row.astype(np.int64)

# def build_query_window(query_blast, seq_file, w_start=None, w_end=None, w_dict=None):
#     # TODO: need to keep track of the offset of the reference alignment matrix
#     blast_tab = read_blast(query_blast)
    
#     # get aligned sequences
#     sequences = get_seqs(seq_file, blast_tab)
#     rows = []
#     acclist = []
#     for qid, subtab in blast_tab.groupby('qseqid'):
#         coord_mat = subtab[['qstart', 'qend', 'sstart', 'send']].to_numpy()
#         # coord_mat[:, 2:] -= offset
#         seq_coords, mat_coords = build_coords(coord_mat)
#         rowlen = mat_coords[:, 3].max()
#         row = build_row0(sequences[qid], seq_coords, mat_coords, rowlen)
#         rows.append(row)
#         acclist.append(qid)
    
#     if not w_dict is None:
#         rows = np.array([rows[w_dict[acc][0]:w_dict[acc][1] + 1] for acc in acclist])
#     else:
#         rows = np.array(rows)
#         rows = rows[:, w_start:w_end + 1]
#     return np.array(rows), acclist

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
        
        self.acclist = []
        self.mat_file = None
        self.acc_file = None
        
    def generate_outnames(self, seq_file, out_name=None):
        file_name = seq_file.split('/')[-1].split('.')[0]
        if not out_name is None:
            self.mat_file = f'{self.out_dir}/{out_name}.npz'
            self.acc_file = f'{self.out_dir}/{out_name}.acclist'
        else:
            self.mat_file = f'{self.out_dir}/{file_name}.npz'
            self.acc_file = f'{self.out_dir}/{file_name}.acclist'
        
    def build(self, blast_file, seq_file, out_name=None, evalue=0.005):
        # load blast report
        print('Reading blast report...')
        blast_tab = read_blast(blast_file, evalue)
        if len(blast_tab) == 0:
            logger.warning('No matches in the blast report {blast_file} passed the filter (evalue <= {evalue})')
            return
        
        guide = make_guide(blast_tab.qseqid.values)
        # get dimensions
        mat_dims = get_mat_dims(blast_tab)
        nrows = mat_dims[0]
        ncols = mat_dims[1]
        bounds = np.array([mat_dims[2], mat_dims[3]])
        offset = mat_dims[2]
        
        coord_mat = blast_tab[['qstart', 'qend', 'sstart', 'send']].to_numpy()
        coord_mat[:, 2:] -= offset
        
        # get filtered sequences
        print('Retrieving sequences...')
        sequences = get_seqs(seq_file, blast_tab)
        
        # build matrix
        print('Building matrix...')
        matrix = np.zeros((nrows, ncols), dtype=np.int8)
        for accnum, (acc, idx0, idx1) in enumerate(guide):
            seq = sequences[acc]
            coord_submat = coord_mat[idx0:idx1]            
            for coord in coord_submat:
                matrix[accnum, coord[2]:coord[3]] = seq[coord[0]:coord[1]]
            self.acclist.append(acc)
        
        # generate out_files
        self.generate_outnames(seq_file, out_name)
        
        # store output
        # np.save(self.mat_file, matrix, allow_pickle=False)
        np.savez(self.mat_file, bounds=bounds, matrix=matrix) # save the matrix along with the bounds array
        with open(self.acc_file, 'w') as list_handle:
            list_handle.write('\n'.join(self.acclist))