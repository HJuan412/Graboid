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
logger = logging.getLogger('Graboid.mapper.Matrix')

#%% vars
bases = 'nacgturykmswbdhv'
tr_dict = {base:idx for idx, base in enumerate(bases)} | {base:idx for idx, base in enumerate(bases.upper())}
tr_dict['-'] = 0
# compliment dict, used to translate revese complement sequences
rc_dict = tr_dict.copy()
rc_dict['a'] = 4
rc_dict['A'] = 4
rc_dict['c'] = 3
rc_dict['C'] = 3
rc_dict['g'] = 2
rc_dict['G'] = 2
rc_dict['t'] = 1
rc_dict['T'] = 1
rc_dict['u'] = 1
rc_dict['U'] = 1

#%% functions
def make_transdict():
    translation_dict = {}
    translation_dict['lower'] = {base:idx for idx, base in enumerate(bases)}
    translation_dict['upper'] = {base:idx for idx, base in enumerate(bases.upper())}
    return translation_dict

# read blast file
def read_blast(blast_file, evalue = 0.005):
    # read blast file, register match orientations, flip reversed matches, sort by qseqid and qstart, subtract 1 from qstart and sstart to adjust for python indexing
    colnames = 'qseqid pident length qstart qend sstart send evalue'.split(' ')
    blast_tab = pd.read_csv(blast_file,
                            sep = '\t',
                            header = None,
                            names = colnames)
    # blast_tab.rename(columns = columns, inplace = True)
    blast_tab = blast_tab.loc[blast_tab['evalue'] <= evalue]
    smat = blast_tab[['sstart', 'send']].values
    
    # get orients (0: forward, 1:reverse)
    orients = (smat[:,0] > smat[:,1]).reshape((-1,1)).astype(int)
    # flip inverted matches
    smat = np.sort(smat, 1)
    # adjust for python indexing
    blast_tab.qstart -= 1
    smat[:,0] -= 1
    # update values
    blast_tab[['sstart', 'send']] = smat
    blast_tab['orient'] = orients
    blast_tab.sort_values(['qseqid', 'qstart'], ignore_index=True, inplace=True)
    return blast_tab

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

def get_mat_dims(blast_tab):
    # get total dimensions for the sequence matrix
    nrows = len(blast_tab.qseqid.unique())
    subject_coords = blast_tab[['sstart', 'send']].to_numpy()
    subject_coords.sort(axis=1)
    lower = subject_coords[:,0].min()
    upper = subject_coords[:,1].max()
    # lower = blast_tab.sstart.min() # used to calculate the offset, most times this value is 0
    # upper = blast_tab.send.max()
    ncols = upper - lower
    return nrows, ncols, lower, upper

# read seq_file
def read_seqfile(seq_file):
    seq_dict = {}
    with open(seq_file, 'r') as handle:
        for head, seq in sfp(handle):
            acc = head.split(' ')[0]
            seq_dict[acc] = seq
            
    return seq_dict

# DEPRECATED
def get_seqs(seq_file, blast_tab):
    # get matched sequences and convert to numerical values
    accs = blast_tab.qseqid.unique()
    seq_dict = read_seqfile(seq_file)
    # filtered_seqs = {}
    # for acc in accs:
    #     filtered_seqs[acc] = np.array([tr_dict[base] for base in seq_dict[acc]], dtype = np.int8)
    filtered_seqs = {seq_dict[acc] for acc in accs}
    return filtered_seqs

def get_numseq(seq, trans_dict):
    # turn sequence to numeric code
    numseq = np.array([trans_dict[base] for base in seq], dtype = np.int8)
    return numseq

#%%
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
            self.acc_file = f'{self.out_dir}/{out_name}.accs'
        else:
            self.mat_file = f'{self.out_dir}/{file_name}.npz'
            self.acc_file = f'{self.out_dir}/{file_name}.accs'
        
    def build(self, blast_file, seq_file, out_name=None, evalue=0.005, keep=False):
        # load blast report
        print('Reading blast report...')
        blast_tab = read_blast(blast_file, evalue)
        if len(blast_tab) == 0:
            logger.warning(f'No matches in the blast report {blast_file} passed the filter (evalue <= {evalue})')
            return
        
        # get dimensions
        nrows, ncols, lower, upper = get_mat_dims(blast_tab)
        bounds = np.array([lower, upper])
        offset = lower # lower value already shifted 1 to the left by read_blast
        
        # get coordinates
        coord_mat = blast_tab[['qstart', 'qend', 'sstart', 'send', 'orient']].to_numpy()
        # offset subject coordinates (substract the lower bound of the match coordinates)
        coord_mat[:, [2,3]] -= offset
        
        # get filtered sequences
        print('Retrieving sequences...')
        # sequences = get_seqs(seq_file, blast_tab)
        sequences = read_seqfile(seq_file)
        
        # build matrix
        print('Building matrix...')
        matrix = np.zeros((nrows, ncols), dtype=np.int8)
        for q_idx, (qry, qry_tab) in enumerate(blast_tab.groupby('qseqid')):
            seq = sequences[qry]
            coords = coord_mat[qry_tab.index]
            for match in coords:
                # if orient (row 4) is 0, get original sequence, else get reversed
                if match[4] == 0:
                    numseq = get_numseq(seq[match[0]:match[1]], tr_dict)
                else:
                    numseq = get_numseq(seq[match[0]:match[1]][::-1], rc_dict)
                matrix[q_idx, match[2]:match[3]] = numseq
            self.acclist.append(qry)
        
        # generate out_files
        self.generate_outnames(seq_file, out_name)
        
        # store output
        # np.save(self.mat_file, matrix, allow_pickle=False)
        np.savez(self.mat_file, bounds=bounds, matrix=matrix) # save the matrix along with the bounds array
        with open(self.acc_file, 'w') as list_handle:
            list_handle.write('\n'.join(self.acclist))
        logger.info(f'Stored matrix of dimensions {matrix.shape} in {self.mat_file}')
        logger.info(f'Stored accession_list in {self.acc_file}')
        
        if keep:
            # use this to retrieve generating directly from this method
            return matrix, bounds, self.acclist
        return None