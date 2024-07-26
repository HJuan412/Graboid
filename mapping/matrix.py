#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 11:38:29 2022

@author: hernan
Build an alignment matrix from the blast report
"""

#%% libraries
from Bio.SeqIO.FastaIO import SimpleFastaParser as sfp
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re

# graboid modules
from . import blast
#%% vars
bases = 'nacgt'
r_bases = 'ntgca'
special_chars = '-rykmswbdhv'
special = {char:0 for char in special_chars + special_chars.upper()}
tr_dict = {base:idx for idx, base in enumerate(bases)} | {base:idx for idx, base in enumerate(bases.upper())}
tr_dict.update({'u':4, 'U':4})
tr_dict.update(special)
# compliment dict, used to translate revese complement sequences
rc_dict = {base:idx for idx, base in enumerate(r_bases)} | {base:idx for idx, base in enumerate(r_bases.upper())}
rc_dict.update({'u':1, 'U':1})
rc_dict.update(special)
seq2num = {1:tr_dict, -1:rc_dict}
#%% functions
# read blast file
# TODO: replace read_blast with read_blast0
def read_blast0(blast_file, evalue = 0.005):
    # read blast file, register match orientations, flip reversed matches, sort by qseqid and qstart, subtract 1 from qstart and sstart to adjust for python indexing
    # blast file preformatted (comma separated), contains column names and last row with qseqid = Reference, length = reference marker length
    # returns processed blast table and used marker length
    blast_tab = pd.read_csv(blast_file)
    
    # check for empty report
    if len(blast_tab) == 0:
        raise Exception(f'Blast report {blast_file} is empty. Verify that the blast parameters are correct.')
    
    ref_len = blast_tab.iloc[-1].loc['length']
    # filter blast report for evalue
    blast_tab = blast_tab.query('evalue <= @evalue').copy()
    if len(blast_tab) == 0:
        raise Exception(f'No matches in the blast report {blast_file} passed the filter (evalue <= {evalue})')
        
    # process match coordinates
    # match orientations: 1 = Forward, -1 = Reverse
    blast_tab['Orient'] = 1
    blast_tab.loc[blast_tab.sstart > blast_tab.send, 'Orient'] = -1
    # flip inverted matches
    # return blast_tab
    blast_tab[['sstart', 'send']] = np.sort(blast_tab[['sstart', 'send']], axis=1)
    # adjust for python indexing (shift 1 position to the left)
    blast_tab.qstart -= 1
    blast_tab.sstart -= 1
    blast_tab.sort_values(['qseqid', 'qstart'], ignore_index=True, inplace=True)
    return blast_tab, ref_len

def read_blast(blast_file, evalue = 0.005):
    # read blast file, register match orientations, flip reversed matches, sort by qseqid and qstart, subtract 1 from qstart and sstart to adjust for python indexing
    # blast file preformatted (comma separated), contains column names and last row with qseqid = Reference, length = reference marker length
    # returns processed blast table and used marker length
    blast_tab = pd.read_csv(blast_file)
    marker_len = 0
    
    # check for empty report
    if len(blast_tab) == 0:
        raise Exception(f'Blast report {blast_file} is empty. Verify that the blast parameters are correct.')

    marker_len = blast_tab.iloc[-1].loc['length']
    
    # filter blast report for evalue
    blast_tab = blast_tab.loc[blast_tab['evalue'] <= evalue]
    if len(blast_tab) == 0:
        raise Exception(f'No matches in the blast report {blast_file} passed the filter (evalue <= {evalue})')
        
    # process match coordinates
    subject_coords = blast_tab[['sstart', 'send']].values    
    # get orients (0: forward, 1:reverse)
    orients = (subject_coords[:,0] > subject_coords[:,1]).reshape((-1,1)).astype(int)
    # flip inverted matches
    subject_coords = np.sort(subject_coords, 1)
    # adjust for python indexing (shift 1 position to the left)
    blast_tab.qstart -= 1
    subject_coords[:,0] -= 1
    # update values
    blast_tab[['sstart', 'send']] = subject_coords
    blast_tab['orient'] = orients
    blast_tab.sort_values(['qseqid', 'qstart'], ignore_index=True, inplace=True)
    return blast_tab, marker_len

def get_mat_dims(blast_tab):
    # get total dimensions for the sequence matrix
    nrows = len(blast_tab.qseqid.unique())
    lower = blast_tab.sstart.min()
    upper = blast_tab.send.max()
    return nrows, lower, upper

def read_seqfile(seq_file):
    seq_dict = {}
    with open(seq_file, 'r') as handle:
        for head, seq in sfp(handle):
            acc = head.split(' ')[0]
            seq_dict[acc] = seq
            
    return seq_dict

def get_numseq(seq, trans_dict):
    # turn sequence to numeric code
    numseq = np.array([trans_dict[base] for base in seq], dtype = np.int8)
    return numseq

def load_map(map_dir):
    # load a map and a corresponding accession list
    # returns accession list, alignment map, alignment bounds
    map_file = glob(map_dir + '*__map.npz')[0]
    acc_file = glob(map_dir + '*__map.accs')[0]
    map_npz = np.load(map_file)
    with open(acc_file, 'r') as acc_handle:
        accs = acc_handle.read().splitlines()
    
    return accs, map_npz['matrix'], map_npz['bounds'], map_npz['coverage']

def get_coverage(coords, ref_len):
    coverage = np.zeros(ref_len, dtype=np.int32)
    for coo in coords:
        coverage[coo[0]:coo[1]] += 1
    return coverage

def get_mesas(coverage, dropoff=0.05, min_height=0.1, min_width=2):
    # returns an array with columns [mesa start, mesa end, mesa width, mesa average height]
    # dropoff : percentage of a mesa's max height to determine dropoff threshold (mesa's edge)
    # min_width : minimum width of a mesa
    # if min_height is a percentage of the max coverage, calculate and replace
    if isinstance(min_height, float):
        min_height = max(coverage) * min_height
    # edge values of 0 inserted into cov to indicate the ends of the alignment
    cov = np.insert([.0,.0], 1, coverage.astype(float))
    diffs = np.diff(cov).astype(float)
    mesas = []
    # get index of highest location and its height
    top = np.argmax(cov)
    height = coverage[top]
    while height > min_height:
        drop_threshold = height * dropoff
        # get upper bound
        # first crossing of the threshold AFTER top (at least 1 position to the right of top)
        upper = top + np.argmax(abs(diffs[top:]) >= drop_threshold)
        upper = min(upper, len(coverage))
        # get lower bound
        # last crossing of the threshold BEFORE upper
        lower = upper - np.argmax(abs(diffs[:upper][::-1]) >= drop_threshold) - 1
        lower = max(lower, 0)
        mean_height = coverage[lower:upper].mean()
        mesas.append([lower, upper, mean_height])
        # replace mesa values from coverage & diff arrays with -inf so it is never chosen as top and always breaks a window
        # displaced 1 to the right to account for leading value inserted
        cov[lower + 1:upper + 1] = -np.inf
        diffs[lower + 1:upper + 1] = -np.inf
        top = np.argmax(cov)
        height = cov[top]
    
    mesas = np.array(mesas)
    # filter messas by min_width & sort by position, add widths
    mesas = np.insert(mesas, 2, np.diff(mesas[:,:2]).flatten(), 1)
    mesas = mesas[mesas[:,2] > min_width]
    mesas = mesas[np.argsort(mesas[:,0]).flatten()]
    return mesas
#%%
def plot_coverage_data(blast_file, out_file=None, evalue = 0.005, figsize=(12,7)):
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
    if out_file:
        # store coverage plot
        pass

#%% classes
def build_map(seq_file, blast_db, prefix, evalue=0.005, threads=1):    
    """
    Align the given sequence file against a reference file using BLAST. Output
    alignment as a numberic matrix (npz file).
    Translation code:
        0 : Missing data (gaps, N values, ambiguous characters)
        1 : A
        2 : C
        3 : G
        4 : T/U

    Parameters
    ----------
    seq_file : str
        Sequence file to be aligned, in fasta format.
    blast_db : str
        Blast database to be used in the alignment (path + prefix).
    prefix : str
        Common name to the generated files (path + prefix).
    evalue : float, optional
        Evalue threshold for the blast alignment. The default is 0.005.
    threads : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    matrix_file : str
        Generated alignment file (prefix + "__map.npz").
        Contans 3 arrays:
            matrix : alignment array (2D array)
            bounds : 2 element array indicating the alignment bounds (1D array)
            coverage : array containing sequence coverage per position of the reference sequence (1D array)
    acc_file : str
        Accession list of sequences included in the alignment.
    nrows : int
        Number of rows present in the alignment matrix.
    ncols : int
        Number of colmns present in the alignment matrix.

    """
    blast_out = f'{prefix}.blast'
    matrix_file = f'{prefix}__map.npz'
    acc_file = f'{prefix}__map.acc'
    
    # perform blast
    blast.blast(seq_file, blast_db, blast_out, threads)
    blast_tab, ref_len = read_blast0(blast_out, evalue)
    
    # get dimensions
    nrows, lower, upper = get_mat_dims(blast_tab)
    ncols = upper - lower
    
    print('Retrieving sequences...')
    sequences = read_seqfile(seq_file)
    
    print('Building matrix...')
    acclist = []
    matrix = np.zeros((nrows, ref_len), dtype=np.int8)
    
    for idx, (acc, subtab) in enumerate(blast_tab.groupby('qseqid')):
        acclist.append(acc)
        seq = sequences[acc]
        for _, match in subtab.iterrows():
            numseq = get_numseq(seq[match.qstart:match.qend][::match.Orient], seq2num[match.Orient])
            matrix[idx, match.sstart:match.send] = numseq
    
    # calculate coverage
    coverage = (matrix > 0).sum(axis=0)
    
    # clip matrix
    matrix = matrix
    
    # store output
    # save the matrix along with the bounds and coverage array (coverage array done over the entire length of the marker reference)
    np.savez_compressed(matrix_file, bounds=np.array([lower, upper]), matrix=matrix, coverage=coverage)
    with open(acc_file, 'w') as list_handle:
        list_handle.write('\n'.join(acclist))
    return matrix_file, acc_file, nrows, ncols

class MatBuilder:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.mat_file = None
        self.acc_file = None
        
        self.matrix = None
        self.bounds = None
        self.coverage = None
        self.acclist = None
        
    def build(self, blast_file, seq_file, evalue=0.005, dropoff=0.05, min_height=0.1, min_width=2, keep=True):
        # generate out names
        out_name = re.sub('.*/', self.out_dir + '/', re.sub('\..*', '__map', seq_file))
        self.mat_file = out_name + '.npz'
        self.acc_file = out_name + '.accs'
        # load blast report
        print('Reading blast report...')
        try:
            blast_tab, marker_len = read_blast(blast_file, evalue)
            self.marker_len = marker_len
        except Exception as excp:
            raise excp
        
        # get dimensions & coverage
        nrows, ncols, lower, upper = get_mat_dims(blast_tab)
        bounds = np.array([lower, upper])
        offset = lower # lower value already shifted 1 to the left by read_blast
        
        # get coordinates
        coord_mat = blast_tab[['qstart', 'qend', 'sstart', 'send', 'orient']].to_numpy().astype(int)
        # get coverage
        coverage = get_coverage(coord_mat[:, [2,3]], marker_len)
        mesas = get_mesas(coverage, dropoff, min_height, min_width)
        # offset subject coordinates (substract the lower bound of the subject coordinates)
        # columns 0 and 1 are the coordinates to extract from the query sequene
        # columns 2 and 3 are their position on the matrix (corrected with the offset)
        # column 4 indicates the sequence orientation
        coord_mat[:, [2,3]] -= offset
        
        # get filtered sequences
        print('Retrieving sequences...')
        sequences = read_seqfile(seq_file)
        
        # build matrix
        print('Building matrix...')
        acclist = []
        matrix = np.zeros((nrows, ncols), dtype=np.int8)
        for q_idx, (qry, qry_tab) in enumerate(blast_tab.groupby('qseqid')):
            # get acc and indexes for each sequence
            seq = sequences[qry]
            coords = coord_mat[qry_tab.index]
            for match in coords:
                # if orient (row 4) is 0, get original sequence, else get reversed complemened
                if match[4] == 0:
                    numseq = get_numseq(seq[match[0]:match[1]], tr_dict)
                else:
                    numseq = get_numseq(seq[match[0]:match[1]][::-1], rc_dict)
                matrix[q_idx, match[2]:match[3]] = numseq
            acclist.append(qry)
        self.acclist = acclist
        # store output
        # save the matrix along with the bounds and coverage array (coverage array done over the entire length of the marker reference)
        np.savez_compressed(self.mat_file, bounds=bounds, matrix=matrix, coverage=coverage, mesas=mesas)
        with open(self.acc_file, 'w') as list_handle:
            list_handle.write('\n'.join(self.acclist))
        
        if keep:
            self.matrix = matrix
            self.bounds = bounds
            self.coverage = coverage
            self.mesas = mesas
            self.acclist = acclist
        return
