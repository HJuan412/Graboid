#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 16:41:37 2024

@author: hernan
Contains all functions needed to build build the alignment matrix
"""

from Bio.SeqIO.FastaIO import SimpleFastaParser as sfp
import numpy as np
import pandas as pd
import re
import subprocess

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

#%% blast functions

## Guide sequence database
def get_header(fasta_file):
    # use this to extract the header of the reference fasta sequence
    with open(fasta_file, 'r') as fasta_handle:
        for title, seq in sfp(fasta_handle):
            return title

def check_fasta(fasta_file):
    # checks the given file contains at least one fasta sequence
    nseqs = 0
    with open(fasta_file, 'r') as fasta_handle:
        for title, seq in sfp(fasta_handle):
            nseqs += 1
    return nseqs

def check_ref(ref_file):
    nseqs = check_fasta(ref_file)
    if nseqs > 1:
        raise Exception(f'Reference file must contain ONE sequence. File {ref_file} contains {nseqs}')

def makeblastdb(guide_file, db_prefix):
    # check that reference file is valid
    check_ref(guide_file)
    guide_header = get_header(guide_file)
    
    # build the reference BLAST database
    cline = f'makeblastdb -in {guide_file} -dbtype nucl -parse_seqids -input_type fasta -out {db_prefix}'.split()
    subprocess.run(cline, capture_output=True)
    return guide_header

## Blast search & results parsing
def blast(seq_file, guide_db, out_file, threads=1):
    # perform ungapped blast
    outfmt=['6 qseqid pident length qstart qend sstart send evalue']
    cline = f'blastn -task blastn -db {guide_db} -query {seq_file} -out {out_file} -ungapped -num_threads {threads} -outfmt'.split() + outfmt
    subprocess.run(cline, capture_output=True)
    
    blast_tab = pd.read_csv(out_file, sep='\t', header=None, names='qseqid pident length qstart qend sstart send evalue'.split())
    if len(blast_tab) == 0:
        raise Exception(f'Blast search of file {seq_file} on against guide sequence {guide_db} yielded no results')
    
    # overwrite blast report with column names
    blast_tab.to_csv(out_file, index=False)
    return

def read_blast(blast_file, evalue = 0.005):
    # read blast file, register match orientations, flip reversed matches, sort by qseqid and qstart, subtract 1 from qstart and sstart to adjust for python indexing
    # blast file preformatted (comma separated), contains column names and last row with qseqid = Reference, length = reference marker length
    # returns processed blast table and used marker length
    blast_tab = pd.read_csv(blast_file)
    
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
    return blast_tab

#%%
def read_seqfile(seq_file):
    seq_dict = {}
    with open(seq_file, 'r') as handle:
        for head, seq in sfp(handle):
            acc = head.split(' ')[0]
            seq_dict[acc] = seq
            
    return seq_dict

def get_guide_len(guide_db):
    # retrieve reference marker data
    bdbcmd_cline = f'blastdbcmd -db {guide_db} -dbtype nucl -entry all -outfmt %l'.split()
    guide_len = int(re.sub('\\n', '', subprocess.run(bdbcmd_cline, capture_output=True).stdout.decode()))
    return guide_len

def get_numseq(seq, trans_dict):
    # turn sequence to numeric code
    numseq = np.array([trans_dict[base] for base in seq], dtype = np.int8)
    return numseq

def build_map(seq_file, blast_db, prefix, evalue=0.005, threads=1, clip=False):    
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
    threads : int, optional
        Number of threads to be used in the blast search. The default is 1.
    clip : bool, optional
        Store only the covered columns of the alignment matrix to reduce file size. The default is False.

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
    blast(seq_file, blast_db, blast_out, threads)
    blast_tab = read_blast(blast_out, evalue)
    
    # get dimensions
    lower = blast_tab.sstart.min()
    upper = blast_tab.send.max()
    nrows = len(blast_tab.qseqid.unique())
    ncols = upper - lower
    marker_len = get_guide_len(blast_db) # length of the guide sequence is taken as the marker's length
    
    print('Retrieving sequences...')
    sequences = read_seqfile(seq_file)
    
    print('Building matrix...')
    acclist = []
    matrix = np.zeros((nrows, marker_len), dtype=np.int8)
    
    for idx, (acc, subtab) in enumerate(blast_tab.groupby('qseqid')):
        acclist.append(acc)
        seq = sequences[acc]
        for _, match in subtab.iterrows():
            numseq = get_numseq(seq[match.qstart:match.qend][::match.Orient], seq2num[match.Orient])
            matrix[idx, match.sstart:match.send] = numseq
    
    # calculate coverage
    coverage = (matrix > 0).sum(axis=0)
    
    # clip matrix
    if clip:
        matrix = matrix[:,lower:upper+1]
    
    # store output
    # save the matrix along with the bounds and coverage array (coverage array done over the entire length of the marker reference)
    np.savez_compressed(matrix_file, bounds=np.array([lower, upper]), matrix=matrix, coverage=coverage, marker_len=marker_len)
    with open(acc_file, 'w') as list_handle:
        list_handle.write('\n'.join(acclist))
    return matrix_file, acc_file, nrows, ncols