#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:23:54 2022

@author: hernan
Direct dataset_construction
"""

#%% Libraries
import logging
import os
from Bio.SeqIO.FastaIO import SimpleFastaParser as sfp
from glob import glob

from mapping import blast
from mapping import matrix

#%% set logger
map_logger = logging.getLogger('Graboid.mapper')
map_logger.setLevel(logging.INFO)

#%% aux functions
def check_fasta(fasta_file):
    # checks the given file contains at least one fasta sequence
    nseqs = 0
    with open(fasta_file, 'r') as fasta_handle:
        for title, seq in sfp(fasta_handle):
            nseqs += 1
    return nseqs

def get_header(fasta_file):
    # use this to extract the header of the reference fasta sequence
    with open(fasta_file, 'r') as fasta_handle:
        for title, seq in sfp(fasta_handle):
            return title
        
def check_ref(ref_file):
    nseqs = 0
    marker_len = 0
    with open(ref_file, 'r') as fasta_handle:
        for title, seq in sfp(fasta_handle):
            nseqs += 1
            marker_len = len(seq)
        if nseqs > 1:
            raise Exception(f'Reference file must contain ONE sequence. File {ref_file} contains {nseqs}')
    return marker_len
            
def build_blastdb(ref_seq, db_dir):
    """
    Build a blast database from a reference sequence.

    Parameters
    ----------
    ref_seq : str
        Path to the file to build the database from. Must contain a single sequence.
    db_dir : str
        Path to the directory to contain the generated database files.

    Returns
    -------
    db_name : str
        Path and prefix of the generated database files.
    ref_len : int
        Length of the reference sequence.
    ref_header : str
        Header of the reference sequence.

    """
    # build a blast database
    # ref_seq : fasta file containing the reference sequences
    
    db_name = f'{db_dir}/db'
    
    # check that reference file is valid
    ref_len = check_ref(ref_seq)
    ref_header = get_header(ref_seq)
    
    # build the blast database
    blast.makeblastdb(ref_seq, db_name)
    
    return db_name, ref_len, ref_header

#%% classes
class Director:
    def __init__(self, out_dir, warn_dir, logger=map_logger):
        # directories
        self.out_dir = out_dir
        self.warn_dir = warn_dir
        
        # attributes
        self.db_dir = None
        
        # workers
        self.blaster = blast.Blaster(out_dir)
        self.mapper = matrix.MatBuilder(out_dir)
        
        self.logger = logger
    @property
    def accs(self):
        return self.mapper.acclist
    @property
    def blast_report(self):
        return self.blaster.report
    @property
    def mat_file(self):
        return self.mapper.mat_file
    @property
    def acc_file(self):
        return self.mapper.acc_file
    @property
    def matrix(self):
        return self.mapper.matrix
    @property
    def bounds(self):
        return self.mapper.bounds
    @property
    def coverage(self):
        return self.mapper.coverage
    @property
    def mesas(self):
        return self.mapper.mesas
        
    def direct(self, fasta_file, db_dir, evalue=0.005, dropoff=0.05, min_height=0.1, min_width=2, threads=1, keep=True):
        # fasta file is the file to be mapped
        # evalue is the max evalue threshold for the blast report
        # db_dir points to the blast database: should be <path to db files>/<db prefix>
        
        self.db_dir = db_dir
        
        print('Performing blast alignment of retrieved sequences against reference sequence...')
        # perform BLAST
        try:
            self.blaster.blast(fasta_file, db_dir, threads)
        except Exception:
            raise
        print('BLAST is Done!')
        
        # generate matrix, register mesas
        print('Building alignment matrix...')
        try:
            self.mapper.build(self.blast_report, fasta_file, evalue, dropoff, min_height, min_width, keep)
        except Exception as excp:
            self.logger.error(excp)
        print('Done!')
        self.logger.info(f'Stored alignment matrix of dimensions {self.matrix.shape} in {self.mat_file}')
        self.logger.info(f'Stored accession list with {len(self.accs)} records in {self.acc_file}')
        return
