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
            
def build_blastdb(ref_seq, db_dir, clear=False, logger=map_logger):
    # build a blast database
    # ref_seq : fasta file containing the reference sequences
    # db_dir : directory to where the generated db files will be stored
    # clear : overwrite existing db files
    
    # check database directory
    try:
        db_name = blast.check_db_dir(db_dir)
        # base exists 
        if clear:
            logger.info(f'Overwriting database {db_name} using file {ref_seq}')
        else:
            logger.info('A blast database of the name {db_name} already exists in the specified route. To overwrite it run this function again with clear set as True')
            return
    except Exception:
        db_name = db_dir + '/db'
    # clear previous db_files (if present)
    for file in glob(db_dir + '/*.n'):
        os.remove(file)
    # check that reference file is valid
    marker_len = check_ref(ref_seq)    
    # build the blast database
    blast.makeblastdb(ref_seq, db_name)
    logger.info(f'Generated blast databse {db_name} using the file {ref_seq} in directory {db_dir}')
    return marker_len

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
        
        print(f'Performing blast alignment of {fasta_file}...')
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
        self.logger.info(f'Stored accession list with {len(self.accs)} in {self.acc_file}')
        return
