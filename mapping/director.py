#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:23:54 2022

@author: hernan
Direct dataset_construction
"""

#%% Libraries
from Bio.SeqIO.FastaIO import SimpleFastaParser as sfp
import logging
from mapping import blast
from mapping import matrix
import os

#%% set logger
logger = logging.getLogger('Graboid.mapper')
logger.setLevel(logging.DEBUG)

#%% functions
def make_dirs(base_dir):
    os.makedirs(f'{base_dir}/data', exist_ok=bool)
    os.makedirs(f'{base_dir}/warnings', exist_ok=bool)

def check_fasta(fasta_file):
    # checks the given file contains at least one fasta sequence
    nseqs = 0
    with open(fasta_file, 'r') as fasta_handle:
        for rec in sfp(fasta_handle):
            nseqs += 1
    return nseqs
#%% classes
class Director:
    def __init__(self, out_dir, warn_dir):
        self.out_dir = out_dir
        self.warn_dir = warn_dir
        
        # attributes
        self.db_dir = None
        self.blast_report = None
        self.mat_file = None
        self.acc_file = None
        self.dims = None
        
        # workers
        self.blaster = blast.Blaster(out_dir)
        self.mapper = matrix.MatBuilder(out_dir)
    
    @property
    def accs(self):
        return self.mapper.acclist
    def get_files(self, seq_file, seq_name=None):
        # use this to check if a map file already exists
        self.mapper.generate_outnames(seq_file, seq_name=None)
        return self.mapper.mat_file, self.mapper.acc_file
    
    def set_blastdb(self, db_dir):
        # establish db_dir as the blast database (must contain 6 *.n* files)
        check, db_files = blast.check_db_dir(db_dir)
        n_files = len(db_files)
        if not check:
            logger.error(f'Found {n_files} files in {db_dir}. Must contain 6')
            return
        self.db_dir = db_dir
        
    def build_blastdb(self, ref_seq, ref_name=None, clear=True):
        # build a blast database
        # check sequences in ref_seq
        n_refseqs = check_fasta(ref_seq)
        if n_refseqs != 1:
            logger.error(f'Reference file must contain ONE sequence. File {ref_seq} contains {n_refseqs}')
            return
        # build database directory
        if not ref_name is None:
            db_dir = f'{self.out_dir}/{ref_name}'
        else:
            ref = ref_seq.split('/')[-1].split('.')[0]
            db_dir = f'{self.out_dir}/{ref}'
        os.mkdir(db_dir)
        # build database
        self.blaster.make_ref_db(ref_seq, db_dir, clear)
        self.db_dir = db_dir
        
    def direct(self, fasta_file, out_name=None, evalue=0.005, threads=1, keep=False):
        # fasta file is the file to be mapped
        # out_name, optional file name for the generated matrix, otherwise generated automatically
        # evalue is the max evalue threshold for the blast report
        
        # build reference database (if not already present)
        if self.db_dir is None:
            logger.error('No BLAST database set. Set or create one with methods set_blastdb() or build_blastdb()')
            return
        
        print(f'Performing blast alignment of {fasta_file}...')
        # perform BLAST
        self.blaster.blast(fasta_file, self.db_dir, out_name, threads)
        self.blast_report = self.blaster.report
        if self.blast_report is None:
            logger.error('No blast report found. What happened?')
            return
        print('BLAST is Done!')
        
        # generate matrix
        print('Building alignment matrix...')
        # if keep == True, keep generated matrix, bounds and acclist in map_data, otherwise map_data is None
        map_data = self.mapper.build(self.blast_report, fasta_file, out_name, evalue, keep)
        self.mat_file = self.mapper.mat_file
        self.acc_file = self.mapper.acc_file
        print('Done!')
        if keep:
            self.matrix = map_data[0]
            self.bounds = map_data[1]
            self.acclist = map_data[2]
        return
