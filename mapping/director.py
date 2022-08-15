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
from data_fetch.dataset_construction import blast
from data_fetch.dataset_construction import matrix2
import os

#%% set logger
logger = logging.getLogger('mapping_logger')
logger.setLevel(logging.DEBUG)
# set formatter
fmtr = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')

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
#%% classes
class Director:
    def __init__(self, out_dir, warn_dir):
        self.out_dir = out_dir
        self.warn_dir = warn_dir
        
        # set handlers
        self.warn_handler = logging.FileHandler(warn_dir + '/dataset.warnings')
        self.warn_handler.setLevel(logging.WARNING)
        self.log_handler = logging.StreamHandler()
        self.log_handler.setLevel(logging.DEBUG)
        # create formatter
        self.warn_handler.setFormatter(fmtr)
        self.log_handler.setFormatter(fmtr)
        # add handlers
        logger.addHandler(self.warn_handler)
        logger.addHandler(self.log_handler)
        
        # attributes
        self.db_dir = None
        self.blast_report = None
        self.mat_file = None
        self.acc_file = None
        self.dims = None
        
        # workers
        self.blaster = blast.Blaster()
        self.mapper = matrix2.MatBuilder(out_dir)
        
    def set_blastdb(self, db_dir):
        # establish db_dir as the blast database (must contain 6 *.n* files)
        check, db_files = blast.check_db_dir(db_dir)
        n_files = len(db_files)
        if not check:
            logger.warning(f'Found {n_files} files in {db_dir}. Must contain 6')
            return
        self.db_dir = db_dir
        
    def build_blastdb(self, ref_seq, ref_name=None, clear=True):
        # build a blast database
        # check sequences in ref_seq
        n_refseqs = check_fasta(ref_seq)
        if n_refseqs != 1:
            logger.warning(f'Reference file must contain ONE sequence. File {ref_seq} contains {n_refseqs}')
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
        print(f'Generated blast reference databse at directory {self.db_dir}')
        
    def direct(self, fasta_file, out_name=None, evalue=0.005, threads=1):
        # fasta file is the file to be mapped
        # out_name, optional file name for the generated matrix, otherwise generated automatically
        # evalue is the max evalue threshold for the blast report
        
        # build reference database (if not already present)
        if self.db_dir is None:
            logger.info('No BLAST database set. Set or create one with methods set_blastdb() or build_blastdb()')
            return
        
        # perform blast
        if not out_name is None:
            blast_out = f'{self.out_dir}/{out_name}.BLAST'
        else:
            blast_out = fasta_file.split('/')[-1].split('.')[0]
            blast_out = f'{self.out_dir}/{blast_out}.BLAST'
        
        print(f'Performing blast alignment of {fasta_file}...')
        # perform BLAST
        self.blaster.blast(fasta_file, blast_out, self.db_dir, threads)
        self.blast_report = self.blaster.report
        if self.blast_report is None:
            return
        
        print(f'Done! Blast report saved as {blast_out}')
        
        # generate matrix
        print('Building alignment matrix...')
        self.mapper.build(self.blast_report, fasta_file, out_name, evalue)
        self.mat_file = self.mapper.mat_file
        self.acc_file = self.mapper.acc_file
        print('Done!')
        print(f'Alignment matrix saved as {self.mapper.mat_file}')
        print(f'Accession list saved as {self.mapper.acc_file}')
        return        
