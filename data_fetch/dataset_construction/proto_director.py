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
import os
import ref_catalog
import blast
import matrix2

#%% set logger
logger = logging.getLogger('mapping_logger')
logger.setLevel(logging.DEBUG)
# set formatter
fmtr = logging.Formatter('%(asctime) - %(levelname)s: %(message)s')

#%% functions
def check_in_dir(dirname, prefix):
    # check that all corresponding files are present
    suffixes = ['.fasta', '.tax', '.taxguide', '.taxid']
    exp_files = {sfx:None for sfx in suffixes}
    warnings = []
    for sfx in suffixes:
        filename = f'{dirname}/{prefix}{sfx}'
        if os.path.isfile(filename):
            exp_files[sfx] = filename
        else:
            warnings.append(f'WARNING: file {filename} not found in {dirname}')
    return exp_files['.fasta'], warnings

def get_ref_dir(marker):
    ref_path = ref_catalog.catalog[marker.upper()]
    return ref_path

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
        self.warn_handler = logging.FileHandler(warn_dir)
        self.warn_handler.setLevel(logging.WARNING)
        self.log_handler = logging.StreamHandler()
        self.log_handler.setLevel(logging.DEBUG)
        # create formatter
        self.warn_handler.setFormatter(fmtr)
        self.log_handler.setFormatter(fmtr)
        # add handlers
        logger.addHandler(self.warn_handler)
        logger.addHandler(self.log_handler)
        
        self.__check_in_dir()
        self.__get_ref_dir()
        
        # TODO: set workers here
        self.db_dir = None
        self.blast_report = None
        self.blaster = blast.Blaster()
    
    def __check_in_dir(self):
        # check that all corresponding files are present
        self.seq_file = None
        suffixes = ['.fasta', '.tax', '.taxguide', '.taxid']
        exp_files = {sfx:None for sfx in suffixes}
        
        for sfx in suffixes:
            filename = f'{self.in_dir}/{self.taxon}_{self.marker}{sfx}'
            if os.path.isfile(filename):
                exp_files[sfx] = filename
            else:
                self.warnings.append(f'WARNING: file {filename} not found in {self.in_dir}')
        
        self.seq_file = exp_files['.fasta']
    
    def __get_ref_dir(self):
        # TODO handle custom references
        self.ref_path = None
        ref_path = ref_catalog.catalog[self.marker.upper()]
        self.ref_path = ref_path
    
    def set_workers(self):
        self.worker_blast = blast.Blaster(self.taxon, self.marker, self.seq_file, self.ref_path, self.out_dir, self.warn_dir)
        #TODO: fix BOLD seq files, blast hates the gaps ('-'). Fix it at the fetch or split step AHHHHHHH
        self.worker_matrix = matrix2.MatBuilder(self.taxon, self.marker, self.worker_blast.out_file, self.seq_file, self.out_dir, self.warn_dir)
    
    def build_dataset(self, threads = 1):
        print(f'Performing BLAST alignment of {self.taxon} {self.marker}')
        self.worker_blast.blast(threads)
        print(f'Building matrix of {self.taxon} {self.marker}')
        self.worker_matrix.build_matrix()
        print('Finished!')
    
    def set_dbdir(self, db_dir):
        check, db_files = blast.check_db_dir(db_dir)
        n_files = len(db_files)
        if not check:
            logger.warning(f'Found {n_files} files in {db_dir}. Must contain 6')
            return
        self.db_dir = db_dir
        
    def build_dbdir(self, ref_seq, ref_name=None):
        ref = ref_seq.split('/')[-1].split('.')[0]
        db_dir = f'{self.out_dir}/{ref}'
        if not ref_name is None:
            db_dir = f'{self.out_dir}/{ref_name}'
        self.db_dir = db_dir
        
    def direct(self, fasta_file, ref_seq=None, out_name=None, ref_name=None, evalue=0.005, clear=True, threads=1):
        # fasta file is the file to be mapped
        # out_name, optional file name for the generated matrix, otherwise generated automatically
        # ref_seq, reference sequence file, must contain ONE sequence
        # evalue is the max evalue threshold for the blast report
        
        # build reference database (if not already present)
        if self.db_dir is None:
            n_refseqs = check_fasta(ref_seq)
            if n_refseqs != 1:
                logger.warning(f'Reference file must contain ONE sequence. File {ref_seq} contains {n_refseqs}')
                return
            self.build_dbdir(ref_seq, ref_name)
            self.blaster.make_ref_db(ref_seq, self.db_dir, clear)
            print(f'Generated blast reference databse at directory {self.db_dir}')
        
        # perform blast
        blast_out = fasta_file.split('/')[-1].split('.')[0]
        blast_out = f'{self.out_dir}/{blast_out}.BLAST'
        if not out_name is None:
            blast_out = f'{self.out_dir}/{out_name}'
        
        print(f'Performing blast alignment of {fasta_file}...')
        # perform BLAST
        self.blast_report = self.blaster.blast(fasta_file, blast_out, threads)
        
        if self.blast_report is None:
            return
        
        print(f'Done! Blast report saved as {blast_out}')
        
        # generate matrix
        return
