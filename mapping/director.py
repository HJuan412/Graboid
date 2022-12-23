#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:23:54 2022

@author: hernan
Direct dataset_construction
"""

#%% Libraries
import argparse
from Bio.SeqIO.FastaIO import SimpleFastaParser as sfp
import logging
from mapping import blast
from mapping import matrix
import os
import re

#%% set logger
logger = logging.getLogger('Graboid.mapper')
logger.setLevel(logging.DEBUG)

#%% arg parser
parser = argparse.ArgumentParser(prog='Graboid MAPPING',
                                 usage='%(prog)s ARGS [-h]',
                                 description='Graboid MAPPING aligns the downloaded sequences to a specified reference sequence. Alignment is stored as a numeric matrix with an accession list')
parser.add_argument('-f', '--fasta_file',
                    nargs='+',
                    help='Fasta file with the sequences to map',
                    type=str)
parser.add_argument('-r', '--ref_seq',
                    default=None,
                    help='Marker sequence to be used as base of the alignment',
                    type=str)
parser.add_argument('-rn', '--ref_name',
                    default=None,
                    help='OPTIONAL. Name for the generated BLAST database',
                    type=str)
parser.add_argument('-gb', '--gb_dir',
                    default=None,
                    help='Graboid working directory. Directory containing the generated files',
                    type=str)
parser.add_argument('-e', '--evalue',
                    default=0.005,
                    help='E-value threshold for the BLAST matches. Default: 0.005',
                    type=float)
parser.add_argument('-t', '--threads',
                    default=1,
                    help='Number of threads to be used in the BLAST alignment. Default: 1',
                    type=int)
parser.add_argument('-ow', '--overwrite',
                    action='store_true',
                    help='Overwrite existing files in case of collision')

#%% aux functions
def make_dirs(base_dir): # TODO: delete this?
    os.makedirs(f'{base_dir}/data', exist_ok=True)
    os.makedirs(f'{base_dir}/warnings', exist_ok=True)

def check_fasta(fasta_file):
    # checks the given file contains at least one fasta sequence
    nseqs = 0
    with open(fasta_file, 'r') as fasta_handle:
        for rec in sfp(fasta_handle):
            nseqs += 1
    return nseqs

def build_blastdb(ref_seq, ref_dir='.', ref_name=None, clear=False):
    # build a blast database
    # ref_seq : fasta file containing the reference sequences
    # ref_dir : directory to where the generated db files will be stored
    # ref_name : optional name for the generated db files
    # clear : overwrite existing db files
    
    # check database directory
    if ref_name is None:
        ref_name = re.sub('.*/', '', re.sub('\..*', '', ref_seq))
    db_dir = f'{ref_dir}/{ref_name}'
    db_out = f'{ref_dir}/{ref_name}/{ref_name}'
    
    check, db_files = blast.check_db_dir(db_out)
    if check:
        # base exists 
        if clear:
            logger.info(f'Overwriting database {ref_name} using file {ref_seq}')
        else:
            logger.info('A blast database of the name {ref_name} already exists in the specified route. To overwrite it run this function again with clear set as True')
            return db_out

    # clear previous db_files (if present)
    for file in db_files:
        os.remove(file)

    # check sequences in ref_seq
    n_refseqs = check_fasta(ref_seq)
    if n_refseqs != 1:
        print(f'Reference file must contain ONE sequence. File {ref_seq} contains {n_refseqs}')
        return None
    
    # build the blast database
    os.makedirs(db_dir, exist_ok=True)
    blast.makeblastdb(ref_seq, db_out)
    logger.info(f'Generated blast databse {ref_name} using {ref_seq} at directory {db_dir}')
    return db_out
    
#%% classes
class Director:
    def __init__(self, out_dir='', warn_dir=''):
        # directories
        self.out_dir = out_dir
        self.warn_dir = warn_dir
        
        # attributes
        self.db_dir = None
        self.mat_file = None
        self.acc_file = None
        self.dims = None
        
        # workers
        self.blaster = blast.Blaster(out_dir)
        self.mapper = matrix.MatBuilder(out_dir)
    
    @property
    def accs(self):
        return self.mapper.acclist
    @property
    def blast_report(self):
        return self.blaster.report
    
    def get_files(self, seq_file, seq_name=None):
        # use this to check if a map file already exists
        self.mapper.generate_outnames(seq_file, seq_name=None)
        return self.mapper.mat_file, self.mapper.acc_file
        
    def direct(self, fasta_file, db_dir, out_name=None, evalue=0.005, threads=1, keep=False):
        # fasta file is the file to be mapped
        # out_name, optional file name for the generated matrix, otherwise generated automatically
        # evalue is the max evalue threshold for the blast report
        # db_dir points to the blast database: should be <path to db files>/<db prefix>
        # keep signals the director to conserve the generated map
        
        # check blast database
        check, db_files = blast.check_db_dir(db_dir)
        if not check:
            logger.error(f'Found {len(db_files)} files in {db_dir}. Must contain 6')
            return
        self.db_dir = db_dir
        
        print(f'Performing blast alignment of {fasta_file}...')
        # perform BLAST
        self.blaster.blast(fasta_file, db_dir, out_name, threads)
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
        logger.info(f'Generated alignment map files: {self.mat_file} (alignment matrix) and {self.acc_file} (accession index)')
        if keep:
            self.matrix = map_data[0]
            self.bounds = map_data[1]
            self.acclist = map_data[2]
        return

#%% main body
def main(fasta_file, out_name=None, evalue=0.005, threads=1, keep=False, out_dir='', warn_dir='', ref_seq=None, ref_name=None, db_dir=None, ref_dir='.'):
    # procure blast database
    if not ref_seq is None:
        db_dir = build_blastdb(ref_seq, ref_dir, ref_name)
    elif db_dir is None:
        print('Can\'t perform BLAST. Either provide a reference sequence file as --base_seq or a BLAST database as --db_dir')
        return
    
    # locate fasta file
    try:
        nseqs = check_fasta(fasta_file)
    except FileNotFoundError:
        print(f'Fasta file {fasta_file} not found')
        return
    if nseqs == 0:
        print(f'Fasta file {fasta_file} is empty')
        return
    
    # align and build matrix
    map_director = Director(out_dir, warn_dir)
    map_director.direct(fasta_file=fasta_file,
                        db_dir=db_dir,
                        out_name=out_name,
                        evalue=evalue,
                        threads=threads,
                        keep=keep)
    return map_director

if __name__ == '__main__':
    args = parser.parse_args()
    # build needed directories (if needed)
    if not os.path.isdir(args.gb_dir):
        print(f'Specified graboid directory {args.gb_dir} does not exist. A new one will be created')
    blastdb_dir = args.gb_dir + '/blast_dbs'
    map_dir = args.gb_dir + '/alignments/queries'
    warn_dir = args.gb_dir + '/warnings'
    os.makedirs(blastdb_dir, exist_ok=True)
    os.makedirs(map_dir, exist_ok=True)
    os.makedirs(warn_dir, exist_ok=True)
    
    # locate or create the blast database
    db_dir = build_blastdb(args.ref_seq, blastdb_dir, args.ref_name, args.overwrite)
    if db_dir is None:
        print('Failed to locate or create blast database. Aborting.')
    else:
        # execute main
        for fasta in args.fasta_file:
            main(fasta_file = fasta,
                 evalue = args.evalue,
                 threads = args.threads,
                 out_dir = map_dir,
                 warn_dir = warn_dir,
                 db_dir = db_dir)
