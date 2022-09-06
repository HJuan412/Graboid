#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 10:28:14 2021

@author: hernan
Blast downloaded sequences against references
"""

#%% libraries
from Bio.Blast.Applications import NcbiblastnCommandline as blast_cline
from Bio.Blast.Applications import NcbimakeblastdbCommandline as makeblast_cline
from glob import glob
import logging
import os

#%% set logger
logger = logging.getLogger('Graboid.mapper.blast')

#%% functions
def blast(query, ref, out_file, threads=1):
    # perform ungapped blast
    cline = blast_cline(cmd='blastn',
                        task='blastn',
                        db=ref,
                        query=query,
                        out=out_file,
                        outfmt="6 qseqid pident length qstart qend sstart send evalue",
                        ungapped=True,
                        num_threads=threads)
    cline()

def makeblastdb(ref_file, db_prefix):
    # build the reference BLAST database
    cline = makeblast_cline(input_file=ref_file,
                            out=db_prefix,
                            dbtype='nucl',
                            input_type='fasta')
    cline()

def check_db_dir(db_dir):
    db_files = glob(f'{db_dir}/*.n*')
    check = False
    if len(db_files) == 6:
        check = True
    return check, db_files

#%% classes
class Blaster:
    def __init__(self):
        self.report = None

    def make_ref_db(self, ref_file, db_dir, clear=False):
        check, db_files = check_db_dir(db_dir)
        if clear or not check:
            # found db files are incomplete or option 'clear' is enabled
            # delete previous files and create new databases
            for file in db_files:
                os.remove(file)
            makeblastdb(ref_file, f'{db_dir}/ref')
            logger.info(f'Generated blast reference databse at directory {db_dir}')
            return
        logger.warning(f'Directory {db_dir} already contains a database. Set \'clear\' to False if you wish to replace it.')
    
    def blast(self, fasta_file, db_dir, out_name=None, threads=1):
        check, db_files = check_db_dir(db_dir)
        if not check:
            logger.error(f'Incomplete BLAST database ({len(db_files)} files found)')
            return
        db_prefix = db_files[0].split('.n')[0]
        
        # set output name
        if not out_name is None:
            blast_out = f'{self.out_dir}/{out_name}.BLAST'
        else:
            blast_out = fasta_file.split('/')[-1].split('.')[0]
            blast_out = f'{self.out_dir}/{blast_out}.BLAST'
        # perform BLAST
        print(f'Blasting {fasta_file}')
        try:
            blast(fasta_file, db_prefix, blast_out, threads)
            self.report = blast_out
            logger.info(f'Generated blast report at {blast_out}')
        except:
            # TODO develop warning
            logger.error(f'Unable to blast {fasta_file}')
