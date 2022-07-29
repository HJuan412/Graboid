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
logger = logging.getLogger('mapping_logger.blast')

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
        self.db = None
        self.report = None
        
    def make_ref_db(self, ref_file, db_dir, clear=False):
        check, db_files = check_db_dir(db_dir)
        self.db = f'{db_dir}/ref'
        if clear or not check:
            # found db files are incomplete or option 'clear' is enabled
            # delete previous files and create new databases
            for file in db_files:
                os.remove(file)
            makeblastdb(ref_file, self.db)
            return
        logger.warning(f'Directory {db_dir} already contains a database. Ignoring.')
    
    def blast(self, fasta_file, out_file, threads=1):
        if self.db is None:
            logger.warning('No database detected. Aborting')
            return
        
        print(f'Blasting {fasta_file}')
        try:
            blast(fasta_file, self.db, out_file, threads)
            return out_file
        except:
            # TODO develop warning
            logger.warning(f'Unable to blast {fasta_file}')
        return None
