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
import re

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
                            parse_seqids=True,
                            input_type='fasta')
    cline()

def check_db_dir(db_dir):
    # counts the database files present at the given location, check returns True if six .n* files are found
    db_files = glob(f'{db_dir}/*.n*')
    check = False
    if len(db_files) == 6:
        check = True
    return check, db_files

#%% classes
class Blaster:
    def __init__(self, out_dir):
        self.report = None
        self.out_dir = out_dir
    
    def blast(self, fasta_file, db_dir, out_name=None, threads=1):
        self.report = None
        check, db_files = check_db_dir(db_dir)
        if not check:
            logger.error(f'Incomplete BLAST database ({len(db_files)} files found)')
            return
        
        # set output name
        if out_name is None:
            out_name = re.sub('.*/', '', re.sub('\..*', '', fasta_file))
        blast_out = f'{self.out_dir}/{out_name}.BLAST'
        
        # perform BLAST
        print(f'Blasting {fasta_file}')
        try:
            blast(fasta_file, db_dir, blast_out, threads)
            self.report = blast_out
            logger.info(f'Generated blast report at {blast_out}')
        except:
            # TODO develop warning
            logger.error(f'Unable to blast {fasta_file}')
