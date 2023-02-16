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
import pandas as pd
import re
import subprocess

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
    # retrieve reference marker data
    bdbcmd_cline = f'blastdbcmd -db {ref} -dbtype nucl -entry all -outfmt %l'.split()
    ref_marker_len = int(re.sub('\\n', '', subprocess.run(bdbcmd_cline, capture_output=True).stdout.decode()))
    blast_tab = pd.read_csv(out_file, sep='\t', header=None, names='qseqid pident length qstart qend sstart send evalue'.split())
    if len(blast_tab) == 0:
        raise Exception(f'Blast search of file {query} on database {ref} yielded no results')
    ref_row = pd.Series(index=blast_tab.columns)
    # add the reference length as an extra row in the report
    ref_row.at['qseqid', 'length', 'evalue'] = ['Reference', ref_marker_len, 100] # evalue of 100 means this row is always filtered out
    # overwrite blast report with column names and reference row
    pd.concat([blast_tab, ref_row.to_frame().T]).to_csv(out_file, index=False)

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
    db_files = glob(db_dir + '/*.n*')
    if len(db_files) == 6:
        db_name = re.sub('\..*', '', db_files[0])
        return db_name
    raise Exception(f'Incomplete BLAST database ({len(db_files)} files found)')

#%% classes
class Blaster:
    def __init__(self, out_dir):
        self.report = None
        self.out_dir = out_dir
    
    def blast(self, fasta_file, db_dir, threads=1):
        self.report = None
        try:
            db_name = check_db_dir(db_dir)
        except Exception as excp:
            logger.error(excp)
            raise
        
        # set output name
        blast_out = re.sub('.*/', self.out_dir + '/', re.sub('\..*', '.BLAST', fasta_file))
        
        # perform BLAST
        print(f'Blasting {fasta_file}')
        try:
            blast(fasta_file, db_name, blast_out, threads)
            self.report = blast_out
            logger.info(f'Generated blast report at {blast_out}')
        except Exception as excp:
            logger.error(excp)
            raise
