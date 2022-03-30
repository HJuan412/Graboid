#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 10:28:14 2021

@author: hernan
Blast downloaded sequences against references
"""

#%% libraries
from Bio.Blast.Applications import NcbiblastnCommandline as blast_cline
from glob import glob
import os

#%% functions
def blast(query, ref, out_file, threads = 1):
    # perform ungapped blast
    cline = blast_cline(cmd = 'blastn', task = 'blastn', db = ref, query = query, out = out_file, outfmt = "6 qseqid pident length qstart qend sstart send evalue", ungapped = True, num_threads = threads)
    cline()
#%% classes
class Blaster():
    def __init__(self, taxon, marker, in_file, ref_dir, out_dir, warn_dir):
        # in_file: fasta sequences
        # ref_dir: directory contianing the blast database
        # out_dir: directory containing the blast report
        # warn_dir: directory containing the generated warnings
        self.taxon = taxon
        self.marker = marker
        self.in_file = in_file
        self.ref_dir = ref_dir
        self.out_dir = out_dir
        self.warn_dir = warn_dir
        self.warnings = []
        self.out_file = f'{self.out_dir}/{self.taxon}_{self.marker}.blast'
        self.__check_ref_files()

    def __check_ref_files(self):
        self.db_name = None
        if not os.path.isdir(self.ref_dir):
            self.warnings.append(f'WARNING: specified BLAST database directory {self.ref_dir} not found')
            return

        db_files = glob(f'{self.ref_dir}/*.n*')
        if len(db_files) < 6:
            self.warnings.append(f'WARNING: Missing BLAST database files in directory {self.in_dir}')
            return
        
        self.db_name = db_files[0].split('.')[0]
    
    def __check_warnings(self):
        if len(self.warnings) > 0:
            with open(f'{self.warn_dir}/warning.blast', 'w') as warn_handle:
                warn_handle.write('\n'.join(self.warnings))

    def blast(self, threads = 1):
        if self.db_name is None:
            return
        
        print(f'Blasting {self.taxon} {self.marker}')
        try:
            blast(self.in_file, self.db_name, self.out_file, threads)
        except:
            # TODO develop warning
            self.warnings.append(f'WARNING: Unable to blast {self.taxon} {self.marker}')
        
        self.__check_warnings()