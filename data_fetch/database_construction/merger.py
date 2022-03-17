#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 13:30:40 2021

@author: hernan

Compare and merge temporal sequence files
"""

#%% libraries
from Bio import SeqIO
from glob import glob
import os
import shutil

#%% classes
class Merger():
    def __init__(self, taxon, marker, databases, in_dir, out_dir, warn_dir, old_file = None):
        self.taxon = taxon
        self.marker = marker
        self.prefix = f'{taxon}_{marker}'
        self.databases = databases
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.warn_dir = warn_dir
        self.old_file = old_file
        self.warnings = []
        self.not_found = set()
        self.__get_files()
        self.__rm_old_seqs()
    
    def __rm_old_seqs(self):
        # read the old sequence file (if present) and remove sequences to be replaced
        new_accs = self.acc_list.index
        if self.old_file is None:
            return
        records = []
        with open(self.old_file, 'r') as old_handle:
            for record in SeqIO.parse(old_handle, 'fasta'):
                acc = record.id.spit('.')[0].split('-')[0]
                if not acc in new_accs:
                    records.append(record)
        with open(f'{self.in_dir}/{self.prefix}_old.tmp', 'w') as kept_handle:
            # store the cropped old file as a temporal file along with the new ones
            SeqIO.write(records, kept_handle, 'fasta')
        
    def __get_files(self):
        # get the taxonomy and sequence files in dicts of the form {database: file}
        seq_files = {}
        tax_files = {}
        for db in self.databases:
            seq_file = f'{self.in_dir}/{self.prefix}_{db}.tmp'
            tax_file = f'{self.in_dir}/{self.prefix}_{db}.tax'
            check_seq = os.path.isfile(seq_file)
            check_tax = os.path.isfile(tax_file)
            missing = []
            if not check_seq:
                missing.append(seq_file)
            if not check_tax:
                missing.append(tax_file)
            
            # if either file is missing, list a warning and skip this database
            if len(missing) > 0:
                for m in missing:
                    self.warnings.append(f'WARNING: file {m} not found')
                continue
            seq_files[db] = seq_file
            tax_files[db] = tax_file
        
        self.seq_files = seq_files
        self.tax_files = tax_files
    
    def __merge(self):
        # merge all temporal sequence files into the output path
        # TODO: merge old files as well
        files = glob(f'{self.in_dir}/{self.prefix}_*.tmp')
        with open(f'{self.out_dir}/{self.prefix}.fasta', 'w') as out_handle:
            for file in files:
                with open(file, 'r') as in_handle:
                    shutil.copyfileobj(in_handle, out_handle)
    
    # TODO: merge taxonomies

    def __check_warnings(self):
        if len(self.warnings) > 0:
            with open(f'{self.warn_dir}/warnings.merge', 'w') as handle:
                handle.write('\n'.join(self.warnings))
    
    def merge(self):
        self.__merge()
        self.__check_warnings()