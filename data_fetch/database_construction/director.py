#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:53:46 2021

@author: hernan
Director for database creation and updating
"""

#%% libraries
from datetime import datetime
import logging
import os
import shutil

import surveyor as surv
import lister as lstr
import fetcher as ftch
import taxonomist as txnm
import merger as mrgr

#%% set logger
logger = logging.getLogger('database_logger')
logger.setLevel(logging.DEBUG)
# set formatter
fmtr = logging.Formatter('%(asctime) - %(levelname)s: %(message)s')
#%% Manage directories
def make_dirs(base_dir):
    os.makedirs(f'{base_dir}/data', exist_ok=bool)
    os.makedirs(f'{base_dir}/tmp', exist_ok=bool)
    os.makedirs(f'{base_dir}/warnings', exist_ok=bool)
        
# handle fasta
def fasta_name(fasta):
    return fasta.split('/')[-1].split('.')[0]

def move_file(file, dest, mv=False):
    if mv:
        shutil.move(file, dest)
    else:
        shutil.copy(file, dest)

#%% Main
class Director:
    def __init__(self, out_dir, tmp_dir, warn_dir):
        self.out_dir = out_dir
        self.tmp_dir = tmp_dir
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
        
        # set workers
        self.surveyor = surv.Surveyor(tmp_dir)
        self.lister = lstr.Lister(tmp_dir)
        self.fetcher = ftch.Fetcher(tmp_dir)
        self.taxonmoist = txnm.Taxonomist(tmp_dir)
        self.merger = mrgr.Merger(out_dir)
        
        # get outfiles
        self.get_out_files()
    
    def clear_tmp(self):
        for tmp_file in os.listdir(self.tmp_dir):
            os.remove(tmp_file)
    
    def set_ranks(self, ranks):
        fmt_ranks = [rk.lower() for rk in ranks]
        self.taxonomist.set_ranks(fmt_ranks)

    def direct_fasta(self, fasta_file, chunksize=500, max_attempts=3, mv = False):
        seq_path = f'{self.out_dir}/{fasta_name(fasta_file)}.fasta'
        if mv:
            shutil.move(fasta_file, seq_path)
        else:
            shutil.copy(fasta_file, seq_path)
        
        # generate taxtmp file
        print(f'Retrieving TaxIDs for {fasta_file}...')
        self.fetcher.fetch_tax_from_fasta(fasta_file)
        
        print('Reconstructing taxonomies...')
        self.taxonomist.out_dir = self.out_dir # dump tax table to out_dir
        self.taxonomist.taxing(self.fetcher.tax_files, chunksize, max_attempts)
        
        print('Building output files...')
        self.merger.merge_from_fasta(seq_path, self.taxonomist.out_files)
        self.get_out_files()
    
    def direct(self, taxon, marker, databases, chunksize=500, max_attempts=3):
        print('Surveying databases...')
        for db in databases:
            self.surveyor.survey(taxon, marker, db, max_attempts)
        print('Building accession lists...')
        self.lister.build_list(self.surveyor.out_files)
        print('Fetching sequences...')
        # TODO: create method to fetch failed sequences (if any)
        self.fetcher.fetch(self.lister.out_file, chunksize, max_attempts)
        print('Reconstructing taxonomies...')
        self.taxonomist.taxing(self.fetcher.tax_files, chunksize, max_attempts)
        print('Merging sequences...')
        self.merger.merge(self.fetcher.seq_files, self.taxonomist.out_files)
        self.get_out_files()
    
    def get_out_files(self):
        self.seq_file = self.merger.seq_out
        self.acc_file = self.merger.acc_out
        self.tax_file = self.merger.tax_out
        self.taxguide_file = self.merger.taxguide_out
