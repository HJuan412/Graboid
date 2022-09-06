#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:53:46 2021

@author: hernan
Director for database creation and updating
"""

#%% libraries
import logging
import os
import shutil

from database import surveyor as surv
from database import lister as lstr
from database import fetcher as ftch
from database import taxonomist as txnm
from database import merger as mrgr

#%% set logger
logger = logging.getLogger('Graboid.database')
logger.setLevel(logging.DEBUG)

#%% functions
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
        
        # set workers
        self.surveyor = surv.Surveyor(tmp_dir)
        self.lister = lstr.Lister(tmp_dir)
        self.fetcher = ftch.Fetcher(tmp_dir)
        self.taxonomist = txnm.Taxonomist(tmp_dir)
        self.merger = mrgr.Merger(out_dir)
        
        # get outfiles
        self.get_out_files()
    
    def clear_tmp(self):
        tmp_files = self.get_tmp_files()
        for file in tmp_files:
            os.remove(file)
    
    def set_ranks(self, ranks=['phylum', 'class', 'order', 'family', 'genus', 'species']):
        fmt_ranks = [rk.lower() for rk in ranks]
        logger.INFO(f'Taxonomic ranks set as {" ".join(fmt_ranks)}')
        self.taxonomist.set_ranks(fmt_ranks)
        self.merger.set_ranks(fmt_ranks)

    def direct_fasta(self, fasta_file, chunksize=500, max_attempts=3, mv = False):
        seq_path = f'{self.out_dir}/{fasta_name(fasta_file)}.fasta'
        if mv:
            shutil.move(fasta_file, seq_path)
        else:
            shutil.copy(fasta_file, seq_path)
        logger.info(f'Moved fasta file {fasta_file} to location {self.out_dir}')
        # generate taxtmp file
        print(f'Retrieving TaxIDs for {fasta_file}...')
        self.fetcher.fetch_tax_from_fasta(fasta_file)
        
        print('Reconstructing taxonomies...')
        # taxonomy needs no merging so it is saved directly to out_dir
        self.taxonomist.out_dir = self.out_dir # dump tax table to out_dir
        self.taxonomist.taxing(self.fetcher.tax_files, chunksize, max_attempts)
        self.taxonomist.out_files = {} # clear out_files container so the generated file is not found by get_tmp_files
        
        print('Building output files...')
        self.merger.merge_from_fasta(seq_path, self.taxonomist.out_files['NCBI'])
        self.get_out_files()
        print('Done!')
    
    def direct(self, taxon, marker, databases, chunksize=500, max_attempts=3):
        print('Surveying databases...')
        for db in databases:
            self.surveyor.survey(taxon, marker, db, max_attempts)
        print('Building accession lists...')
        self.lister.build_list(self.surveyor.out_files)
        print('Fetching sequences...')
        self.fetcher.set_bold_file(self.surveyor.out_files['BOLD'])
        self.fetcher.fetch(self.lister.out_file, chunksize, max_attempts)
        print('Reconstructing taxonomies...')
        self.taxonomist.taxing(self.fetcher.tax_files, chunksize, max_attempts)
        print('Merging sequences...')
        self.merger.merge(self.fetcher.seq_files, self.taxonomist.out_files)
        self.get_out_files()
        print('Done!')
    
    def get_tmp_files(self):
        tmp_files = []
        for file in self.surveyor.out_files.values():
            tmp_files.append(file)
        tmp_files.append(self.lister.out_file)
        for file in self.fetcher.seq_files.values():
            tmp_files.append(file)
        for file in self.fetcher.tax_files.values():
            tmp_files.append(file)
        for file in self.taxonomist.out_files.values():
            tmp_files.append(file)
        return tmp_files
    
    def get_out_files(self):
        self.seq_file = self.merger.seq_out
        self.acc_file = self.merger.acc_out
        self.tax_file = self.merger.tax_out
        self.guide_file = self.merger.taxguide_out
