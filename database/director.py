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
import re
import shutil

from Bio import Entrez
from database import surveyor as surv
from database import lister as lstr
from database import fetcher as ftch
from database import taxonomist as txnm
from database import merger as mrgr
from mapping import director as mp
from preprocess import feature_selection as fsele

#%% set logger
logger = logging.getLogger('Graboid.database')
logger.setLevel(logging.DEBUG)

#%% functions
# Entrez
def set_entrez(email, apikey):
    Entrez.email = email
    Entrez.api_key = apikey

def move_file(file, dest, mv=False):
    if mv:
        shutil.move(file, dest)
    else:
        shutil.copy(file, dest)

#%%
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
    
    def clear_tmp(self):
        tmp_files = self.get_tmp_files()
        for file in tmp_files:
            os.remove(file)
    
    def set_ranks(self, ranks=['phylum', 'class', 'order', 'family', 'genus', 'species']):
        # set taxonomic ranks to retrieve for the training data.
        # propagate to taxonomist and merger
        fmt_ranks = [rk.lower() for rk in ranks]
        logger.INFO(f'Taxonomic ranks set as {" ".join(fmt_ranks)}')
        self.taxonomist.set_ranks(fmt_ranks)
        self.merger.set_ranks(fmt_ranks)

    def direct_fasta(self, fasta_file, chunksize=500, max_attempts=3, mv=False):
        # direct database construction from a prebuilt fasta file
        # sequences should have a valid genbank accession
        fasta_name = re.sub('.*/', '', re.sub('\..*', '', fasta_file))
        seq_path = f'{self.out_dir}/{fasta_name}.fasta'
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
        
        print('Building output files...')
        self.merger.merge_from_fasta(seq_path, self.taxonomist.out_files['NCBI'])
        self.get_out_files()
        self.taxonomist.out_files = {} # clear out_files container so the generated file is not found by get_tmp_files
        print('Done!')
    
    def direct(self, taxon, marker, databases, chunksize=500, max_attempts=3):
        # build database from zero, needs a valid taxon (ideally at a high level such as phylum or class) and marker gene
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
    
    @property
    def seq_file(self):
        return self.merger.seq_out
    @property
    def acc_file(self):
        return self.merger.acc_out
    @property
    def tax_file(self):
        return self.merger.tax_out
    @property
    def guide_file(self):
        return self.merger.taxguide_out
    @property
    def rank_file(self):
        return self.merger.rank_dict_out
    @property
    def valid_file(self):
        return self.merger.valid_rows_out

#%% main body
def main(out_dir, tmp_dir, warn_dir, email, api_key, ranks=None, bold=False, taxon=None, marker=None, fasta=None, chunksize=500, max_attempts=3, mv=False, keep_tmp=False, evalue=0.005, threads=1, keep=False):
    # fetch fasta files, align to reference, build map, quantify information
    # store everything to an EMPTY directory
    if len(os.listdir(out_dir)) > 0:
        print(f'Error: Designated output directory {out_dir} is not empty')
        return
    os.makedirs(out_dir + '/' + tmp_dir)
    os.makedirs(out_dir + '/' + warn_dir)
    
    # fetch sequences
    db_director = Director(out_dir, tmp_dir, warn_dir)
    try:
        set_entrez(email, api_key)
    except AttributeError:
        print('Missing email adress and/or API key')
    # user specified ranks to use
    if not ranks is None:
        db_director.set_ranks(ranks)
    # set databases
    databases = ['NCBI']
    if bold:
        databases.append('BOLD')
    
    if not fasta is None:
        # build db using fasta file (overrides taxon, mark)
        db_director.direct_fasta(fasta,
                                 chunksize,
                                 max_attempts,
                                 mv)
    elif not (taxon is None or marker is None):
        #build db using tax & mark
        db_director.direct(taxon,
                           marker,
                           databases,
                           chunksize,
                           max_attempts)
    else:
        print('No search parameters provided. Either set a path to a fasta file in the --fasta argument or a taxon and a marker in the --taxon and --marker arguments')
        return
    
    # clear temporal files
    if not keep_tmp:
        db_director.clear_tmp()
    
    # build map
    map_director = mp.Director(out_dir, warn_dir)
    map_director.direct(fasta_file=db_director.seq_file,
                        db_dir=out_dir,
                        evalue=evalue,
                        threads=threads,
                        keep=keep)
    
    # quantify information
    selector = fsele.Selector(out_dir)
    selector.set_matrix(map_director.matrix, map_director.bounds, db_director.tax_file)
    selector.build_tabs()
    selector.save_order_mat()

if __name__ == '__main__':
    pass