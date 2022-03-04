#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:53:46 2021

@author: hernan
Director for database creation and updating
"""

#%% libraries
from datetime import datetime
import os

import surveyor as surv
import lister as lstr
import fetcher as ftch
import taxonomist as txnm

# TODO: these all go away
# import db_surveyor as surv
# import db_lister as lstr
# import db_fetcher as ftch
# import db_BOLD_postproc as bdpp
import db_merger as mrgr

import tax_director as txdr
#%% Manage directories
def generate_dirname():
    t = datetime.now()
    dirname = f'{t.day}_{t.month}_{t.year}-{t.hour}_{t.minute}_{t.second}'
    return dirname

def generate_subdirnames(dirname):
    return f'{dirname}/Summaries', f'{dirname}/Sequence_files', f'{dirname}/Taxonomy_files', f'{dirname}/Acc_lists', f'{dirname}/Warnings'

def new_directories():
    dirname = generate_dirname()
    summ_dir, seq_dir, tax_dir, acc_dir, warn_dir = generate_subdirnames(dirname)
    
    os.mkdir(dirname)
    for sdir in [summ_dir, seq_dir, tax_dir, acc_dir, warn_dir]:
        os.mkdir(sdir)
    return summ_dir, seq_dir, tax_dir, acc_dir, warn_dir

#%% Test parameters
taxons = ['Nematoda', 'Platyhelminthes']
markers = ['18S', '28S', 'COI']
bold = True
ena = False
ncbi = False

old_accs = None # for lister
chunk_size = 500 # for fetcher
#%% Main
class Director():
    # def __init__(self, taxons, markers, db_dir = None, bold = True, ena = False, ncbi = True, email = 'hernan.juan@gmail.com', apikey = '7c100b6ab050a287af30e37e893dc3d09008'):
    #     self.db_dir = db_dir
    #     self.set_directories()
    #     self.create_dirs()
    #     self.bold = bold
    #     self.ena = ena
    #     self.ncbi = ncbi
    #     self.t1, self.t2 = get_surv_tools(bold, ena, ncbi)
    #     self.taxons = taxons
    #     self.markers = markers
    #     self.set_workers(taxons, markers)
    #     ftch.set_entrez(email, apikey)
    
    def __init__(self, db_dir, tmp_dir, wrn_dir, taxon, marker, databases = ['NCBI', 'BOLD']):
        self.db_dir = db_dir
        self.tmp_dir = tmp_dir
        self.wrn_dir = wrn_dir
        self.taxon = taxon
        self.marker = marker
        self.databases = databases
        self.prefix = f'{taxon}_{marker}'
        self.updating = False # checks if database already exists

    def check_dir(self):
        if os.path.isdir(self.db_dir):
            if os.path.isfile(self.db_dir + f'/{self.prefix}.fasta') and os.path.isfile(self.db_dir + f'/{self.prefix}.tax'):
                self.updating = True
            # TODO: print warining if missing files
    
    def make_dirs(self):
        os.makedirs(self.db_dir, exist_ok=bool)
        os.makedirs(self.tmp_dir, exist_ok=bool)
        os.makedirs(self.warn_dir, exist_ok=bool)

    def set_workers(self):
        # TODO: update workers
        self.surveyor = surv.Surveyor(self.taxon, self.marker, self.databases, self.tmp_dir, self.warn_dir)
        self.lister = lstr.Lister(self.taxon, self.marker, self.tmp_dir, self.warn_dir) # TODO: incorporate database updating (already in lister, just need to add it here)
        self.fetcher = ftch.Fetcher(self.taxon, self.marker, self.lister.merged, self.tmp_dir, self.warn_dir)
        self.taxer = txnm.Taxonomist(self.taxon, self.marker, self.databases, self.tmp_dir, self.warn_dir)
        self.merger = mrgr.Merger(self.seq_dir, self.seq_dir, self.warn_dir, db_order = ['NCBI', 'BOLD', 'ENA'])
    
    def direct_survey(self, ntries = 3):
        self.surveyor.survey(ntries)
    
    def direct_listing(self):
        self.lister.make_list()
    
    #TODO add marker filter to the BOLD fetcher
    def direct_fetching(self, chunk_size):
        self.fetcher.fetch(chunk_size)
    
    #TODO add option to edit used ranks
    def direct_taxing(self, chunksize):
        self.taxer.taxing(chunksize)

    def direct_bold_pp(self):
        self.bold_postprocessor.set_bold_tab()
        self.bold_postprocessor.process(self.markers)
    
    def direct_merging(self):
        self.merger.merge()
    
    def direct(self, chunk_size, markers):
        print('Surveying databases...')
        self.direct_survey()
        print('Building accession lists...')
        self.direct_listing()
        print('Reconstructing taxonomies...')
        self.direct_tax_reconstruction()
        print('Fetching sequences...')
        self.direct_fetching(chunk_size)
        print('Processing BOLD files...')
        self.direct_bold_pp()
        print('Comparing and merging...')
        self.direct_merging()

#TODO: once Director class is tested, remove this shit funcition
def make_database(taxons, markers, bold, ena, ncbi, dirname = None):
    # generate directories
    if dirname is None:
        summ_dir, seq_dir, tax_dir, acc_dir, warn_dir = new_directories()
    else:
        # TODO: en este caso, el directorio ya existe, qué hacer con archivos preexistentes?
        summ_dir, seq_dir, tax_dir, acc_dir, warn_dir = generate_subdirnames(dirname)
    
    # TODO: handle old directory
    # set email and api key
    ftch.set_entrez()
    
    # select survey tools
    t1, t2 = get_surv_tools(bold, ena, ncbi)

    # instance classes
    surveyor = surv.Surveyor(taxons, markers, t1, t2, summ_dir, warn_dir)
    lister = lstr.Lister(summ_dir, acc_dir, warn_dir, old_accs)
    fetcher = ftch.Fetcher(acc_dir, seq_dir, warn_dir)
    bold_postprocessor = bdpp.BOLDPostProcessor(seq_dir, seq_dir, warn_dir)
    merger = mrgr.Merger(seq_dir, seq_dir, warn_dir, db_order = ['NCBI', 'BOLD', 'ENA'])
    
    
    # Survey databases
    print('Surveying databases...')
    surveyor.survey()
    # Build accession lists
    print('Building accession lists...')
    lister.build_lists()
    # Reconstruct taxonomies
    txdr.TaxDirector(tax_dir, summ_dir, acc_dir)
    txdr.tax_reconstruct()
    # # Fetch sequences
    print('Fetching sequences...')
    fetcher.fetch(chunk_size)
    # # Postprocess BOLD data
    print('Processing BOLD files...')
    bold_postprocessor.process(markers)
    # # Compare and merge
    print('Comparing and merging...')
    merger.merge()