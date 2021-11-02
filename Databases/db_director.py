#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:53:46 2021

@author: hernan
Director for database creation and updating
"""

#%% libraries
from Bio import Entrez
from datetime import datetime
import os

import db_surveyor as surv
import db_lister as lstr
import db_fetcher as ftch
import db_BOLD_postproc as boldpp
import db_merger as mrgr

import tax_director as taxdir
#%% Manage directories
def generate_dirname():
    t = datetime.now()
    dirname = f'{t.day}_{t.month}_{t.year}-{t.hour}_{t.minute}_{t.second}'
    return dirname

def generate_subdirs(dirname):
    return f'{dirname}/Summaries', f'{dirname}/Sequence_files', f'{dirname}/Taxonomy_files', f'{dirname}/Acc_lists', f'{dirname}/Warnings'

def new_directories():
    dirname = generate_dirname()
    summ_dir, seq_dir, tax_dir, acc_dir, warn_dir = generate_subdirs(dirname)
    
    os.mkdir(dirname)
    for sdir in [summ_dir, seq_dir, tax_dir, acc_dir, warn_dir]:
        os.mkdir(sdir)
    return summ_dir, seq_dir, tax_dir, acc_dir, warn_dir

#%% Entrez
def set_entrez(email = "hernan.juan@gmail.com", apikey = "7c100b6ab050a287af30e37e893dc3d09008"):
    Entrez.email = email
    Entrez.api_key = apikey

#%% Test parameters
# taxons = ['Nematoda', 'Platyhelminthes']
# markers = ['18S', '28S', 'COI']
# bold = True
# ena = False
# ncbi = True
taxons = ['Nematoda', 'Platyhelminthes']
markers = ['18S', '28S', 'COI']
bold = True
ena = False
ncbi = False
#%% Main
def make_database(taxons, markers, bold, ena, ncbi, dirname = None):
    # generate directories
    if dirname is None:
        summ_dir, seq_dir, tax_dir, acc_dir, warn_dir = new_directories()
    else:
        # TODO: en este caso, el directorio ya existe, qué hacer con archivos preexistentes?
        summ_dir, seq_dir, tax_dir, acc_dir, warn_dir = generate_subdirs(dirname)
    
    # set email and api key
    set_entrez()
    
    # Survey databases
    print('Surveying databases...')
    surv.survey(summ_dir, taxons, markers, bold, ena, ncbi)
    # Build accession lists
    print('Building accession lists...')
    lstr.acc_list(summ_dir, acc_dir)
    # Reconstruct taxonomies
    taxdir.TaxDirector(tax_dir, summ_dir, acc_dir)
    taxdir.tax_reconstruct()
    # # Fetch sequences
    print('Fetching sequences...')
    ftch.fetch_sequences(acc_dir, seq_dir, warn_dir)
    # # Postprocess BOLD data
    print('Processing BOLD files...')
    boldpp.processor(seq_dir)
    # # Compare and merge
    print('Comparing and merging...')
    mrgr.merger(seq_dir)