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

import db_surveyor as surv
import db_lister as lstr
import db_fetcher as ftch
import db_BOLD_postproc as bdpp
import db_merger as mrgr

import tax_director as txdr
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

def get_surv_tools(bold = True, ena = False, ncbi = True):
    t1 = []
    t2 = []
    if bold:
        t1.append(surv.SurveyBOLD)
    if ena:
        t2.append(surv.SurveyENA)
    if ncbi:
        t2.append(surv.SurveyNCBI)
    
    return t1, t2

#%% Test parameters
taxons = ['Nematoda', 'Platyhelminthes']
markers = ['18S', '28S', 'COI']
bold = True
ena = False
ncbi = False

old_accs = None # for lister
chunk_size = 500 # for fetcher
#%% Main
def make_database(taxons, markers, bold, ena, ncbi, dirname = None):
    # generate directories
    if dirname is None:
        summ_dir, seq_dir, tax_dir, acc_dir, warn_dir = new_directories()
    else:
        # TODO: en este caso, el directorio ya existe, qué hacer con archivos preexistentes?
        summ_dir, seq_dir, tax_dir, acc_dir, warn_dir = generate_subdirs(dirname)
    
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