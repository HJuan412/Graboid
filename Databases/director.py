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
import db_survey
import acc_lister
import seq_fetcher
import BOLD_post_processing as bold_pp
import db_merge
import os
#%% Manage directories
def generate_dirname():
    t = datetime.now()
    dirname = f'{t.day}_{t.month}_{t.year}-{t.hour}_{t.minute}_{t.second}'
    return dirname

def generate_subdirs(dirname):
    return f'{dirname}/Summaries', f'{dirname}/Sequence_files', f'{dirname}/Taxonomy_files', f'{dirname}/Acc_lists'

#%% Entrez
def set_entrez(email = "hernan.juan@gmail.com", apikey = "7c100b6ab050a287af30e37e893dc3d09008"):
    Entrez.email = email
    Entrez.api_key = apikey

#%% Main
# generate directories
dirname = generate_dirname()
summ_dir, seq_dir, tax_dir, acc_dir = generate_subdirs(dirname)

os.mkdir(dirname)
for sdir in [summ_dir, seq_dir, tax_dir, acc_dir]:
    os.mkdir(sdir)

taxons = ['Nematoda', 'Platyhelminthes']
markers = ['18S', '28S', 'COI']
bold = True
ena = False
ncbi = True
# Survey databases
db_survey.survey(summ_dir, taxons, markers, bold, ena, ncbi)    
# Build accession lists
acc_lister.acc_list(summ_dir, acc_dir)
# Fetch sequences
seq_fetcher.fetch(acc_dir, seq_dir)
# Postprocess BOLD data
bold_files = bold_pp.locate_BOLD_files(seq_dir)
for bold_file in bold_files:
    bold_pp.process_file(bold_file)
# Compare and merge
db_merge.select_entries(seq_dir)