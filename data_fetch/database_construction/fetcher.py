#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 14:21:22 2021

@author: hernan
"""

#%% libraries
from Bio import Entrez
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from glob import glob
from math import ceil

import numpy as np
import pandas as pd
import urllib3

#%% functions
# Entrez
# TODO: remove mail and apikey
def set_entrez(email = "hernan.juan@gmail.com", apikey = "7c100b6ab050a287af30e37e893dc3d09008"):
    Entrez.email = email
    Entrez.api_key = apikey

# acc data handling
def get_file_data(filename):
    split_file = filename.split('/')[-1].split('.acc')[0].split('_')

    return split_file + [filename]

def build_acc_tab(acc_dir):
    acc_files = glob(f'{acc_dir}/*.acc')
    n_files = len(acc_files)
    acc_tab = pd.DataFrame(index = range(n_files), columns = ['Taxon', 'Marker', 'Database', 'File'])
    
    for idx, file in enumerate(acc_files):
        
        acc_tab.at[idx] = get_file_data(file)
    return acc_tab

# accession list handling
def acc_slicer(acc_list, chunksize):
    # slice the list of accessions into chunks
    n_seqs = len(acc_list)
    for i in np.arange(0, n_seqs, chunksize):
        yield acc_list[i:i+chunksize]

# fetch functions
def fetch_ncbi(acc_list, out_seqs, out_taxs):
    # download sequences from NCBI, generates a temporal fasta and am acc:taxID list
    seq_handle = Entrez.efetch(db = 'nucleotide', id = acc_list, rettype = 'fasta', retmode = 'xml')
    seqs_recs = Entrez.read(seq_handle)
    
    seqs = []
    taxs = []

    for acc, seq in zip(acc_list, seqs_recs):
        seqs.append(SeqRecord(id=acc, seq = Seq(seq['TSeq_sequence'])))
        taxs.append(','.join(acc, seq['TSeq_taxid']))
    
    with open(out_seqs, 'a') as out_handle0:
        SeqIO.write(seqs, out_handle0, 'fasta')

    with open(out_taxs, 'a') as out_handle1:
        out_handle1.write('\n'.join(taxs + ['']))
    
def fetch_api(apiurl, out_handle, tax_handle = None):
    http = urllib3.PoolManager()
    r = http.request('GET', apiurl, preload_content = False)
    for chunk in r.stream():
        out_handle.write(chunk)
    r.release_conn()
    return

def fetch_ena(acc_list, out_seqs, out_taxs = None): # out_taxs kept for compatibility
    accstring = '%2C'.join(acc_list)
    apiurl = f'https://www.ebi.ac.uk/ena/browser/api/fasta/{accstring}?lineLimit=0'
    with open(out_seqs, 'ab') as out_handle:
        fetch_api(apiurl, out_handle)
    return apiurl

def fetch_bold(acc_list, out_seqs, out_taxs = None):
    accstring = '|'.join(acc_list)
    apiurl = f'http://www.boldsystems.org/index.php/API_Public/sequence?ids={accstring}'
    with open(out_seqs, 'ab') as out_handle:
        fetch_api(apiurl, out_handle)
    return

#%% classes
class DbFetcher():
    def __init__(self, acc_tab, database, prefix):
        self.acc_tab = acc_tab
        self.database = database
        self.__set_fetchfunc(database)
        self.out_seqs = f'{prefix}_{database}.tmp'
        self.out_taxs = f'{prefix}_{database}.tax'
        self.warnings = []
    
    def __set_fetchfunc(self, database):
        if database == 'BOLD':
            self.fetch_func = fetch_bold
        elif database == 'ENA':
            self.fetch_func = fetch_ena
        elif database == 'NCBI':
            self.fetch_func = fetch_ncbi

    def fetch(self, chunk_size = 500):
        chunks = acc_slicer(self.acc_tab['Accession'], chunk_size)
        nchunks = ceil(len(self.marked_accs)/chunk_size)

        for idx, chunk in enumerate(chunks):
            print(f'{self.database}, {self.taxon} {self.marker}. Chunk {idx + 1} of {nchunks}')
            try:
                self.fetch_func(chunk, self.out_seqs, self.out_taxs)
            except:
                self.warnings += chunk

class Fetcher():
    # TODO: fetch sequences stored in warn_dir
    def __init__(self, taxon, marker, acc_file, out_dir, warn_dir):
        self.taxon = taxon
        self.marker = marker
        self.acc_tab = pd.read_csv(acc_file)
        self.out_dir = out_dir
        self.warn_dir = warn_dir
        self.prefix = f'{out_dir}/{taxon}_{marker}'
        self.out_tax = f'{out_dir}/{taxon}_{marker}_NCBI.taxsumm'
        self.warnings = []
        self.__check_acclists(acc_file)
        self.__set_fetchers()

    def __check_acclists(self, acc_file):
        # make sure acc_tab is not empty
        self.accs = True
        if len(self.acc_tab) == 0:
            self.warnings.append(f'WARNING: Accession table {acc_file} is empty') # TODO: get acc_file from Lister
            self.accs = False

    def __set_fetchers(self):
        # prepare the fetcher objects needed for each database
        fetchers = []
        if self.accs:
            for dbase, acc_subtab in self.acc_tab.groupby('Database'):
                fetchers.append(DbFetcher(acc_subtab, dbase, self.out_tax))
        
        self.fetchers = fetchers

    def fetch(self, chunk_size = 500):
        for ftch in self.fetchers:
            ftch.fetch(chunk_size)
        
        self.check_fetch_warnings()
        self.save_warnings()
    
    def check_fetch_warnings(self):
        for ftch in self.fetchers:
            n_warns = len(ftch.warnings)
            total_recs = len(ftch.acc_tab)
            if n_warns > 0:
                self.warnings.append(f'WARNING: Failed to download {n_warns} of {total_recs} sequences from {ftch.database}:')
                self.warnings += ftch.warnings
    
    def save_warnings(self):
        if len(self.warnings) > 0:
            with open(f'{self.warn_dir}/warnings.ftch', 'w') as handle:
                handle.write('\n'.join(self.warnings))