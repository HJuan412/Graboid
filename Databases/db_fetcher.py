#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 14:21:22 2021

@author: hernan
"""

#%% libraries
from Bio import Entrez
from datetime import datetime
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

def generate_filename(tax, mark, db, timereg = True):
    t = datetime.now()
    filename = f'{tax}_{mark}_{db}_{t.day}-{t.month}-{t.year}_{t.hour}-{t.minute}-{t.second}.fasta'
    return filename

# fetch functions
def fetch_ncbi(acc_list, out_handle):
    seq_handle = Entrez.efetch(db = 'nucleotide', id = acc_list, rettype = 'fasta', retmode = 'text')
    out_handle.write(bytearray(seq_handle.read(), 'utf-8'))
    return

def fetch_api(apiurl, out_handle):
    http = urllib3.PoolManager()
    r = http.request('GET', apiurl, preload_content = False)
    for chunk in r.stream():
        out_handle.write(chunk)
    r.release_conn()
    return

def fetch_ena(acc_list, out_handle):
    accstring = '%2C'.join(acc_list)
    apiurl = f'https://www.ebi.ac.uk/ena/browser/api/fasta/{accstring}?lineLimit=0'
    fetch_api(apiurl, out_handle)
    return apiurl

def fetch_bold(acc_list, out_handle):
    accstring = '|'.join(acc_list)
    apiurl = f'http://www.boldsystems.org/index.php/API_Public/sequence?ids={accstring}'
    fetch_api(apiurl, out_handle)
    return

#%% classes
class DbFetcher():
    def __init__(self, in_file, taxon, marker, database, out_dir, warn_dir):
        self.acc_tab = pd.read_csv(in_file, index_col = 0)
        self.marked_accs = self.acc_tab.loc[self.acc_tab['Entry'] >= 1, 'Accession'].tolist() # get entries with Entry code 1 or 2
        self.taxon = taxon
        self.marker = marker
        self.database = database
        self.set_fetchfunc(database)
        self.generate_filenames()
        self.out_dir = out_dir
        self.warn_dir = warn_dir
    
    def set_fetchfunc(self, database):
        if database == 'BOLD':
            self.fetch_func = fetch_bold
        elif database == 'ENA':
            self.fetch_func = fetch_ena
        elif database == 'NCBI':
            self.fetch_func = fetch_ncbi
    
    def generate_filenames(self):
        self.out_file = f'{self.out_dir}/{self.taxon}_{self.marker}_{self.database}.tmp'
        self.warn_file = f'{self.warn_dir}/{self.taxon}_{self.marker}_{self.database}.lis'
    
    def add_to_warn_file(self, acc_list):
        with open(self.warn_file, 'a') as warn_handle:
            warn_handle.write('\n'.join(acc_list))
        return

    def fetch(self, chunk_size):
        chunks = acc_slicer(self.marked_accs, chunk_size)
        nchunks = ceil(len(self.marked_accs)/chunk_size)
        
        with open(self.out_file, 'wb') as out_handle: # change mode to 'ab' so an interrupted download can be completed using a Fetcher instance with (warn_dir passed as out_dir)
            for idx, chunk in enumerate(chunks):
                print(f'{self.database}. Chunk {idx + 1} of {nchunks}')
                try:
                    self.fetch_func(chunk, out_handle)
                except:
                    print(f'Error downloading chunk {idx + 1}. Accessions stored in {self.warn_dir}')
                    self.add_to_warn_file( chunk)

class Fetcher():
    # TODO: fetch sequences stored in warn_dir
    def __init__(self, in_dir, out_dir, warn_dir):
        self.acc_tab = build_acc_tab(in_dir)
        self.out_dir = out_dir
        self.warn_dir = warn_dir
    
    def check_acclists(self):
        # make sure acc_tab is not empty
        if len(self.acc_tab) == 0:
            with open(f'{self.warn_dir}/fetcher.warn', 'w') as warn_handle:
                warn_handle.write('No accession list files found in directory {self.in_dir}')
            return False
        return True
    
    def fetch(self, chunk_size):
        if self.check_acclists():
            for idx, row in self.acc_tab.iterrows():
                taxon = row['Taxon']
                marker = row['Marker']
                dbase = row['Database']
                file = row['File']
                dbfetch = DbFetcher(file, taxon, marker, dbase, self.out_dir, self.warn_dir)
                dbfetch.fetch(chunk_size)
