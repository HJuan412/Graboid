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
def set_entrez(email = "hernan.juan@gmail.com", apikey = "7c100b6ab050a287af30e37e893dc3d09008"):
    Entrez.email = email
    Entrez.api_key = apikey
# acc data handling
def get_file_data(filename):
    split_file = filename.split('/')[-1].split('.tab')[0].split('_')

    file_tax = split_file[1]
    file_mark = split_file[2]
    return file_tax, file_mark, filename

def build_acc_tab(acc_dir):
    acc_files = glob(f'{acc_dir}/acc_*.tab')
    n_files = len(acc_files)
    acc_tab = pd.DataFrame(index = range(n_files), columns = ['Taxon', 'Marker', 'File'])
    
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

# TODO: delete (is a Fetcher method now)
# def fetch_sequences(acc_tab, dbase, tax, marker, chunksize, outdir):
#     acc_list = acc_tab['Accession'].tolist()
#     chunks = acc_slicer(acc_list, chunksize)
#     outfile = f'{outdir}/{tax}_{marker}_{dbase}.tmp'
#     with open(outfile, 'wb') as out_handle:
#         for idx, chunk in enumerate(chunks):
#             print(f'{dbase}. Chunk {idx + 1} of {len(acc_list) / chunksize}')
#             if dbase == 'NCBI':
#                 fetch_ncbi(chunk, out_handle)
#             elif dbase == 'ENA':
#                 fetch_ena(chunk, out_handle)
#             elif dbase == 'BOLD':
#                 fetch_bold(chunk, out_handle)
#     return

#%% classes
class Fetcher():
    def __init__(self, dbase, fetch_func, out_dir, chunksize = 500):
        self.dbase = dbase
        self.fetch = fetch_func
        self.out_dir = out_dir
        self.chunksize = chunksize
    
    def generate_filename(self, taxon, marker = ''):
        filename = f'{self.out_dir}/{taxon}_{marker}_{self.dbase}.tmp'
        return filename
    
    def get_nchunks(self, acc_list_len):
        nchunks = ceil(acc_list_len/self.chunksize)
        return nchunks

    def fetch(self, acc_list, taxon, marker = ''):
        chunks = acc_slicer(acc_list, self.chunksize)
        nchunks = self.get_nchunks(len(acc_list))
        out_file = self.generate_filename(taxon, marker)
        
        with open(out_file, 'wb') as out_handle:
            for idx, chunk in enumerate(chunks):
                print(f'{self.dbase}. Chunk {idx + 1} of {nchunks}')
                self.fetch(chunk, out_handle)
        return
#%% Main
def fetch(acc_dir, out_dir, chunksize = 500):
    # list accsession files
    acc_file_tab = build_acc_tab(acc_dir)
    
    for _, row in acc_file_tab.iterrows():
        taxon = row['Taxon']
        marker = row['Marker']
        file = row['File']
        acc_file = pd.read_csv(file, index_col = 0)
        
        fetchers = {'BOLD':Fetcher('BOLDr', fetch_bold, out_dir, chunksize), # BOLDr signifies there are the raw BOLD files and must still be processed
                    'ENA':Fetcher('ENA', fetch_ena, out_dir, chunksize),
                    'NCBI':Fetcher('NCBI', fetch_ncbi, out_dir, chunksize)}
        for dbase, sub_tab in acc_file.groupby('Database'):
            print(f'Fetching {taxon} {marker} sequences from {dbase} database')
            fetcher = fetchers[dbase]
            acc_list = sub_tab['Accession'].tolist()
            fetcher.fetch(acc_list, taxon, marker)
        return