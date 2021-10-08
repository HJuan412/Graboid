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

# TODO: Verbose
def acc_slicer(acc_list, chunksize):
    # slice the list of accessions into chunks
    n_seqs = len(acc_list)
    for i in np.arange(0, n_seqs, chunksize):
        yield acc_list[i:i+chunksize]

def generate_filename(tax, mark, db, timereg = True):
    t = datetime.now()
    filename = f'{tax}_{mark}_{db}_{t.day}-{t.month}-{t.year}_{t.hour}-{t.minute}-{t.second}.fasta'
    return filename

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

def fetch_sequences(acc_tab, dbase, tax, marker, chunksize, outdir):
    acc_list = acc_tab['Accession'].tolist()
    chunks = acc_slicer(acc_list, chunksize)
    outfile = f'{outdir}/{tax}_{marker}_{dbase}.tmp'
    with open(outfile, 'wb') as out_handle:
        for idx, chunk in enumerate(chunks):
            print(f'{dbase}. Chunk {idx + 1} of {len(acc_list) / chunksize}')
            if dbase == 'NCBI':
                fetch_ncbi(chunk, out_handle)
            elif dbase == 'ENA':
                fetch_ena(chunk, out_handle)
            elif dbase == 'BOLD':
                fetch_bold(chunk, out_handle)
    return
#%% Main
def fetch(acc_dir, outdir, chunksize = 500):
    # list accsession files
    acc_tab = build_acc_tab(acc_dir)
    
    for _, row in acc_tab.iterrows():
        taxon = row['Taxon']
        marker = row['Marker']
        file = row['File']
        acc_file = pd.read_csv(file, index_col = 0)
        
        for dbase, sub_tab in acc_file.groupby('Database'):
            fetch_sequences(sub_tab, dbase, taxon, marker, chunksize, outdir)
        return