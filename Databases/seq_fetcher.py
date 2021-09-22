#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 14:21:22 2021

@author: hernan
"""

#%% libraries
from Bio import Entrez
from datetime import datetime
import numpy as np
import pandas as pd
import urllib3

#%% functions
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

def fetch_sequences(acc_list, dbase, outfile, chunksize):
    chunks = acc_slicer(acc_list, chunksize)
    with open(outfile, 'wb') as out_handle:
        for chunk in chunks:
            if dbase == 'NCBI':
                fetch_ncbi(chunk, out_handle)
            elif dbase == 'ENA':
                fetch_ena(chunk, out_handle)
            elif dbase == 'BOLD':
                fetch_bold(chunk, out_handle)
    return

#%% Main
acc_file = '/home/hernan/PROYECTOS/Graboid/Databases/Acc_lists/accessions_20-9-2021_15-31-56.csv'
def main(acc_file):
    # read acc_table
    # for tax/mark/db download file
    acc_tab = pd.read_csv(acc_file, index_col = 0)
    
    taxons = acc_tab['Taxon'].unique()
    markers = acc_tab['Marker'].unique()
    databases = acc_tab['Database'].unique()

    for taxon in taxons:
        for marker in markers:
            for database in databases:
                sub_tab = acc_tab.loc[(acc_tab['Taxon'] == taxon) & (acc_tab['Marker'] == marker) & (acc_tab['Database'] == database)]
            
                if sub_tab.shape[0] > 0:
                    outfile = generate_filename(taxon, marker, database)
                    fetch_sequences(sub_tab['Accession'].unique(), database, f'Seq_files/{outfile}', 500)
    
    return